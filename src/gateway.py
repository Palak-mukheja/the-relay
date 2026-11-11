
import asyncio
import random
import re
import time

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Request:
    """One request waiting to pass through the gateway."""

    fn: Callable
    estimated_tokens: int
    submitted_at: float
    deadline: float
    future: asyncio.Future


class Gateway:

    def __init__(
        self,
        max_concurrency=4,
        queue_size=50,
        tpm=8000,
        deadline_seconds=15,
        retry_budget_fraction=0.10,
        max_retries=1,
        base_backoff=0.5,
        max_backoff=5.0,
        jitter=0.25,
        honor_retry_after=True
    ):

        if max_concurrency < 1 or queue_size < 1 or tpm < 1:
            raise ValueError("Concurrency, queue size and TPM must be positive.")

        self.max_concurrency = max_concurrency
        self.queue_size = queue_size
        self.tpm = tpm
        self.deadline_seconds = deadline_seconds

        self.retry_budget_enabled = retry_budget_fraction is not None

        self.retry_budget_fraction = (
            retry_budget_fraction
            if retry_budget_fraction is not None
            else 0.0
        )

        self.max_retries = max_retries
        self.honor_retry_after = honor_retry_after

        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

        # Bounded queue: requests cannot accumulate indefinitely.
        self.queue = asyncio.Queue(maxsize=queue_size)

        # Token bucket.
        # It starts with one minute's worth of available tokens.
        self.token_capacity = float(tpm)
        self.available_tokens = float(tpm)
        self.last_refill = time.monotonic()

        # Retry credits accumulate as requests are admitted.
        self.retry_credits = 0.0

        self.workers = []
        self.running = False
        self.closing = False

        self.queue_peak = 0
        self.accepted_requests = 0

    async def start(self):
        """Start the fixed number of request workers."""

        if self.running:
            return

        if self.closing:
            raise RuntimeError("Gateway is already closing.")

        self.running = True

        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.max_concurrency)
        ]

    async def close(self):
        """Finish queued work and stop workers."""

        if not self.running:
            return

        self.closing = True

        await self.queue.join()

        for _ in self.workers:
            await self.queue.put(None)

        await asyncio.gather(*self.workers)

        self.workers.clear()
        self.running = False

    async def submit(
        self,
        fn,
        estimated_tokens=2500,
        request_timeout=None
    ):
        """
        Submit one synchronous request function.

        The function should return a dictionary containing at least
        a status, and preferably tokens_in and tokens_out.
        """

        if not self.running or self.closing:
            raise RuntimeError("Start the gateway before submitting requests.")

        if estimated_tokens < 1:
            raise ValueError("estimated_tokens must be positive.")

        submitted = time.monotonic()

        timeout = (
            request_timeout
            if request_timeout is not None
            else self.deadline_seconds
        )

        deadline = submitted + timeout

        future = asyncio.get_running_loop().create_future()

        job = Request(
            fn=fn,
            estimated_tokens=estimated_tokens,
            submitted_at=submitted,
            deadline=deadline,
            future=future
        )

        try:
            self.queue.put_nowait(job)

        except asyncio.QueueFull:
            # Shed immediately instead of allowing an unlimited queue.
            return {
                "status": "timeout",
                "infra_status": "timeout",
                "detail": "Request shed because the gateway queue is full.",
                "queue_wait_ms": 0.0,
                "work_ms": 0.0,
                "retry_wait_ms": 0.0,
                "total_ms": 0.0,
                "gateway_retries": 0,
                "gateway_attempts": 0,
                "queue_peak": self.queue_peak
            }

        self.accepted_requests += 1
        if self.retry_budget_enabled:
            self.retry_credits += self.retry_budget_fraction

        self.queue_peak = max(
            self.queue_peak,
            self.queue.qsize()
        )

        return await future

    def _refill_tokens(self):
        """Continuously refill the token bucket at TPM / 60."""

        now = time.monotonic()

        elapsed = now - self.last_refill

        refill_rate = self.tpm / 60.0

        self.available_tokens = min(
            self.token_capacity,
            self.available_tokens + elapsed * refill_rate
        )

        self.last_refill = now

    async def _reserve_tokens(self, amount, deadline):
        """
        Reserve estimated tokens before allowing a request through.

        Returns how long the request waited for token capacity.
        Returns None if the deadline would be exceeded.
        """

        started = time.monotonic()

        while True:

            self._refill_tokens()

            if self.available_tokens >= amount:
                self.available_tokens -= amount

                return (time.monotonic() - started) * 1000

            deficit = amount - self.available_tokens

            refill_rate = self.tpm / 60.0

            wait_seconds = deficit / refill_rate

            remaining = deadline - time.monotonic()

            if remaining <= 0 or wait_seconds > remaining:
                return None

            await asyncio.sleep(wait_seconds)

    def _adjust_tokens(self, reserved, actual):
        """Correct the estimate using actual provider token usage."""

        if actual is None:
            return

        difference = reserved - actual

        self._refill_tokens()

        self.available_tokens = min(
            self.token_capacity,
            self.available_tokens + difference
        )

    @staticmethod
    def _error_text(result):
        return str(
            result.get("error_message")
            or result.get("detail")
            or result.get("reply")
            or ""
        )

    @classmethod
    def _is_retryable(cls, result):

        text = cls._error_text(result).lower()
        # Daily exhaustion is permanent for this budget window.
        # Retrying cannot restore the daily token allowance.
        if (
            "daily_budget_exhausted" in text
            or "daily token budget exhausted" in text
        ):
            return False

        status_code = result.get("status_code")

        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None

        return (
            status_code in (429, 502, 503, 504)
            or any(code in text for code in (
                "http 429", "http 502", "http 503", "http 504"
            ))
            or "rate limit" in text
        )

    def _adapt_tpm_from_error(self, result):
        """Update token pacing when the provider advertises a new limit."""

        text = self._error_text(result)

        match = re.search(
            r"x-ratelimit-limit-tokens['\"]?\s*:\s*['\"]?(\d+)",
            text,
            re.IGNORECASE
        )

        if not match:
            return

        new_tpm = int(match.group(1))

        if new_tpm <= 0:
            return

        if new_tpm != self.tpm:

            self._refill_tokens()

            self.tpm = new_tpm
            self.token_capacity = float(new_tpm)

            self.available_tokens = min(
                self.available_tokens,
                self.token_capacity
            )

    @classmethod
    def _retry_after(cls, result):

        text = cls._error_text(result)

        patterns = [
            r"retry-after[=: ]+([\d.]+)",
            r"try again in\s+([\d.]+)\s*(?:s|seconds)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                return float(match.group(1))

        return 0.0

    async def _worker(self):

        while True:

            job = await self.queue.get()

            if job is None:
                self.queue.task_done()
                break

            result = await self._process(job)

            if not job.future.done():
                job.future.set_result(result)

            self.queue.task_done()

    async def _process(self, job):

        queue_wait_ms = 0.0
        work_ms = 0.0
        retry_wait_ms = 0.0

        retries = 0
        attempts = 0

        reserved_tokens = job.estimated_tokens
        result = {}

        # Time spent waiting in the queue.
        queue_wait_ms += (
            time.monotonic() - job.submitted_at
        ) * 1000

        while True:

            if time.monotonic() >= job.deadline:

                result = {
                    "status": "timeout",
                    "error_message": "Request deadline exceeded."
                }

                break

            token_wait = await self._reserve_tokens(
                reserved_tokens,
                job.deadline
            )

            if token_wait is None:

                result = {
                    "status": "timeout",
                    "error_message": "Token budget wait exceeded deadline."
                }

                break

            if attempts == 0:
                queue_wait_ms += token_wait
            else:
                retry_wait_ms += token_wait

            attempts += 1

            remaining = job.deadline - time.monotonic()

            work_started = time.monotonic()

            try:

                result = await asyncio.wait_for(
                    asyncio.to_thread(job.fn),
                    timeout=max(0.001, remaining)
                )

                if not isinstance(result, dict):
                    result = {
                        "status": "error",
                        "error_message": "Request function must return a dictionary."
                    }

            except asyncio.TimeoutError:

                result = {
                    "status": "timeout",
                    "error_message": "Request exceeded its deadline."
                }

            except Exception as exc:

                result = {
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}"
                }

            finally:

                work_ms += (
                    time.monotonic() - work_started
                ) * 1000

            actual_tokens = None

            if (
                result.get("tokens_in") is not None
                and result.get("tokens_out") is not None
            ):
                actual_tokens = (
                    result["tokens_in"] + result["tokens_out"]
                )

            error_text = self._error_text(result).lower()

            # A rejected 429 request did not consume its reservation.
            if "http 429" in error_text or "rate limit" in error_text:
                self._adjust_tokens(reserved_tokens, 0)

            elif actual_tokens is not None:
                self._adjust_tokens(
                    reserved_tokens,
                    actual_tokens
                )

            # Adapt if the service advertises a changed TPM limit.
            self._adapt_tpm_from_error(result)

            retryable = self._is_retryable(result)

            if not retryable:
                break

            if retries >= self.max_retries:
                break

            if self.retry_budget_enabled and self.retry_credits < 1:
                break

            retry_after = self._retry_after(result)

            exponential = min(
                self.max_backoff,
                self.base_backoff * (2 ** retries)
            )

            backoff_delay = (
                exponential + random.uniform(0, self.jitter)
            )

            if self.honor_retry_after:
                delay = max(retry_after, backoff_delay)
            else:
                delay = backoff_delay

            remaining = job.deadline - time.monotonic()

            if delay + 0.05 >= remaining:
                break

            # Spend one global retry credit.
            if self.retry_budget_enabled:
                self.retry_credits -= 1

            retries += 1

            retry_started = time.monotonic()

            await asyncio.sleep(delay)

            retry_wait_ms += (
                time.monotonic() - retry_started
            ) * 1000

        # Map final result to infrastructure status.
        original_status = result.get("status")

        if original_status == "timeout":
            infra_status = "timeout"

        elif (
            "daily_budget_exhausted" in self._error_text(result).lower()
            or "daily token budget exhausted" in self._error_text(result).lower()
        ):
            infra_status = "rate_limited"

        elif self._is_retryable(result):
            text = self._error_text(result).lower()

            if "429" in text or "rate limit" in text:
                infra_status = "rate_limited"
            else:
                infra_status = "error"

        elif original_status == "error":
            infra_status = "error"

        else:
            infra_status = "ok"

        total_ms = (
            queue_wait_ms
            + work_ms
            + retry_wait_ms
        )

        result.update({
            "infra_status": infra_status,
            "queue_wait_ms": queue_wait_ms,
            "work_ms": work_ms,
            "retry_wait_ms": retry_wait_ms,
            "total_ms": total_ms,
            "gateway_attempts": attempts,
            "gateway_retries": retries,
            "queue_peak": self.queue_peak
        })

        return result
