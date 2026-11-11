
import json
import math
import random
import threading
import time

from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = ROOT / "calibration.json"


class StandinState:

    def __init__(self, calibration_path=CALIBRATION_FILE, seed=42):

        self.lock = threading.Lock()
        self.rng = random.Random(seed)

        self.config = {
            "mode": "normal",

            # Scaled to 20x the measured real limits.
            "rpm_limit": 600,
            "tpm_limit": 160000,
            "daily_token_budget": 4000000,

            "max_concurrency": 100,

            "slow_delay_ms": 2000,
            "failure_probability": 0.0,

            "load_slowdown_threshold": 8,
            "load_slowdown_ms": 100,

            # None means use the calibration file.
            "base_ttft_ms": None,
            "base_total_ms": None
        }

        self.request_times = deque()
        self.token_events = deque()

        self.daily_tokens = 0
        self.request_count = 0
        self.inflight = 0

        self.ttft_distribution, self.total_distribution = (
            self._load_calibration(calibration_path)
        )

    # -------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------

    def _load_calibration(self, path):

        # Fallback values from our measured GPT-OSS 120B data.
        ttft = {"p50": 477.50, "p95": 1161.87}
        total = {"p50": 485.66, "p95": 1168.37}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            models = data.get("models", {})

            selected = {}

            if isinstance(models, dict):
                for name, values in models.items():
                    if "120b" in name.lower():
                        selected = values
                        break

            if not selected:
                selected = data

            def find_distribution(obj, names):

                if not isinstance(obj, dict):
                    return None

                for key, value in obj.items():

                    if (
                        any(name in key.lower() for name in names)
                        and isinstance(value, dict)
                        and ("p50" in value or "log_mu" in value)
                    ):
                        return value

                    result = find_distribution(value, names)

                    if result:
                        return result

                return None

            found_ttft = find_distribution(
                selected, ("ttft", "first_token")
            )

            found_total = find_distribution(
                selected, ("total_ms", "total_latency")
            )

            if found_ttft:
                ttft = found_ttft

            if found_total:
                total = found_total

            print("Loaded calibration data:", path)

        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            print("Using fallback calibration values.")

        return ttft, total

    def _sample(self, distribution):

        if "log_mu" in distribution and "log_sigma" in distribution:
            mu = float(distribution["log_mu"])
            sigma = float(distribution["log_sigma"])

        else:
            p50 = float(distribution["p50"])
            p95 = float(distribution["p95"])

            mu = math.log(max(p50, 1))
            sigma = (
                math.log(max(p95, 1)) - mu
            ) / 1.6448536269514722

        return self.rng.lognormvariate(mu, max(0, sigma))

    def sample_latency(self):

        with self.lock:

            ttft = (
                self.config["base_ttft_ms"]
                if self.config["base_ttft_ms"] is not None
                else self._sample(self.ttft_distribution)
            )

            total = (
                self.config["base_total_ms"]
                if self.config["base_total_ms"] is not None
                else self._sample(self.total_distribution)
            )

            extra_load = max(
                0,
                self.inflight - self.config["load_slowdown_threshold"]
            ) * self.config["load_slowdown_ms"]

            if self.config["mode"] == "slow":
                extra_load += self.config["slow_delay_ms"]

            ttft += extra_load
            total = max(total + extra_load, ttft + 8)

            return ttft, total

    # -------------------------------------------------
    # TOKEN COUNTING / LIMITS
    # -------------------------------------------------

    def _prune(self, now):

        while self.request_times and now - self.request_times[0] >= 60:
            self.request_times.popleft()

        while self.token_events and now - self.token_events[0][0] >= 60:
            self.token_events.popleft()

    def _token_usage(self):
        return sum(tokens for _, tokens in self.token_events)

    def _retry_after(self, required_tokens=0, rpm=False):

        now = time.monotonic()

        if rpm and self.request_times:
            return max(
                1,
                60 - (now - self.request_times[0])
            )

        if required_tokens and self.token_events:

            total = self._token_usage()
            needed = total + required_tokens - self.config["tpm_limit"]
            freed = 0

            for timestamp, tokens in self.token_events:
                freed += tokens

                if freed >= needed:
                    return max(1, 60 - (now - timestamp))

        return 60

    def rate_headers(self):

        with self.lock:

            now = time.monotonic()
            self._prune(now)

            return {
                "x-ratelimit-limit-requests": str(
                    self.config["rpm_limit"]
                ),
                "x-ratelimit-remaining-requests": str(
                    max(0, self.config["rpm_limit"] - len(self.request_times))
                ),
                "x-ratelimit-limit-tokens": str(
                    self.config["tpm_limit"]
                ),
                "x-ratelimit-remaining-tokens": str(
                    max(
                        0,
                        self.config["tpm_limit"] - self._token_usage()
                    )
                ),
                "x-ratelimit-remaining-tokens-day": str(
                    max(
                        0,
                        self.config["daily_token_budget"] - self.daily_tokens
                    )
                )
            }

    def admit(self, reserved_tokens):

        with self.lock:

            now = time.monotonic()

            self.request_count += 1
            self._prune(now)

            token_usage = self._token_usage()

            if (
                self.daily_tokens + reserved_tokens
                > self.config["daily_token_budget"]
            ):
                return {
                    "allowed": False,
                    "code": "daily_budget_exhausted",
                    "message": "Stand-in daily token budget exhausted.",
                    "retry_after": None
                }

            if len(self.request_times) >= self.config["rpm_limit"]:
                return {
                    "allowed": False,
                    "code": "rpm_limit",
                    "message": "Stand-in requests-per-minute limit reached.",
                    "retry_after": self._retry_after(rpm=True)
                }

            if token_usage + reserved_tokens > self.config["tpm_limit"]:
                return {
                    "allowed": False,
                    "code": "tpm_limit",
                    "message": "Stand-in tokens-per-minute limit reached.",
                    "retry_after": self._retry_after(
                        required_tokens=reserved_tokens
                    )
                }

            if self.inflight >= self.config["max_concurrency"]:
                return {
                    "allowed": False,
                    "code": "overloaded",
                    "message": "Stand-in concurrency limit reached.",
                    "retry_after": 1
                }

            self.request_times.append(now)
            self.token_events.append((now, reserved_tokens))

            self.daily_tokens += reserved_tokens
            self.inflight += 1

            return {"allowed": True}

    def release(self):

        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def should_fail(self):

        with self.lock:
            return (
                self.config["mode"] == "fail"
                or self.rng.random() < self.config["failure_probability"]
            )

    # -------------------------------------------------
    # ADMIN CONTROLS
    # -------------------------------------------------

    def update_config(self, updates):

        allowed = set(self.config.keys())

        unknown = set(updates.keys()) - allowed

        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")

        with self.lock:

            for key, value in updates.items():

                if key == "mode":
                    if value not in ("normal", "slow", "fail"):
                        raise ValueError("Invalid mode.")

                elif value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"{key} must be numeric.")

                    if value < 0:
                        raise ValueError(f"{key} cannot be negative.")

                self.config[key] = value

        return self.snapshot()

    def reset(self, seed=42):

        with self.lock:

            self.request_times.clear()
            self.token_events.clear()

            self.daily_tokens = 0
            self.request_count = 0
            self.inflight = 0

            self.rng = random.Random(seed)

        return self.snapshot()

    def snapshot(self):

        with self.lock:

            self._prune(time.monotonic())

            return {
                "config": dict(self.config),
                "requests_seen": self.request_count,
                "tokens_per_minute_used": self._token_usage(),
                "daily_tokens_used": self.daily_tokens,
                "requests_in_current_minute": len(self.request_times),
                "inflight": self.inflight
            }
