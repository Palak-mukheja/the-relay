
import httpx
import json
import time

from datetime import datetime, timezone
from typing import Callable

from src.config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_COMPLETION_TOKENS,
    REASONING_EFFORT
)


class GroqClient:

    def __init__(self):

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Set it in your environment."
            )

        # Create an asynchronous HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    async def stream_chat(
    self,
    messages,
    model=DEFAULT_MODEL,
    on_text_chunk: Callable[[str], None] | None = None,
    max_completion_tokens=None
     ):

        # Record request start time
        start_time = time.perf_counter()

        start_timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        first_token_time = None

        answer_parts = []
        usage = None
        response_headers = {}

        # Prepare the API request
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
            "max_completion_tokens": (
            MAX_COMPLETION_TOKENS
            if max_completion_tokens is None
            else max_completion_tokens
           ),
            "reasoning_effort": REASONING_EFFORT
        }

        try:

            # Send request and receive streaming response
            async with self.client.stream(
                "POST",
                GROQ_API_URL,
                json=payload
            ) as response:

                status_code = response.status_code

                # Capture only safe rate-limit headers
                response_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.startswith("x-ratelimit-")
                    or key == "retry-after"
                }

                # Handle unsuccessful requests
                if status_code != 200:

                    error_body = (
                        await response.aread()
                    ).decode("utf-8", errors="replace")

                    elapsed = (
                        time.perf_counter() - start_time
                    ) * 1000

                    return {
                        "status": (
                            "rate_limited"
                            if status_code == 429
                            else "error"
                        ),
                        "status_code": status_code,
                        "start_time": start_timestamp,
                        "first_token_ms": None,
                        "total_ms": round(elapsed, 2),
                        "answer": "",
                        "tokens_in": None,
                        "tokens_out": None,
                        "reasoning_tokens": None,
                        "rate_limit_headers": response_headers,
                        "error": error_body[:500]
                    }

                # Process incoming SSE lines
                async for line in response.aiter_lines():

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    # End of stream
                    if data == "[DONE]":
                        break

                    if not data:
                        continue

                    chunk = json.loads(data)

                    # Collect token usage from final chunk
                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    # Read generated answer
                    for choice in chunk.get("choices", []):

                        delta = choice.get("delta", {})

                        text = delta.get("content")

                        # Ignore internal reasoning chunks.
                        # We measure the first visible answer text.
                        if text:

                            if first_token_time is None:

                                first_token_time = (
                                    time.perf_counter()
                                )

                            answer_parts.append(text)

                            # Optional callback for live display
                            if on_text_chunk:
                                on_text_chunk(text)

            # Calculate timing after stream ends
            end_time = time.perf_counter()

            total_ms = (
                end_time - start_time
            ) * 1000

            first_token_ms = None

            if first_token_time is not None:
                first_token_ms = (
                    first_token_time - start_time
                ) * 1000

            # Extract token usage
            tokens_in = None
            tokens_out = None
            reasoning_tokens = None

            if usage:

                tokens_in = usage.get("prompt_tokens")
                tokens_out = usage.get("completion_tokens")

                details = usage.get(
                    "completion_tokens_details", {}
                )

                reasoning_tokens = details.get(
                    "reasoning_tokens"
                )

            return {
                "status": "ok",
                "status_code": 200,
                "start_time": start_timestamp,
                "first_token_ms": (
                    round(first_token_ms, 2)
                    if first_token_ms is not None
                    else None
                ),
                "total_ms": round(total_ms, 2),
                "answer": "".join(answer_parts),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "reasoning_tokens": reasoning_tokens,
                "rate_limit_headers": response_headers,
                "error": None
            }

        except httpx.TimeoutException as exc:

            elapsed = (
                time.perf_counter() - start_time
            ) * 1000

            return {
                "status": "timeout",
                "status_code": None,
                "start_time": start_timestamp,
                "first_token_ms": None,
                "total_ms": round(elapsed, 2),
                "answer": "",
                "tokens_in": None,
                "tokens_out": None,
                "reasoning_tokens": None,
                "rate_limit_headers": response_headers,
                "error": str(exc)
            }

        except (httpx.RequestError, json.JSONDecodeError) as exc:

            elapsed = (
                time.perf_counter() - start_time
            ) * 1000

            return {
                "status": "error",
                "status_code": None,
                "start_time": start_timestamp,
                "first_token_ms": None,
                "total_ms": round(elapsed, 2),
                "answer": "",
                "tokens_in": None,
                "tokens_out": None,
                "reasoning_tokens": None,
                "rate_limit_headers": response_headers,
                "error": str(exc)
            }
