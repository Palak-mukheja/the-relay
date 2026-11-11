
import asyncio
import json
import uuid

from src.groq_client import GroqClient
from src.logger import Logger
from src.config import BASE_DIR


MODEL = "openai/gpt-oss-120b"

TESTS = [
    ("isolated_low", 10),
    ("isolated_high", 2000)
]


async def main():

    logger = Logger(
        BASE_DIR / "results" / "tpm_controlled.csv"
    )

    try:
        async with GroqClient() as groq:

            for scenario, token_limit in TESTS:

                print("\nWaiting 65 seconds for a fresh TPM window...")
                await asyncio.sleep(65)

                print("\nTesting:", scenario)
                print("Maximum completion tokens:", token_limit)

                result = await groq.stream_chat(
                    messages=[
                        {
                            "role": "user",
                            "content": "Reply with exactly: OK"
                        }
                    ],
                    model=MODEL,
                    max_completion_tokens=token_limit
                )

                headers = result["rate_limit_headers"]

                limit = headers.get(
                    "x-ratelimit-limit-tokens"
                )

                remaining = headers.get(
                    "x-ratelimit-remaining-tokens"
                )

                print("Status:", result["status"])
                print("Input tokens:", result["tokens_in"])
                print("Output tokens:", result["tokens_out"])
                print("Reasoning tokens:", result["reasoning_tokens"])
                print("TPM limit:", limit)
                print("Remaining TPM:", remaining)
                print("Reset:", headers.get(
                    "x-ratelimit-reset-tokens"
                ))

                if limit and remaining:
                    print(
                        "Header-implied tokens used:",
                        int(limit) - int(remaining)
                    )

                logger.log(
                    request_id=str(uuid.uuid4()),
                    scenario=scenario,
                    source="real",
                    model=MODEL,
                    start_time=result["start_time"],
                    first_token_ms=result["first_token_ms"],
                    total_ms=result["total_ms"],
                    status=result["status"],
                    status_code=result["status_code"],
                    retries=0,
                    tokens_in=result["tokens_in"],
                    tokens_out=result["tokens_out"],
                    reasoning_tokens=result["reasoning_tokens"],
                    retrieval_setting="not_applicable",
                    answer_grade="",
                    rate_limit_headers=json.dumps(headers),
                    error_message=result["error"] or ""
                )

    finally:
        logger.close()

    print("\nControlled TPM experiment completed.")


if __name__ == "__main__":
    asyncio.run(main())
