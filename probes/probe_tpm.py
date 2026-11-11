
import asyncio
import json
import uuid

from src.groq_client import GroqClient
from src.logger import Logger
from src.config import BASE_DIR


MODEL = "openai/gpt-oss-120b"

PROMPT = "Reply with exactly: OK"

TESTS = [
    ("low_output_limit", 10),
    ("high_output_limit", 2000)
]


async def main():

    output_file = (
        BASE_DIR / "results" / "tpm_probe_raw.csv"
    )

    logger = Logger(output_file)

    try:

        async with GroqClient() as groq:

            for scenario, token_limit in TESTS:

                print("\n" + "=" * 55)
                print("Scenario:", scenario)
                print("Maximum completion tokens:", token_limit)
                print("=" * 55)

                result = await groq.stream_chat(
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT
                        }
                    ],
                    model=MODEL,
                    max_completion_tokens=token_limit
                )

                headers = result["rate_limit_headers"]

                print("\nStatus:", result["status"])
                print("HTTP:", result["status_code"])

                print("Input tokens:", result["tokens_in"])
                print("Output tokens:", result["tokens_out"])
                print("Reasoning tokens:", result["reasoning_tokens"])

                print(
                    "Remaining TPM:",
                    headers.get("x-ratelimit-remaining-tokens")
                )

                print(
                    "TPM reset:",
                    headers.get("x-ratelimit-reset-tokens")
                )

                print("Error:", result["error"])

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

    print("\nTPM experiment completed.")
    print("Results saved in results/tpm_probe_raw.csv")


if __name__ == "__main__":
    asyncio.run(main())
