
import asyncio
import json
import uuid

from src.groq_client import GroqClient
from src.logger import Logger
from src.config import BASE_DIR


MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b"
]

REQUESTS_PER_MODEL = 15
DELAY_SECONDS = 1.5

PROMPT = (
    "Reply with exactly this string and nothing else: "
    "RELAY_PROBE_OK"
)


async def main():

    output_file = BASE_DIR / "results" / "probe_raw.csv"
    logger = Logger(output_file)

    try:

        async with GroqClient() as groq:

            for model in MODELS:

                print("\n" + "=" * 55)
                print(f"Testing model: {model}")
                print("=" * 55)

                successful = 0

                for i in range(1, REQUESTS_PER_MODEL + 1):

                    print(
                        f"\nRequest {i}/{REQUESTS_PER_MODEL}: ",
                        end="",
                        flush=True
                    )

                    result = await groq.stream_chat(
                        messages=[
                            {
                                "role": "user",
                                "content": PROMPT
                            }
                        ],
                        model=model
                    )

                    # Save every request, including failures
                    logger.log(
                        request_id=str(uuid.uuid4()),
                        scenario="probe_a_latency",
                        source="real",
                        model=model,
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
                        rate_limit_headers=json.dumps(
                            result["rate_limit_headers"]
                        ),
                        error_message=result["error"] or ""
                    )

                    print(
                        f"Status={result['status']} | "
                        f"TTFT={result['first_token_ms']} ms | "
                        f"Total={result['total_ms']} ms | "
                        f"Tokens={result['tokens_in']}/{result['tokens_out']}"
                    )

                    if result["status"] != "ok":

                        print(
                            "Stopping this model's probe because "
                            "the request failed. No automatic retries."
                        )
                        break

                    successful += 1

                    if i < REQUESTS_PER_MODEL:
                        await asyncio.sleep(DELAY_SECONDS)

                print(
                    f"\nSuccessful requests for {model}: "
                    f"{successful}/{REQUESTS_PER_MODEL}"
                )

                # Pause before moving to the next model
                await asyncio.sleep(2)

    finally:
        logger.close()

    print("\nProbe A completed.")
    print("Raw results saved in results/probe_raw.csv")


if __name__ == "__main__":
    asyncio.run(main())
