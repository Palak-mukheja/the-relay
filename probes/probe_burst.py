
import asyncio
import json
import uuid

from src.groq_client import GroqClient
from src.logger import Logger
from src.config import BASE_DIR


MODEL = "openai/gpt-oss-120b"

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# Space batches to avoid accidentally hitting RPM
PAUSE_BETWEEN_BATCHES = 65

PROMPT = (
    "Reply with exactly this string and nothing else: "
    "RELAY_PROBE_OK"
)


async def send_request(groq, logger, batch_size, request_number):

    result = await groq.stream_chat(
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        model=MODEL
    )

    logger.log(
        request_id=str(uuid.uuid4()),
        scenario=f"concurrency_{batch_size}",
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
        rate_limit_headers=json.dumps(
            result["rate_limit_headers"]
        ),
        error_message=result["error"] or ""
    )

    print(
        f"Request {request_number}: "
        f"Status={result['status']} | "
        f"TTFT={result['first_token_ms']} ms | "
        f"Total={result['total_ms']} ms"
    )

    return result


async def main():

    output_file = (
        BASE_DIR / "results" / "concurrency_raw.csv"
    )

    logger = Logger(output_file)

    try:

        async with GroqClient() as groq:

            for level in CONCURRENCY_LEVELS:

                print("\n" + "=" * 55)
                print(f"CONCURRENCY LEVEL: {level}")
                print("=" * 55)

                # All requests in this batch start concurrently
                tasks = [
                    send_request(
                        groq,
                        logger,
                        level,
                        i + 1
                    )
                    for i in range(level)
                ]

                results = await asyncio.gather(*tasks)

                successes = sum(
                    r["status"] == "ok"
                    for r in results
                )

                failures = len(results) - successes

                print(
                    f"\nBatch completed: "
                    f"{successes}/{level} successful"
                )

                if failures:
                    print(
                        "A failure occurred. "
                        "Stopping further concurrency tests."
                    )
                    break

                if level != CONCURRENCY_LEVELS[-1]:

                    print(
                        f"\nWaiting {PAUSE_BETWEEN_BATCHES} seconds..."
                    )

                    await asyncio.sleep(
                        PAUSE_BETWEEN_BATCHES
                    )

    finally:
        logger.close()

    print("\nConcurrency experiment completed.")
    print("Results saved in results/concurrency_raw.csv")


if __name__ == "__main__":
    asyncio.run(main())
