
import asyncio
import json
import uuid
import time

from src.groq_client import GroqClient
from src.logger import Logger
from src.config import BASE_DIR


MODEL = "openai/gpt-oss-120b"

TARGET_REQUESTS = 40
BATCH_SIZE = 4
MAX_RECOVERY_POLLS = 65

PROMPT = "Reply with exactly: OK"


async def send_request(groq, logger, scenario, number):

    result = await groq.stream_chat(
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        model=MODEL
    )

    headers = result["rate_limit_headers"]

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

    print(
        f"Request {number}: "
        f"{result['status']} | "
        f"HTTP {result['status_code']} | "
        f"Total: {result['total_ms']} ms"
    )

    return result


async def main():

    output_file = (
        BASE_DIR / "results" / "rpm_probe_raw.csv"
    )

    logger = Logger(output_file)

    total_attempts = 0
    successful_before_limit = 0
    rate_limit_result = None

    try:

        async with GroqClient() as groq:

            print("\nStarting RPM probe...")
            print("Model:", MODEL)

            probe_start = time.perf_counter()

            # Send requests in small concurrent batches
            while total_attempts < TARGET_REQUESTS:

                batch_size = min(
                    BATCH_SIZE,
                    TARGET_REQUESTS - total_attempts
                )

                tasks = [
                    send_request(
                        groq,
                        logger,
                        "rpm_probe",
                        total_attempts + i + 1
                    )
                    for i in range(batch_size)
                ]

                results = await asyncio.gather(*tasks)

                total_attempts += batch_size

                for result in results:

                    if result["status"] == "ok":
                        successful_before_limit += 1

                    elif result["status"] == "rate_limited":
                        rate_limit_result = result

                if any(
                    r["status"] != "ok"
                    for r in results
                ):
                    print("\nA request failed. Stopping initial probe.")
                    break

            probe_duration = (
                time.perf_counter() - probe_start
            )

            print("\nInitial probe duration:", round(
                probe_duration, 2
            ), "seconds")

            print(
                "Successful requests before stop:",
                successful_before_limit
            )

            # Only poll for recovery after an actual HTTP 429
            if rate_limit_result:

                print("\nRATE LIMIT DETECTED!")

                print(
                    "Retry-After:",
                    rate_limit_result["rate_limit_headers"].get(
                        "retry-after"
                    )
                )

                recovery_start = time.perf_counter()

                print("\nChecking recovery once per second...")

                for poll in range(1, MAX_RECOVERY_POLLS + 1):

                    await asyncio.sleep(1)

                    result = await send_request(
                        groq,
                        logger,
                        "rpm_recovery",
                        poll
                    )

                    if result["status"] == "ok":

                        recovery_time = (
                            time.perf_counter()
                            - recovery_start
                        )

                        print("\nAPI RECOVERED!")

                        print(
                            "Recovery time:",
                            round(recovery_time, 2),
                            "seconds"
                        )

                        break

                    elif result["status"] != "rate_limited":

                        print(
                            "Unexpected error during recovery. "
                            "Stopping."
                        )
                        break

                else:
                    print(
                        "Recovery not observed within polling window."
                    )

            else:

                print(
                    "\nNo HTTP 429 was observed. "
                    "Do not claim the RPM limit was reached."
                )

    finally:

        logger.close()

    print("\nRPM probe finished.")
    print("Results: results/rpm_probe_raw.csv")


if __name__ == "__main__":
    asyncio.run(main())
