
import asyncio
import json
import uuid

from src.groq_client import GroqClient
from src.logger import Logger
from src.config import BASE_DIR


MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]

TEST_ORDER = [
    MODELS[0],
    MODELS[1],
    MODELS[0],
    MODELS[1],
    MODELS[0],
    MODELS[1]
]

PROMPT = "Reply with exactly: OK"


async def main():

    logger = Logger(
        BASE_DIR / "results" / "model_budget_probe.csv"
    )

    observations = {
        model: [] for model in MODELS
    }

    try:

        async with GroqClient() as groq:

            for index, model in enumerate(TEST_ORDER, start=1):

                print(f"\nRequest {index}/6")
                print("Model:", model)

                result = await groq.stream_chat(
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT
                        }
                    ],
                    model=model,
                    max_completion_tokens=32
                )

                headers = result["rate_limit_headers"]

                observation = {
                    "remaining_requests": headers.get(
                        "x-ratelimit-remaining-requests"
                    ),
                    "remaining_tokens": headers.get(
                        "x-ratelimit-remaining-tokens"
                    ),
                    "tokens_in": result["tokens_in"],
                    "tokens_out": result["tokens_out"],
                    "status": result["status"]
                }

                observations[model].append(observation)

                print("Status:", result["status"])
                print(
                    "Remaining requests:",
                    observation["remaining_requests"]
                )
                print(
                    "Remaining tokens:",
                    observation["remaining_tokens"]
                )

                logger.log(
                    request_id=str(uuid.uuid4()),
                    scenario="model_budget_independence",
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
                    rate_limit_headers=json.dumps(headers),
                    error_message=result["error"] or ""
                )

                if result["status"] != "ok":
                    print("Stopping due to unsuccessful request.")
                    break

                await asyncio.sleep(1)

    finally:
        logger.close()

    print("\n========== BUDGET COMPARISON ==========")

    for model, entries in observations.items():

        print("\nModel:", model)

        for i in range(1, len(entries)):

            previous = entries[i - 1]
            current = entries[i]

            old = previous["remaining_requests"]
            new = current["remaining_requests"]

            if old is not None and new is not None:

                delta = int(old) - int(new)

                print(
                    f"RPD decrease between calls: {delta}"
                )

                if delta == 1:
                    print("Consistent with independent request budgets.")
                elif delta > 1:
                    print("Other calls may also be consuming this budget.")

    print("\nExperiment saved.")
    print("File: results/model_budget_probe.csv")


if __name__ == "__main__":
    asyncio.run(main())
