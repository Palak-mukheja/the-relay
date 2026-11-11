
import argparse
import asyncio
import csv
import json
import math
import time
import uuid

from datetime import datetime
from pathlib import Path

import httpx

from src.gateway import Gateway
from src.logger import Logger
from src.rag_core import (
    load_pages,
    make_chunks,
    Retriever,
    make_groq_llm,
    answer
)


ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = ROOT / "data/evaluation_questions.json"
BASELINE_FILE = ROOT / "results/standin_smoke_test.csv"

OUTPUT_FILE = ROOT / "results/day3_scenarios.csv"
RUN_LOG = ROOT / "results/scenario_runs.jsonl"

ADMIN_URL = "http://127.0.0.1:8000"
STANDIN_URL = ADMIN_URL + "/v1"

MODEL = "openai/gpt-oss-120b"

PAPER_FILES = {
    "Attention Is All You Need": ROOT / "data/papers/attention_v7.pdf",
    "Moshi": ROOT / "data/papers/moshi_v1.pdf",
    "Dynamic Chunking": ROOT / "data/papers/dynamic_chunking_v2.pdf"
}

# Stand-in limits: 40x the measured real limits.
BASE_RPM = 1200
BASE_TPM = 320000
BASE_TPD = 8000000

MAX_COMPLETION_TOKENS = 160

# Conservative gateway reservation.


GATEWAY_CONCURRENCY = 24
GATEWAY_QUEUE_SIZE = 200
REQUEST_DEADLINE = 15

DURATION = 300


SCENARIOS = (
    "steady",
    "burst",
    "overload_naive_retry",
    "overload",
    "limit_halved",
    "daily_exhausted"
)


def load_questions():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    if not questions:
        raise ValueError("Question dataset is empty.")

    return questions



def load_baseline():

    # Prefer actual measurements from the completed steady run.
    measured_file = ROOT / "results/day3_scenarios.csv"

    if measured_file.exists():

        with open(measured_file, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        rows = [
            row for row in rows
            if row.get("scenario") == "steady"
            and row.get("status") == "ok"
            and row.get("source") == "stand-in"
        ]

        if rows:

            input_tokens = [
                int(row["tokens_in"] or 0)
                for row in rows
            ]

            output_tokens = [
                int(row["tokens_out"] or 0)
                for row in rows
            ]

            avg_in = sum(input_tokens) / len(rows)
            avg_out = sum(output_tokens) / len(rows)

            avg_total = avg_in + avg_out

            return {
                "tokens_in": round(avg_in),
                "tokens_out": round(avg_out),
                "tokens_per_answer": avg_total,
                "reserved_per_answer": math.ceil(avg_total)
            }

    # Fallback: use the original smoke-test measurement.
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Neither steady results nor smoke-test baseline exists."
        )

    with open(BASELINE_FILE, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Smoke-test CSV has no data rows.")

    last = rows[-1]

    tokens_in = int(last["tokens_in"])
    tokens_out = int(last["tokens_out"])
    actual_tokens = tokens_in + tokens_out

    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_per_answer": actual_tokens,
        "reserved_per_answer": actual_tokens
    }


def segment_arrivals(start, duration, rate_per_minute):

    """
    Generate evenly spaced arrival times.

    The schedule is independent of request completion.
    """

    count = math.ceil(
        rate_per_minute * duration / 60
    )

    if count <= 0:
        return []

    return [
        (
            start + (i + 0.5) * duration / count,
            rate_per_minute
        )
        for i in range(count)
    ]


def build_schedules(capacity):
    half = 0.5 * capacity

    return {
        "steady": (
            300,
            segment_arrivals(0, 300, half)
        ),

        "burst": (
            300,
            segment_arrivals(0, 140, half)
            + segment_arrivals(140, 20, 4 * capacity)
            + segment_arrivals(160, 140, half)
        ),

        "overload": (
            300,
            segment_arrivals(0, 300, 2 * capacity)
        ),

        "overload_naive_retry": (
            300,
            segment_arrivals(0, 300, 2 * capacity)
        ),

        "limit_halved": (
            300,
            segment_arrivals(0, 300, 0.8 * capacity)
        ),

        "daily_exhausted": (
            360,
            segment_arrivals(0, 360, half)
        )
    }


def print_plan(schedules, capacity, baseline):

    print("\n========== EXPERIMENT PLAN ==========")

    print(f"Measured tokens per answer: "
          f"{baseline['tokens_per_answer']}")

    print(f"Measured input tokens: {baseline['tokens_in']}")
    print(f"Measured output tokens: {baseline['tokens_out']}")

    print(f"Stand-in TPM: {BASE_TPM:,}")
    print(f"Estimated capacity C: {capacity:.2f} requests/minute")

    print("\nScenario schedule:")

    for name, (duration, arrivals) in schedules.items():

        print(
            f"{name:20s} "
            f"duration={duration}s "
            f"scheduled_requests={len(arrivals)}"
        )

    print("\nNo requests have been sent.")
    print("Use --execute to begin the experiments.")


async def admin_request(client, method, endpoint, payload=None):

    response = await client.request(
        method,
        ADMIN_URL + endpoint,
        json=payload,
        timeout=15
    )

    response.raise_for_status()

    return response.json()


def reply_kind(result):

    status = result.get("status")

    if status in ("answered", "refused", "unverified"):
        return status

    return "unavailable"


async def run_one_request(
    index,
    offset,
    scenario,
    questions,
    retriever,
    llm,
    gateway,
    logger,
    seed,
    estimated_tokens,
    run_id,
    experiment_variant
):

    question_record = questions[index % len(questions)]

    question = question_record["question"]

    start_time = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )

    def process():

        return answer(
            question=question,
            retriever=retriever,
            llm=llm,
            allow_retry=False
        )

    try:

        result = await gateway.submit(
            fn=process,
            estimated_tokens=estimated_tokens,
            request_timeout=REQUEST_DEADLINE
        )

    except Exception as exc:

        result = {
            "status": "error",
            "infra_status": "error",
            "error_message": f"{type(exc).__name__}: {exc}",
            "queue_wait_ms": 0.0,
            "work_ms": 0.0,
            "retry_wait_ms": 0.0,
            "total_ms": 0.0,
            "gateway_retries": 0,
            "gateway_attempts": 0,
            "queue_peak": gateway.queue_peak
        }

    infra = result.get("infra_status", "error")

    queue_ms = result.get("queue_wait_ms", 0.0)
    work_ms = result.get("work_ms", 0.0)
    retry_ms = result.get("retry_wait_ms", 0.0)

    total_ms = queue_ms + work_ms + retry_ms

    model_ttft = result.get("model_first_token_ms")

    # The answer is held until citation validation finishes.
    visible_ttft = total_ms if infra == "ok" else None

    detail = (
        result.get("error_message")
        or result.get("detail")
        or ""
    )

    # Identify provider HTTP status, where present.
    status_code = None

    import re

    match = re.search(r"HTTP\s*(\d{3})", detail)

    if match:
        status_code = int(match.group(1))

    kind = reply_kind(result)

    # For the exhausted-budget experiment, report a degraded
    # response rather than pretending an answer was verified.
    if scenario == "daily_exhausted" and (
        "daily" in detail.lower()
        or "budget exhausted" in detail.lower()
    ):
        kind = "degraded"

    logger.log(
        request_id=str(uuid.uuid4()),
        scenario=scenario,
        source="stand-in",
        start_time=start_time,
        first_token_ms=visible_ttft,
        total_ms=total_ms,
        status=infra,
        retries=result.get("gateway_retries", 0),
        tokens_in=result.get("tokens_in") or 0,
        tokens_out=result.get("tokens_out") or 0,
        retrieval_setting="M",
        answer_grade="",

        model="stand-in",
        model_first_token_ms=model_ttft,
        status_code=status_code,
        detail=detail,
        reply_kind=kind,
        reasoning_tokens=result.get("reasoning_tokens"),
        rate_limit_headers=json.dumps(
            result.get("rate_limit_headers", {})
        ),
        error_message=detail,
        queue_wait_ms=queue_ms,
        work_ms=work_ms,
        retry_wait_ms=retry_ms,
        attempts=result.get("gateway_attempts", 0),
        retrieved_pages=json.dumps(
            result.get("retrieved_pages", [])
        ),
        reply_text=result.get("reply", ""),
        seed=seed,
        queue_peak=result.get("queue_peak", gateway.queue_peak)
    )

    return {
        "status": infra,
        "reply_kind": kind
    }


async def run_scenario(
    name,
    duration,
    arrivals,
    questions,
    retriever,
    llm,
    logger,
    baseline,
    seed
):

    print(f"\n========== {name.upper()} ==========")
    run_id = str(uuid.uuid4())

    experiment_variant = (
        "naive_retry"
        if name == "overload_naive_retry"
        else "budgeted_retry"
        if name == "overload"
        else "standard"
    )

    daily_budget = BASE_TPD

    if name == "daily_exhausted":

        # Aim to exhaust the budget near minute three.
        daily_budget = math.ceil(
            0.5
            * (BASE_TPM / baseline["tokens_per_answer"])
            * 3
            * baseline["reserved_per_answer"]
        )

    base_config = {
        "mode": "normal",
        "rpm_limit": BASE_RPM,
        "tpm_limit": BASE_TPM,
        "daily_token_budget": daily_budget,
        "max_concurrency": 100,
        "failure_probability": 0.0,
        "load_slowdown_threshold": 4,
        "load_slowdown_ms": 100
    }

    async with httpx.AsyncClient() as client:

        # Reset counters and set a reproducible random seed.
        await admin_request(
            client,
            "POST",
            "/admin/reset",
            {"seed": seed}
        )

        await admin_request(
            client,
            "PATCH",
            "/admin/config",
            base_config
        )

        naive = name == "overload_naive_retry"

        gateway = Gateway(
            max_concurrency=GATEWAY_CONCURRENCY,
            queue_size=GATEWAY_QUEUE_SIZE,
            tpm=BASE_TPM,
            deadline_seconds=REQUEST_DEADLINE,

            # Naive policy: no global retry budget.
            retry_budget_fraction=None if naive else 0.10,

            max_retries=3 if naive else 1,

            # Naive policy retries immediately.
            base_backoff=0.0 if naive else 0.5,
            max_backoff=0.0 if naive else 5.0,
            jitter=0.0 if naive else 0.25,

            honor_retry_after=not naive
        )

        await gateway.start()

        start = time.monotonic()

        async def dispatch():

            tasks = []

            for index, (offset, rate) in enumerate(
                sorted(arrivals, key=lambda x: x[0])
            ):

                target = start + offset

                await asyncio.sleep(
                    max(0, target - time.monotonic())
                )

                task = asyncio.create_task(
                    run_one_request(
                        index,
                        offset,
                        name,
                        questions,
                        retriever,
                        llm,
                        gateway,
                        logger,
                        seed,
                        baseline["reserved_per_answer"],
                        run_id,
                        experiment_variant

                    )
                )

                tasks.append(task)

            return await asyncio.gather(*tasks)

        async def halve_limits():

            await asyncio.sleep(120)

            print("\nADMIN ACTION: halving RPM and TPM.")

            await admin_request(
                client,
                "PATCH",
                "/admin/config",
                {
                    "rpm_limit": BASE_RPM // 2,
                    "tpm_limit": BASE_TPM // 2
                }
            )

        try:

            if name == "limit_halved":
                results, _ = await asyncio.gather(
                    dispatch(),
                    halve_limits()
                )
            else:
                results = await dispatch()

        finally:
            await gateway.close()

        elapsed = time.monotonic() - start

        state = await admin_request(
            client,
            "GET",
            "/admin/state"
        )

    statuses = {}

    for result in results:
        statuses[result["status"]] = (
            statuses.get(result["status"], 0) + 1
        )

    run_record = {
        "run_id": run_id,
        "experiment_variant": experiment_variant,
        "retry_policy": (
        "immediate_unbounded"
     if name == "overload_naive_retry"
     else "retry_after_backoff_budgeted"
     if name == "overload"
     else "standard"
    ),
        "scheduled_arrivals": "open_loop",
        "scenario": name,
        "seed": seed,
        "scheduled_requests": len(arrivals),
        "logged_rows": len(results),
        "duration_seconds": round(elapsed, 3),
        "offered_requests_per_minute": round(
            len(arrivals) / duration * 60,
            3
        ),
        "capacity_requests_per_minute": round(
            BASE_TPM / baseline["tokens_per_answer"],
            3
        ),
        "tokens_per_answer": baseline["tokens_per_answer"],
        "daily_budget": daily_budget,
        "status_counts": statuses,
        "queue_peak": gateway.queue_peak,
        "standin_final_state": state
    }

    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record) + "\n")

    print(f"Requests scheduled: {len(arrivals)}")
    print(f"Rows produced:      {len(results)}")
    print(f"Duration:           {elapsed:.1f}s")
    print(f"Peak queue:         {gateway.queue_peak}")
    print(f"Status counts:      {statuses}")

    if len(results) != len(arrivals):
        raise RuntimeError(
            f"Row mismatch in {name}: "
            f"{len(results)} scheduled, {len(arrivals)} results."
        )


async def execute(selected):

    baseline = load_baseline()

    capacity = BASE_TPM / baseline["tokens_per_answer"]

    schedules = build_schedules(capacity)

    if selected == "all":
        names = list(SCENARIOS)
    else:
        names = [selected]

    questions = load_questions()

    papers = {
        name: load_pages(str(path))
        for name, path in PAPER_FILES.items()
    }

    chunks = make_chunks(
        papers,
        size_words=400,
        overlap_words=80
    )

    retriever = Retriever(chunks)

    llm = make_groq_llm(
        model=MODEL,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        base_url=STANDIN_URL,
        use_env_key=False
    )

    print(f"\nPapers: {len(papers)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")
    print(f"Capacity C: {capacity:.2f} requests/minute")

    logger = Logger(OUTPUT_FILE)

    try:

        for run_number, name in enumerate(names, start=1):

            duration, arrivals = schedules[name]

            await run_scenario(
                name,
                duration,
                arrivals,
                questions,
                retriever,
                llm,
                logger,
                baseline,
                seed=42 + run_number
            )

    finally:
        logger.close()

    print("\nAll selected scenarios finished.")
    print(f"Raw results: {OUTPUT_FILE}")
    print(f"Run manifest: {RUN_LOG}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print request counts without sending requests."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the selected scenario."
    )

    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS) + ["all"],
        default="all"
    )

    args = parser.parse_args()

    baseline = load_baseline()

    capacity = BASE_TPM / baseline["tokens_per_answer"]

    schedules = build_schedules(capacity)

    if args.execute:
        asyncio.run(execute(args.scenario))
    else:
        print_plan(schedules, capacity, baseline)


if __name__ == "__main__":
    main()
