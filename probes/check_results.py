
import argparse
import csv
import json
import math

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = ROOT / "results/day3_scenarios.csv"
RUN_LOG = ROOT / "results/scenario_runs.jsonl"

REQUIRED_FIELDS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

VALID_STATUSES = {
    "ok",
    "rate_limited",
    "timeout",
    "error"
}

VALID_SOURCES = {
    "real",
    "stand-in"
}


def load_run_manifest():

    if not RUN_LOG.exists():
        return []

    records = []

    with open(RUN_LOG, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError:
                raise ValueError(
                    f"Invalid JSON in run manifest, line {line_number}"
                )

    return records


def check_results(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    errors = []
    warnings = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        fields = reader.fieldnames or []

        # 1. Check the mandatory columns and their order.
        if fields[:len(REQUIRED_FIELDS)] != REQUIRED_FIELDS:

            errors.append(
                "The first 12 CSV columns do not match "
                "the assignment's required order."
            )

        missing = [
            field for field in REQUIRED_FIELDS
            if field not in fields
        ]

        if missing:
            errors.append(f"Missing required columns: {missing}")

        rows = list(reader)

    print("\n========== CSV INTEGRITY CHECK ==========")

    print("File:", csv_path)
    print("Total rows:", len(rows))

    # 2. Check that request IDs are unique.
    request_ids = [
        row.get("request_id", "")
        for row in rows
    ]

    duplicates = [
        request_id
        for request_id, count in Counter(request_ids).items()
        if request_id and count > 1
    ]

    if duplicates:
        errors.append(
            f"Duplicate request IDs found: {len(duplicates)}"
        )

    if any(not request_id for request_id in request_ids):
        errors.append("One or more rows have an empty request ID.")

    # 3. Validate every request row.
    scenario_counts = Counter()
    status_counts = Counter()
    source_counts = Counter()

    latency_values = []
    token_values = []

    for index, row in enumerate(rows, start=2):

        scenario = row.get("scenario", "")
        status = row.get("status", "")
        source = row.get("source", "")

        scenario_counts[scenario] += 1
        status_counts[status] += 1
        source_counts[source] += 1

        if status not in VALID_STATUSES:
            errors.append(
                f"Row {index}: invalid status '{status}'"
            )

        if source not in VALID_SOURCES:
            errors.append(
                f"Row {index}: invalid source '{source}'"
            )

        # 4. Failed requests must not have a first-token time.
        first_token = row.get("first_token_ms", "").strip()

        if status != "ok" and first_token:
            errors.append(
                f"Row {index}: failed request has first_token_ms."
            )

        if status == "ok" and not first_token:
            errors.append(
                f"Row {index}: successful request lacks first_token_ms."
            )

        # 5. Check latency arithmetic.
        try:

            total = float(row["total_ms"])

            queue = float(row["queue_wait_ms"])
            work = float(row["work_ms"])
            retry = float(row["retry_wait_ms"])

            if not all(
                math.isfinite(value)
                for value in (total, queue, work, retry)
            ):
                raise ValueError("Non-finite timing value.")

            if min(total, queue, work, retry) < 0:
                raise ValueError("Negative timing value.")

            calculated = queue + work + retry

            if abs(total - calculated) > 2.0:
                errors.append(
                    f"Row {index}: timing components do not sum "
                    f"to total_ms (difference={abs(total-calculated):.3f} ms)."
                )

            if status == "ok" and first_token:

                ft = float(first_token)

                if not math.isfinite(ft) or ft < 0:
                    raise ValueError("Invalid first-token time.")

                if ft > total + 2.0:
                    errors.append(
                        f"Row {index}: first_token_ms exceeds total_ms."
                    )

                latency_values.append((ft, total))

        except (ValueError, TypeError, KeyError) as exc:
            errors.append(
                f"Row {index}: invalid timing data: {exc}"
            )

        # 6. Collect token usage.
        try:

            tin = int(row["tokens_in"])
            tout = int(row["tokens_out"])

            if tin < 0 or tout < 0:
                raise ValueError("Negative token count.")

            token_values.append(tin + tout)

        except (ValueError, TypeError, KeyError) as exc:
            errors.append(
                f"Row {index}: invalid token counts: {exc}"
            )

    # 7. Check the run manifest against actual CSV row counts.
    manifest = load_run_manifest()

    expected_counts = Counter()

    for run in manifest:
        expected_counts[run["scenario"]] += run["logged_rows"]

        if run["scheduled_requests"] != run["logged_rows"]:
            errors.append(
                f"{run['scenario']}: scheduled and logged row counts differ "
                f"in the run manifest."
            )

    if manifest:

        for scenario, expected in expected_counts.items():

            actual = scenario_counts.get(scenario, 0)

            if actual != expected:
                errors.append(
                    f"{scenario}: manifest expected {expected} rows, "
                    f"CSV contains {actual}."
                )

    else:
        warnings.append(
            "No run manifest found. Scheduled-versus-logged "
            "counts could not be verified."
        )

    # 8. Print results.
    print("\nRows by scenario:")

    for scenario, count in sorted(scenario_counts.items()):
        print(f"{scenario:22s} {count}")

    print("\nRows by status:")

    for status, count in sorted(status_counts.items()):
        print(f"{status:22s} {count}")

    print("\nRows by source:")

    for source, count in sorted(source_counts.items()):
        print(f"{source:22s} {count}")

    print("\nSuccessful requests:", len(latency_values))

    if token_values:
        average_tokens = sum(token_values) / len(token_values)
        print(f"Average tokens per request: {average_tokens:.2f}")

    if len(latency_values) >= 300:

        import numpy as np

        ttft = [x[0] for x in latency_values]
        total = [x[1] for x in latency_values]

        print("\nLatency percentiles (numpy linear method):")

        for name, values in (
            ("User-visible TTFT", ttft),
            ("Total latency", total)
        ):

            print(f"\n{name} (n={len(values)})")

            for percentile in (50, 95, 99):
                value = np.percentile(
                    values,
                    percentile,
                    method="linear"
                )

                print(f"p{percentile}: {value:.2f} ms")

    else:
        print(
            "\np99 not reported: fewer than 300 successful "
            "requests in the combined CSV."
        )

    print("\nWarnings:")

    for warning in warnings:
        print("WARNING:", warning)

    print("\nErrors:")

    if errors:
        for error in errors:
            print("ERROR:", error)

    else:
        print("None")

    print("\nFINAL RESULT:", "PASS" if not errors else "FAIL")

    return not errors


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        default=str(DEFAULT_CSV),
        help="CSV file to validate"
    )

    args = parser.parse_args()

    success = check_results(args.file)

    raise SystemExit(0 if success else 1)
