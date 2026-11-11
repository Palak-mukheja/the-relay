

import json
import re
import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "probe_raw.csv"
CONCURRENCY_FILE = RESULTS_DIR / "concurrency_raw.csv"
OUTPUT_FILE = BASE_DIR / "calibration.json"


def calculate_distribution(values):

    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[values > 0]

    if len(values) == 0:
        return None

    log_values = np.log(values)

    return {
        "sample_size": int(len(values)),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "p99_ms": round(float(np.percentile(values, 99)), 2),
        "log_mu": round(float(np.mean(log_values)), 6),
        "log_sigma": round(float(np.std(log_values)), 6)
    }


def calculate_token_speed(group):

    successful = group[group["status"] == "ok"].copy()

    total = pd.to_numeric(
        successful["total_ms"], errors="coerce"
    )

    ttft = pd.to_numeric(
        successful["first_token_ms"], errors="coerce"
    )

    output_tokens = pd.to_numeric(
        successful["tokens_out"], errors="coerce"
    )

    generation_time = (total - ttft) / 1000

    speeds = output_tokens / generation_time

    speeds = speeds.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    speeds = speeds[speeds > 0]

    if len(speeds) == 0:
        return None

    return {
        "sample_size": int(len(speeds)),
        "p50_tokens_per_second": round(
            float(np.percentile(speeds, 50)), 2
        ),
        "p95_tokens_per_second": round(
            float(np.percentile(speeds, 95)), 2
        ),
        "mean_tokens_per_second": round(
            float(speeds.mean()), 2
        ),
        "note": "Approximate completion-token throughput. "
                "Completion tokens may include reasoning tokens."
    }


# Load latency results
df = pd.read_csv(INPUT_FILE)

df = df[
    df["scenario"] == "probe_a_latency"
].copy()

if df.empty:
    raise ValueError("No Probe A data found.")

calibration = {
    "source": "Real Groq measurements",
    "models": {},
    "concurrency": {},
    "service_limits": {
        "rpm": {
            "value": 30,
            "verified_for": "openai/gpt-oss-120b"
        },
        "tpm": {
            "value": 8000,
            "source": "Console and response headers"
        },
        "rpd": {
            "value": 1000,
            "source": "Console and response headers"
        },
        "tpd": {
            "value": 200000,
            "source": "Console; not independently exhausted"
        }
    }
}


# Calculate per-model latency and throughput
for model, group in df.groupby("model"):

    successful = group[
        group["status"] == "ok"
    ]

    calibration["models"][model] = {

        "requests_attempted": int(len(group)),

        "requests_successful": int(len(successful)),

        "requests_failed": int(
            (group["status"] != "ok").sum()
        ),

        "ttft": calculate_distribution(
            successful["first_token_ms"]
        ),

        "total_time": calculate_distribution(
            successful["total_ms"]
        ),

        "tokens_per_second": calculate_token_speed(
            successful
        )
    }


# Load concurrency results
concurrency_df = pd.read_csv(CONCURRENCY_FILE)

for scenario, group in concurrency_df.groupby("scenario"):

    match = re.fullmatch(
        r"concurrency_(\d+)", scenario
    )

    if not match:
        continue

    level = int(match.group(1))

    successful = group[
        group["status"] == "ok"
    ]

    calibration["concurrency"][str(level)] = {

        "requests_attempted": int(len(group)),

        "requests_successful": int(len(successful)),

        "requests_failed": int(
            (group["status"] != "ok").sum()
        ),

        "ttft": calculate_distribution(
            successful["first_token_ms"]
        ),

        "total_time": calculate_distribution(
            successful["total_ms"]
        )
    }


# Find highest fully successful concurrency
fully_successful = [

    int(level)

    for level, values in calibration["concurrency"].items()

    if (
        values["requests_attempted"] == int(level)
        and values["requests_successful"] == int(level)
    )
]

calibration["concurrency_summary"] = {
    "highest_successful_tested_level": (
        max(fully_successful)
        if fully_successful
        else None
    ),
    "actual_concurrency_limit": "Unknown"
}


# Save calibration file
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        calibration,
        file,
        indent=4
    )


# Print summary
print("\n========== CALIBRATION GENERATED ==========")

for model, values in calibration["models"].items():

    print(f"\nModel: {model}")

    print("Successful:", values["requests_successful"])

    print(
        "TTFT p50:",
        values["ttft"]["p50_ms"], "ms"
    )

    print(
        "TTFT p95:",
        values["ttft"]["p95_ms"], "ms"
    )

    print(
        "Total p50:",
        values["total_time"]["p50_ms"], "ms"
    )

    print(
        "Total p95:",
        values["total_time"]["p95_ms"], "ms"
    )

    print(
        "Estimated tokens/sec:",
        values["tokens_per_second"]["p50_tokens_per_second"]
    )


print("\nConcurrency findings:")

for level, values in calibration["concurrency"].items():

    print(
        f"{level} concurrent requests: "
        f"{values['requests_successful']}/"
        f"{values['requests_attempted']} successful"
    )


print(
    "\nHighest successful concurrency:",
    calibration["concurrency_summary"][
        "highest_successful_tested_level"
    ]
)

print("\nSaved to:", OUTPUT_FILE)
