
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import BASE_DIR

FILE = BASE_DIR / "results" / "probe_raw.csv"

df = pd.read_csv(FILE)

# Only analyse the automated Probe A experiment.
df = df[df["scenario"] == "probe_a_latency"].copy()

if df.empty:
    raise ValueError("No Probe A results found in CSV.")

print("\n========== PROBE A ANALYSIS ==========\n")

for model, group in df.groupby("model"):

    successful = group[group["status"] == "ok"]

    print("=" * 65)
    print("MODEL:", model)
    print("=" * 65)

    print("Requests attempted:", len(group))
    print("Successful:", len(successful))
    print("Failed:", len(group) - len(successful))

    for column, label in [
        ("first_token_ms", "TTFT"),
        ("total_ms", "Total response time")
    ]:

        values = successful[column].dropna().astype(float)

        if len(values) == 0:
            print(f"\n{label}: No successful measurements")
            continue

        print(f"\n{label} (milliseconds)")
        print(f"p50: {np.percentile(values, 50):.2f}")
        print(f"p95: {np.percentile(values, 95):.2f}")
        print(f"p99: {np.percentile(values, 99):.2f}")
        print(f"Maximum: {values.max():.2f}")

    print("\nFailure details:")

    failures = group[group["status"] != "ok"]

    if failures.empty:
        print("No failures.")
    else:
        columns = [
            "status",
            "status_code",
            "total_ms",
            "error_message"
        ]

        print(failures[columns].to_string(index=False))

print("\n========== ANALYSIS COMPLETE ==========")
