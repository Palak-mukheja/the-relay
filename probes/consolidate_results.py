
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# All real experimental CSV files
INPUT_FILES = [
    "probe_raw.csv",
    "concurrency_raw.csv",
    "rpm_probe_raw.csv",
    "tpm_probe_raw.csv",
    "tpm_controlled.csv",
    "model_budget_probe.csv"
]

frames = []

for filename in INPUT_FILES:

    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"WARNING: Missing {filename}")
        continue

    df = pd.read_csv(path)

    print(f"Loaded {filename}: {len(df)} rows")

    frames.append(df)

if not frames:
    raise ValueError("No experiment files found.")

# Combine all records
combined = pd.concat(
    frames,
    ignore_index=True,
    sort=False
)

# Avoid duplicate records if script is rerun
if "request_id" in combined.columns:
    combined = combined.drop_duplicates(
        subset=["request_id"],
        keep="first"
    )

# Required assignment columns
required_columns = [
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

# Add any missing required columns
for column in required_columns:

    if column not in combined.columns:
        combined[column] = ""

# Put required columns first
extra_columns = [
    column for column in combined.columns
    if column not in required_columns
]

combined = combined[
    required_columns + extra_columns
]

# Save consolidated raw data
probe_output = RESULTS_DIR / "probe_raw.csv"

combined.to_csv(
    probe_output,
    index=False
)

# Save assignment's required raw results file
final_output = RESULTS_DIR / "results.csv"

combined.to_csv(
    final_output,
    index=False
)

print("\n========== CONSOLIDATION COMPLETE ==========")
print("Total unique requests:", len(combined))
print("Columns:", len(combined.columns))
print("Probe data:", probe_output)
print("Assignment results:", final_output)

print("\nRequests by scenario:")
print(combined["scenario"].value_counts().to_string())
