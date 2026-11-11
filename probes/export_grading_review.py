
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "results/answer_evaluation.jsonl"
OUTPUT = ROOT / "results/grading_review.csv"

# Read all records.
records = []

with open(SOURCE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# Keep the latest successful/non-error result per question.
# This handles Q13, which initially failed and later succeeded.
latest = {}

for row in records:
    qid = str(row["question_id"])

    if qid not in latest:
        latest[qid] = row
    elif row.get("status") != "error":
        latest[qid] = row

# Prepare grading sheet.
fields = [
    "question_id",
    "question",
    "answerable",
    "gold_answer",
    "generated_answer",
    "status",
    "citations",
    "expected_sources",
    "retrieved_pages",
    "grading_notes",
    "tokens_in",
    "tokens_out"
]

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for qid in sorted(latest, key=lambda x: int(x)):
        row = latest[qid]

        writer.writerow({
            key: json.dumps(row.get(key), ensure_ascii=False)
            if key in ("citations", "expected_sources", "retrieved_pages")
            else row.get(key, "")
            for key in fields
        })

# Print summary.
counts = Counter(row.get("status") for row in latest.values())

print(f"Original records: {len(records)}")
print(f"Unique questions: {len(latest)}")
print(f"Duplicate records: {len(records) - len(latest)}")
print("\nFinal status counts:")

for status, count in sorted(counts.items()):
    print(f"{status}: {count}")

print(f"\nGrading sheet saved to: {OUTPUT}")
