
from pathlib import Path
import json
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "extracted_pages.jsonl"

papers = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        papers[record["paper"]].append(record)

print("EXTRACTION QUALITY CHECK")
print("=" * 50)

for paper, pages in papers.items():

    pages.sort(key=lambda x: x["page"])

    print(f"\nPaper: {paper}")
    print(f"Pages extracted: {len(pages)}")

    short_pages = [
        p["page"] for p in pages
        if len(p["text"].strip()) < 40
    ]

    print(f"Pages with very little text: {short_pages}")

    # Display a sample of extracted text
    sample = next(
        (p for p in pages if p["text"].strip()),
        None
    )

    if sample:
        print(f"\nSample from page {sample['page']}:")
        print(sample["text"][:500])
        print("-" * 50)
