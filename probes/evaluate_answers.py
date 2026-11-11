
import json
import time
from pathlib import Path

from src.rag_core import load_pages, make_chunks, Retriever, make_groq_llm
from src.logger import Logger
from src.pipeline_logging import run_and_log


ROOT = Path(__file__).resolve().parents[1]

PAPER_FILES = {
    "Attention Is All You Need": ROOT / "data/papers/attention_v7.pdf",
    "Moshi": ROOT / "data/papers/moshi_v1.pdf",
    "Dynamic Chunking": ROOT / "data/papers/dynamic_chunking_v2.pdf"
}

QUESTIONS_FILE = ROOT / "data" / "evaluation_questions.json"
OUTPUT_FILE = ROOT / "results/answer_evaluation.jsonl"
CSV_FILE = ROOT / "results/rag_evaluation.csv"

MODEL = "openai/gpt-oss-120b"

# Conservative pacing for the 8,000 TPM limit.
PAUSE_SECONDS = 65


def load_completed_ids():
    completed = set()

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    # Failed requests can be attempted again when resuming.
                    if row.get("status") != "error":
                        completed.add(row["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    return completed


def main():

    # Load the question dataset.
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]

    print(f"Questions loaded: {len(questions)}")

    # Load the three research papers.
    papers = {
        name: load_pages(str(path))
        for name, path in PAPER_FILES.items()
    }

    # Build medium retrieval configuration.
    chunks = make_chunks(
        papers,
        size_words=400,
        overlap_words=80
    )

    retriever = Retriever(chunks)

    print(f"Papers loaded: {len(papers)}")
    print(f"Chunks created: {len(chunks)}")
    print("Retrieval setting: M (400 words, 80 overlap, top-4)")

    # Configure the actual Groq model.
    llm = make_groq_llm(
        model=MODEL,
        max_completion_tokens=256
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_ids()
    remaining = [q for q in questions if q["id"] not in completed]

    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All questions have already been evaluated.")
        return

    logger = Logger(CSV_FILE)

    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as output:

            for index, q in enumerate(remaining, start=1):

                print(f"\n[{index}/{len(remaining)}] Question {q['id']}")
                print(q["question"])

                result = run_and_log(
                    question=q["question"],
                    retriever=retriever,
                    llm=llm,
                    logger=logger,
                    scenario="answer_evaluation",
                    model=MODEL,
                    retrieval_setting="M",
                    source="real",
                    allow_retry=False
                )

                # Save answer and grading information.
                record = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "paper": q["paper"],
                    "answerable": q["answerable"],
                    "gold_answer": q["gold_answer"],
                    "expected_sources": q["expected_sources"],
                    "grading_notes": q["grading_notes"],
                    "generated_answer": result["reply"],
                    "status": result["status"],
                    "citations": result["citations"],
                    "retrieved_pages": result["retrieved_pages"],
                    "first_token_ms": result.get("first_token_ms"),
                    "total_ms": result.get("total_ms"),
                    "tokens_in": result.get("tokens_in"),
                    "tokens_out": result.get("tokens_out"),
                    "attempts": result.get("attempts"),
                    "trace": result.get("trace"),
                    "error_message": result.get("error_message"),
                    "answer_grade": None
                }

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

                print(f"Status: {result['status']}")
                print(f"Tokens: {result['tokens_in']} in / "
                      f"{result['tokens_out']} out")
                print(f"Answer: {result['reply'][:300]}")

                # Stop rather than repeatedly hitting a service error.
                if result["status"] == "error":
                    print("Stopping batch because the model request failed.")
                    print("Rerun later to resume from unfinished questions.")
                    break

                if index < len(remaining):
                    print(f"Pausing {PAUSE_SECONDS} seconds...")
                    time.sleep(PAUSE_SECONDS)

    finally:
        logger.close()

    print("\nEvaluation batch finished.")
    print(f"Answers: {OUTPUT_FILE}")
    print(f"Request metrics: {CSV_FILE}")


if __name__ == "__main__":
    main()
