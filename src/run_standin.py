
from pathlib import Path

from src.rag_core import (
    load_pages,
    make_chunks,
    Retriever,
    make_groq_llm
)
from src.logger import Logger
from src.pipeline_logging import run_and_log


ROOT = Path(__file__).resolve().parents[1]

PAPER_FILES = {
    "Attention Is All You Need": ROOT / "data/papers/attention_v7.pdf",
    "Moshi": ROOT / "data/papers/moshi_v1.pdf",
    "Dynamic Chunking": ROOT / "data/papers/dynamic_chunking_v2.pdf"
}

MODEL = "openai/gpt-oss-120b"
STANDIN_URL = "http://127.0.0.1:8000/v1"


def main():

    papers = {
        name: load_pages(str(path))
        for name, path in PAPER_FILES.items()
    }

    chunks = make_chunks(
        papers,
        size_words=400,
        overlap_words=80
    )

    print(f"Loaded {len(papers)} papers.")
    print(f"Created {len(chunks)} chunks.")

    retriever = Retriever(chunks)

    llm = make_groq_llm(
        model=MODEL,
        max_completion_tokens=160,
        base_url=STANDIN_URL,
        use_env_key=False
    )

    logger = Logger(ROOT / "results/standin_smoke_test.csv")

    try:
        question = input("\nAsk a question about the papers: ").strip()

        if not question:
            print("No question entered.")
            return

        result = run_and_log(
            question=question,
            retriever=retriever,
            llm=llm,
            logger=logger,
            scenario="standin_smoke_test",
            model="stand-in",
            retrieval_setting="M",
            source="stand-in",
            allow_retry=False
        )

        print("\n--- ANSWER ---")
        print(result["reply"])

        print("\n--- STATUS ---")
        print(result["infra_status"])

        print("\n--- REPLY KIND ---")
        print(result["reply_kind"])

        print("\n--- LATENCY ---")
        print("Model TTFT:", result["model_first_token_ms"])
        print("User-visible TTFT:", result["first_token_ms"])
        print("Total:", result["total_ms"])

        print("\n--- TOKENS ---")
        print("Input:", result["tokens_in"])
        print("Output:", result["tokens_out"])

    finally:
        logger.close()


if __name__ == "__main__":
    main()
