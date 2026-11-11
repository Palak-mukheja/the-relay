from pathlib import Path

from src.rag_core import (
    load_pages,
    make_chunks,
    Retriever,
    make_groq_llm
)
from src.logger import Logger
from src.pipeline_logging import run_and_log


# Project paths
ROOT = Path(__file__).resolve().parents[1]

PAPER_FILES = {
    "Attention Is All You Need": ROOT / "data/papers/attention_v7.pdf",
    "Moshi": ROOT / "data/papers/moshi_v1.pdf",
    "Dynamic Chunking": ROOT / "data/papers/dynamic_chunking_v2.pdf"
}

MODEL = "openai/gpt-oss-120b"


def main():

    # 1. Check that all three PDFs exist.
    for name, path in PAPER_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing paper: {path}")

    # 2. Extract text from the PDFs.
    papers = {
        name: load_pages(str(path))
        for name, path in PAPER_FILES.items()
    }

    # 3. Create medium-sized chunks (400 words, 80 overlap).
    chunks = make_chunks(
        papers,
        size_words=400,
        overlap_words=80
    )

    if not chunks:
        raise RuntimeError("No chunks were created.")

    print(f"Loaded {len(papers)} papers.")
    print(f"Created {len(chunks)} chunks.")

    # 4. Build the BM25 retriever.
    retriever = Retriever(chunks)

    # 5. Configure the Groq model.
    llm = make_groq_llm(
    model=MODEL,
    max_completion_tokens=160
    )

    # 6. Prepare the CSV logger.
    logger = Logger(ROOT / "results/rag_smoke_test.csv")

    try:
        question = input("\nAsk a question about the papers: ").strip()

        if not question:
            print("No question entered.")
            return

        # 7. Run the complete pipeline and log the request.
        result = run_and_log(
            question=question,
            retriever=retriever,
            llm=llm,
            logger=logger,
            scenario="rag_smoke_test",
            model=MODEL,
            retrieval_setting="M"
        )

        print("\n--- ANSWER ---")
        print(result["reply"])

        print("\n--- STATUS ---")
        print(result["status"])

        print("\n--- CITATIONS ---")
        print(result["citations"])

        print("\n--- TIMINGS ---")
        print(f"Retrieval: {result['retrieve_ms']:.2f} ms")
        print(f"Model:     {result['llm_ms']:.2f} ms")
        print(f"Total:     {result['total_ms']:.2f} ms")

        print("\n--- TOKENS ---")
        print(f"Input:  {result['tokens_in']}")
        print(f"Output: {result['tokens_out']}")

        print("\n--- TRACE ---")
        print(result["trace"])

        print("\nResults saved to results/rag_smoke_test.csv")

    finally:
        logger.close()


if __name__ == "__main__":
    main()