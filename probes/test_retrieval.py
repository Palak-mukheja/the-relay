
from src.retriever import (
    BM25Retriever,
    RETRIEVAL_SETTINGS
)


QUESTIONS = [
    "What is multi-head attention?",
    "How does Moshi support real-time dialogue?",
    "What is dynamic chunking?"
]


def main():

    for setting in RETRIEVAL_SETTINGS:

        print("\n" + "=" * 65)
        print("RETRIEVAL SETTING:", setting)
        print("=" * 65)

        retriever = BM25Retriever(setting)

        for question in QUESTIONS:

            print("\nQuestion:", question)

            results = retriever.retrieve(question)

            if not results:
                print("No relevant passages found.")
                continue

            for result in results:

                print(
                    f"\nRank: {result['rank']}"
                )

                print(
                    f"Paper: {result['paper']}"
                )

                print(
                    f"Pages: {result['page_start']}"
                    f"-{result['page_end']}"
                )

                print(
                    f"BM25 score: {result['bm25_score']}"
                )

                print(
                    "Text:",
                    result["text"][:300].replace("\n", " ")
                )


if __name__ == "__main__":
    main()
