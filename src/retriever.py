
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# Four required experimental configurations
RETRIEVAL_SETTINGS = {

    "S": {
        "file": "chunks_small.jsonl",
        "fetch_k": 6,
        "keep_k": 6,
        "rerank": False
    },

    "M": {
        "file": "chunks_medium.jsonl",
        "fetch_k": 4,
        "keep_k": 4,
        "rerank": False
    },

    "L": {
        "file": "chunks_large.jsonl",
        "fetch_k": 2,
        "keep_k": 2,
        "rerank": False
    },

    "M+R": {
        "file": "chunks_medium.jsonl",
        "fetch_k": 8,
        "keep_k": 4,
        "rerank": True
    }
}


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "what", "how", "why", "when", "where",
    "does", "do", "did", "of", "to", "in", "on",
    "and", "or", "for", "with", "by", "from",
    "it", "this", "that", "these", "those"
}


def tokenize(text):

    words = re.findall(
        r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOPWORDS
    ]


def load_chunks(filename):

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {filepath}"
        )

    chunks = []

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                chunks.append(json.loads(line))

    return chunks


class BM25Retriever:

    def __init__(self, setting="M"):

        if setting not in RETRIEVAL_SETTINGS:
            raise ValueError(
                f"Unknown retrieval setting: {setting}"
            )

        self.setting = setting

        self.config = RETRIEVAL_SETTINGS[setting]

        self.chunks = load_chunks(
            self.config["file"]
        )

        if not self.chunks:
            raise ValueError("No chunks available.")

        # Tokenize every chunk once
        self.corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        # Build BM25 search index
        self.bm25 = BM25Okapi(self.corpus)

        print(
            f"Loaded {len(self.chunks)} chunks "
            f"for setting {setting}"
        )

    def retrieve(self, query):

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        fetch_k = self.config["fetch_k"]
        keep_k = self.config["keep_k"]

        # Calculate BM25 relevance scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort chunks from highest to lowest score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Remove chunks with no keyword relevance
        candidates = [
            i for i in ranked_indices
            if scores[i] > 0
        ][:fetch_k]

        if not candidates:
            return []

        # Optional inexpensive lexical re-ranking
        if self.config["rerank"]:

            bm25_values = [
                float(scores[i])
                for i in candidates
            ]

            minimum = min(bm25_values)
            maximum = max(bm25_values)

            query_terms = set(query_tokens)

            reranked = []

            for index in candidates:

                document_terms = set(
                    self.corpus[index]
                )

                overlap = (
                    len(query_terms & document_terms)
                    / len(query_terms)
                )

                if maximum > minimum:
                    normalized_bm25 = (
                        (scores[index] - minimum)
                        / (maximum - minimum)
                    )
                else:
                    normalized_bm25 = 1.0

                # Combine BM25 relevance and term coverage
                combined_score = (
                    0.7 * normalized_bm25
                    + 0.3 * overlap
                )

                reranked.append(
                    (index, combined_score)
                )

            reranked.sort(
                key=lambda item: item[1],
                reverse=True
            )

            final_indices = [
                index for index, _ in reranked[:keep_k]
            ]

            rerank_scores = {
                index: score
                for index, score in reranked
            }

        else:

            final_indices = candidates[:keep_k]

            rerank_scores = {}

        # Return passages with citation metadata
        results = []

        for rank, index in enumerate(final_indices, start=1):

            chunk = self.chunks[index].copy()

            chunk["rank"] = rank

            chunk["bm25_score"] = round(
                float(scores[index]), 5
            )

            chunk["rerank_score"] = (
                round(rerank_scores[index], 5)
                if index in rerank_scores
                else None
            )

            results.append(chunk)

        return results
