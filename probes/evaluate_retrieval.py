
import json
import pandas as pd

from pathlib import Path

from src.retriever import (
    BM25Retriever,
    RETRIEVAL_SETTINGS
)


BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = BASE_DIR / "data" / "evaluation_questions.json"

RESULTS_DIR = BASE_DIR / "results"

DETAILS_FILE = RESULTS_DIR / "retrieval_evaluation.csv"

SUMMARY_FILE = RESULTS_DIR / "retrieval_summary.csv"


# ------------------------------------------------
# PAGE NUMBER CONFIGURATION
# ------------------------------------------------

# PDF page number = printed page number + offset.
#
# Initially set to zero. Verify this against the
# footer and PDF viewer for each paper before
# treating the evaluation metrics as final.

PAGE_OFFSETS = {
    "Attention Is All You Need": 0,
    "Moshi": 0,
    "Dynamic Chunking": 0
}


# The name in the evaluation JSON differs from
# the name stored in our extracted chunks.

PAPER_ALIASES = {
    "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling":
        "Dynamic Chunking"
}


def canonical_paper(name):

    return PAPER_ALIASES.get(name, name)


def get_pdf_page(paper, printed_page):

    paper = canonical_paper(paper)

    offset = PAGE_OFFSETS.get(paper, 0)

    return printed_page + offset


# ------------------------------------------------
# LOAD EVALUATION QUESTIONS
# ------------------------------------------------

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    questions = dataset["questions"]

    ids = [q["id"] for q in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found.")

    print("Questions loaded:", len(questions))

    return questions


# ------------------------------------------------
# BUILD GOLD PAGE SET
# ------------------------------------------------

def build_gold_pages(question):

    gold_pages = set()

    for source in question["expected_sources"]:

        paper = canonical_paper(source["paper"])

        for printed_page in source["pages"]:

            pdf_page = get_pdf_page(
                paper,
                printed_page
            )

            gold_pages.add(
                (paper, pdf_page)
            )

    return gold_pages


def build_gold_source_groups(question):

    groups = []

    for source in question["expected_sources"]:

        paper = canonical_paper(source["paper"])

        pages = {
            get_pdf_page(paper, page)
            for page in source["pages"]
        }

        groups.append({
            "paper": paper,
            "pages": pages
        })

    return groups


# ------------------------------------------------
# GET PAGES COVERED BY A RETRIEVED CHUNK
# ------------------------------------------------

def get_chunk_pages(chunk):

    paper = canonical_paper(chunk["paper"])

    start = int(chunk["page_start"])
    end = int(chunk["page_end"])

    return {
        (paper, page)
        for page in range(start, end + 1)
    }


# ------------------------------------------------
# EVALUATE ONE QUESTION UNDER ONE SETTING
# ------------------------------------------------

def evaluate_question(
    question,
    setting,
    retriever
):

    results = retriever.retrieve(
        question["question"]
    )

    answerable = question["answerable"]

    expected_pages = build_gold_pages(question)

    expected_groups = build_gold_source_groups(
        question
    )

    retrieved_pages = set()

    relevant_flags = []

    wrong_paper_chunks = 0

    expected_papers = {
        paper for paper, _ in expected_pages
    }

    for chunk in results:

        chunk_pages = get_chunk_pages(chunk)

        retrieved_pages.update(chunk_pages)

        is_relevant = bool(
            chunk_pages & expected_pages
        )

        relevant_flags.append(is_relevant)

        if chunk["paper"] not in expected_papers:
            wrong_paper_chunks += 1

    # ------------------------------------------------
    # UNANSWERABLE QUESTIONS
    # ------------------------------------------------

    if not answerable:

        return {

            "question_id": question["id"],

            "question_type": question["type"],

            "setting": setting,

            "answerable": False,

            "fetch_k": RETRIEVAL_SETTINGS[setting]["fetch_k"],

            "keep_k": RETRIEVAL_SETTINGS[setting]["keep_k"],

            "returned_chunks": len(results),

            "expected_gold_pages": 0,

            "matched_gold_pages": 0,

            "page_recall": None,

            "page_precision_proxy": None,

            "mrr": None,

            "all_gold_pages_found": None,

            "expected_source_groups": 0,

            "matched_source_groups": 0,

            "source_group_recall": None,

            "all_source_groups_found": None,

            "wrong_paper_chunks": None,

            "wrong_paper_rate": None,

            "unanswerable_has_candidates": bool(results),

            "top_result_paper": (
                results[0]["paper"] if results else None
            ),

            "top_result_pages": (
                f'{results[0]["page_start"]}-'
                f'{results[0]["page_end"]}'
                if results else None
            ),

            "top_bm25_score": (
                results[0]["bm25_score"] if results else None
            )
        }

    # ------------------------------------------------
    # ANSWERABLE QUESTIONS
    # ------------------------------------------------

    matched_pages = retrieved_pages & expected_pages

    page_recall = (
        len(matched_pages) / len(expected_pages)
        if expected_pages else 0
    )

    relevant_chunks = sum(relevant_flags)

    precision = (
        relevant_chunks / len(results)
        if results else 0
    )

    # Reciprocal rank of first relevant chunk
    reciprocal_rank = 0

    for rank, is_relevant in enumerate(
        relevant_flags,
        start=1
    ):

        if is_relevant:

            reciprocal_rank = 1 / rank
            break

    # Check whether each expected source group appears
    matched_groups = 0

    for group in expected_groups:

        group_pages = {
            (group["paper"], page)
            for page in group["pages"]
        }

        if retrieved_pages & group_pages:
            matched_groups += 1

    group_recall = (
        matched_groups / len(expected_groups)
        if expected_groups else 0
    )

    wrong_paper_rate = (
        wrong_paper_chunks / len(results)
        if results else 0
    )

    return {

        "question_id": question["id"],

        "question_type": question["type"],

        "setting": setting,

        "answerable": True,

        "fetch_k": RETRIEVAL_SETTINGS[setting]["fetch_k"],

        "keep_k": RETRIEVAL_SETTINGS[setting]["keep_k"],

        "returned_chunks": len(results),

        "expected_gold_pages": len(expected_pages),

        "matched_gold_pages": len(matched_pages),

        "page_recall": round(page_recall, 4),

        "page_precision_proxy": round(precision, 4),

        "mrr": round(reciprocal_rank, 4),

        "all_gold_pages_found": (
            len(matched_pages) == len(expected_pages)
        ),

        "expected_source_groups": len(expected_groups),

        "matched_source_groups": matched_groups,

        "source_group_recall": round(group_recall, 4),

        "all_source_groups_found": (
            matched_groups == len(expected_groups)
        ),

        "wrong_paper_chunks": wrong_paper_chunks,

        "wrong_paper_rate": round(
            wrong_paper_rate, 4
        ),

        "unanswerable_has_candidates": None,

        "top_result_paper": (
            results[0]["paper"] if results else None
        ),

        "top_result_pages": (
            f'{results[0]["page_start"]}-'
            f'{results[0]["page_end"]}'
            if results else None
        ),

        "top_bm25_score": (
            results[0]["bm25_score"] if results else None
        )
    }


# ------------------------------------------------
# CREATE SUMMARY FOR EACH SETTING
# ------------------------------------------------

def calculate_summary(details):

    summaries = []

    for setting, group in details.groupby("setting"):

        answerable = group[
            group["answerable"] == True
        ]

        unanswerable = group[
            group["answerable"] == False
        ]

        total_chunks = answerable["returned_chunks"].sum()

        wrong_paper_rate = (
            answerable["wrong_paper_chunks"].sum() / total_chunks
            if total_chunks else 0
        )

        # Retrieve the original chunk-level precision counts
        # by summing the available per-question proxies.
        summary = {
          "wrong_paper_chunk_rate": round(
           wrong_paper_rate, 4
            ),
            "setting": setting,

            "questions_tested": len(group),

            "answerable_questions": len(answerable),

            "unanswerable_questions": len(unanswerable),

            "average_chunks_returned": round(
                group["returned_chunks"].mean(), 2
            ),

            "mean_page_recall": round(
                answerable["page_recall"].mean(), 4
            ),

            "mean_page_precision_proxy": round(
                answerable["page_precision_proxy"].mean(), 4
            ),

            "mean_mrr": round(
                answerable["mrr"].mean(), 4
            ),

            "full_gold_page_coverage_pct": round(
                answerable["all_gold_pages_found"].mean() * 100,
                2
            ),

            "mean_source_group_recall": round(
                answerable["source_group_recall"].mean(),
                4
            ),

            "all_source_groups_found_pct": round(
                answerable["all_source_groups_found"].mean() * 100,
                2
            ),

            "unanswerable_with_retrieved_candidates": int(
                unanswerable["unanswerable_has_candidates"].sum()
            )
        }

        summaries.append(summary)

    return pd.DataFrame(summaries)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    questions = load_questions()

    all_results = []

    for setting in RETRIEVAL_SETTINGS:

        print("\n" + "=" * 60)
        print("EVALUATING SETTING:", setting)
        print("=" * 60)

        retriever = BM25Retriever(setting)

        for question in questions:

            result = evaluate_question(
                question,
                setting,
                retriever
            )

            all_results.append(result)

            if question["answerable"]:

                print(
                    f"Q{question['id']}: "
                    f"Recall={result['page_recall']:.2f} | "
                    f"MRR={result['mrr']:.2f} | "
                    f"Chunks={result['returned_chunks']}"
                )

            else:

                print(
                    f"Q{question['id']}: "
                    f"Unanswerable | "
                    f"Candidates={result['returned_chunks']}"
                )

    # Save detailed results
    details = pd.DataFrame(all_results)

    details.to_csv(
        DETAILS_FILE,
        index=False
    )

    # Calculate per-setting summaries
    summary = calculate_summary(details)

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 65)

    print("Total evaluations:", len(details))

    print("\nSUMMARY:")

    print(
        summary.to_string(index=False)
    )

    print("\nDetailed results saved to:")
    print(DETAILS_FILE)

    print("\nSummary saved to:")
    print(SUMMARY_FILE)

    print("\nPAGE OFFSET CONFIGURATION:")
    print(PAGE_OFFSETS)

    print(
        "\nIMPORTANT: Verify printed-page to PDF-page "
        "offsets before treating page-based scores as final."
    )


if __name__ == "__main__":
    main()
