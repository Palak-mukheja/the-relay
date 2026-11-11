
import json
import re

from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "extracted_pages.jsonl"

OUTPUT_DIR = BASE_DIR / "data"


# All configurations use 20% overlap
CHUNK_SETTINGS = {
    "S": {
        "name": "small",
        "chunk_size": 200,
        "overlap": 40
    },
    "M": {
        "name": "medium",
        "chunk_size": 400,
        "overlap": 80
    },
    "L": {
        "name": "large",
        "chunk_size": 800,
        "overlap": 160
    }
}


KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "conclusion",
    "references",
    "related work",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "appendix"
}


def detect_headings(text):
    """
    Detect likely section headings.
    This is heuristic, not perfect.
    """

    headings = []

    for line in text.splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        if not line or len(line) > 120:
            continue

        lower = line.lower()

        # Common unnumbered section headings
        if lower in KNOWN_HEADINGS:
            headings.append(line)
            continue

        # Numbered headings such as 1 Introduction,
        # 3.2 Attention Mechanism, etc.
        if re.match(
            r"^\d+(?:\.\d+)*\.?\s+[A-Z]",
            line
        ):
            headings.append(line)
            continue

        # Roman numeral headings
        if re.match(r"^[IVX]+\.\s+[A-Z]", line):
            headings.append(line)

    return headings


def load_pages():

    pages = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                pages.append(json.loads(line))

    return pages


def prepare_paper_words(pages):

    """
    Build a sequence of words for one paper.

    Each word retains:
    - its original page
    - its start position
    - its end position

    This allows chunks to cross page boundaries
    without losing citation information.
    """

    word_references = []

    page_texts = {}
    page_headings = {}

    for page in sorted(pages, key=lambda x: x["page"]):

        page_number = page["page"]
        text = page["text"]

        page_texts[page_number] = text

        page_headings[page_number] = detect_headings(text)

        matches = list(re.finditer(r"\S+", text))

        for match in matches:

            word_references.append({
                "page": page_number,
                "start": match.start(),
                "end": match.end()
            })

    return word_references, page_texts, page_headings


def build_chunk(
    word_references,
    page_texts,
    page_headings,
    start,
    end,
    paper,
    filename,
    chunk_index,
    chunk_size,
    overlap
):

    selected_words = word_references[start:end]

    if not selected_words:
        return None

    page_order = list(dict.fromkeys(
        word["page"] for word in selected_words
    ))

    # Collect original text fragments page by page.
    fragments = []

    for page_number in page_order:

        page_words = [
            word for word in selected_words
            if word["page"] == page_number
        ]

        first_position = page_words[0]["start"]
        last_position = page_words[-1]["end"]

        original_text = page_texts[page_number]

        fragment = original_text[
            first_position:last_position
        ]

        fragments.append(fragment)

    # Preserve page breaks between fragments.
    chunk_text = "\n[PAGE BREAK]\n".join(fragments)

    # Collect detected headings from covered pages.
    headings = []

    for page_number in page_order:

        for heading in page_headings[page_number]:

            if heading not in headings:
                headings.append(heading)

    return {

        "chunk_id": (
            f"{Path(filename).stem}_"
            f"c{chunk_index}"
        ),

        "paper": paper,

        "filename": filename,

        "page_start": min(page_order),

        "page_end": max(page_order),

        "section_headings": headings,

        "start_word_index": start + 1,

        "end_word_index": end,

        "chunk_index": chunk_index,

        "chunk_size_words": chunk_size,

        "overlap_words": overlap,

        "word_count": len(selected_words),

        "text": chunk_text
    }


def chunk_paper(
    pages,
    chunk_size,
    overlap
):

    if chunk_size <= overlap:
        raise ValueError(
            "Chunk size must be greater than overlap."
        )

    paper = pages[0]["paper"]
    filename = pages[0]["filename"]

    words, page_texts, page_headings = (
        prepare_paper_words(pages)
    )

    chunks = []

    step = chunk_size - overlap

    for chunk_index, start in enumerate(
        range(0, len(words), step)
    ):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = build_chunk(
            word_references=words,
            page_texts=page_texts,
            page_headings=page_headings,
            start=start,
            end=end,
            paper=paper,
            filename=filename,
            chunk_index=chunk_index,
            chunk_size=chunk_size,
            overlap=overlap
        )

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

    return chunks


def main():

    pages = load_pages()

    print("Pages loaded:", len(pages))

    # Group pages by paper.
    # A chunk will never cross between different papers.
    papers = defaultdict(list)

    for page in pages:

        papers[
            (page["paper"], page["filename"])
        ].append(page)

    for setting_id, setting in CHUNK_SETTINGS.items():

        chunk_size = setting["chunk_size"]
        overlap = setting["overlap"]

        print("\n" + "=" * 55)
        print("SETTING:", setting_id)
        print("Chunk size:", chunk_size)
        print("Overlap:", overlap)

        all_chunks = []

        for _, paper_pages in papers.items():

            paper_chunks = chunk_paper(
                paper_pages,
                chunk_size,
                overlap
            )

            all_chunks.extend(paper_chunks)

        output_file = (
            OUTPUT_DIR /
            f"chunks_{setting['name']}.jsonl"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            for chunk in all_chunks:

                file.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

        average_words = (
            sum(c["word_count"] for c in all_chunks)
            / len(all_chunks)
            if all_chunks else 0
        )

        print("Total chunks:", len(all_chunks))

        print(
            "Average words:",
            round(average_words, 2)
        )

        print("Saved to:", output_file)


if __name__ == "__main__":
    main()
