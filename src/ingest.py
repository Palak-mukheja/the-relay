
from pathlib import Path
import json
import fitz  # PyMuPDF


# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder containing our research papers
PAPERS_DIR = BASE_DIR / "data" / "papers"

# File where extracted text will be stored
OUTPUT_FILE = BASE_DIR / "data" / "extracted_pages.jsonl"


# Exact PDF filenames and their identifiers
PAPERS = {
    "attention_v7.pdf": "Attention Is All You Need",
    "moshi_v1.pdf": "Moshi",
    "dynamic_chunking_v2.pdf": "Dynamic Chunking"
}


def extract_papers():
    total_pages = 0
    total_chars = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for filename, paper_name in PAPERS.items():

            pdf_path = PAPERS_DIR / filename

            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF not found: {pdf_path}"
                )

            print(f"\nExtracting: {paper_name}")

            # Open the PDF
            document = fitz.open(pdf_path)

            print(f"Total pages: {len(document)}")

            # Read each page separately
            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text").strip()

                # Store the text along with its source
                record = {
                    "paper": paper_name,
                    "filename": filename,
                    "page": page_number,
                    "text": text
                }

                # Write one JSON object per line
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_pages += 1
                total_chars += len(text)

            document.close()

    print("\nExtraction completed!")
    print(f"Total pages processed: {total_pages}")
    print(f"Total characters extracted: {total_chars}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_papers()
