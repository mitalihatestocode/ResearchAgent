"""
Quick manual test for the PDF extraction tool.

This is NOT a pytest suite (keeping things simple for Phase 1) — it's a
small script you run directly to sanity-check that pdf_extractor.py works
before wiring it into the Streamlit app.

Usage:
    python scripts/try_pdf_extractor.py path/to/your/paper.pdf
"""

import sys
import os

# Allow running this script directly from the tests/ folder by adding
# the project root to Python's import path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pdf_extractor import extract_text_from_pdf


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/try_pdf_extractor.py path/to/paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = extract_text_from_pdf(pdf_path, os.path.basename(pdf_path))

    print("-" * 60)
    print(f"Source:   {result['source_name']}")
    print(f"Status:   {result['status']}")
    print(f"Message:  {result['message']}")
    print(f"Pages:    {result['num_pages']}")
    print("-" * 60)

    if result["status"] == "ok":
        print("First 500 characters of extracted text:\n")
        print(result["full_text"][:500])


if __name__ == "__main__":
    main()
