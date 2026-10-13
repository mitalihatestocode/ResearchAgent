"""
PDF Extraction Tool
--------------------
This is the FIRST tool in our agent's toolbox.

Job: take a PDF file and turn it into clean, page-tagged text.

WHY THIS MATTERS FOR THE AGENT (viva point):
Every later step (RAG chunking, embeddings, analysis) depends on this
working correctly. If we can't reliably extract text, nothing downstream
can be trusted. So this tool is deliberately defensive: it never crashes
the app, it always returns a clear status so the agent (and the UI) knows
whether extraction actually worked.
"""

import pdfplumber


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be processed at all (corrupt/unreadable)."""
    pass


def extract_text_from_pdf(file_path_or_buffer, source_name: str) -> dict:
    """
    Extract text from a PDF, page by page.

    Args:
        file_path_or_buffer: a file path (str) OR a file-like object
                              (Streamlit's uploaded_file works directly here).
        source_name: a human-readable name for this paper (e.g. filename),
                     used later to tag which chunks came from which paper.

    Returns:
        A dictionary describing what happened. We always return a dict
        (never raise for "expected" problems like empty PDFs) so the
        calling code can display a clean message instead of crashing.

        {
            "source_name": str,
            "status": "ok" | "empty" | "error",
            "message": str,           # human-readable explanation
            "full_text": str,         # all pages joined together
            "pages": [                # one entry per page, keeps page numbers
                {"page_number": 1, "text": "..."},
                ...
            ],
            "num_pages": int,
        }
    """
    pages_data = []

    try:
        with pdfplumber.open(file_path_or_buffer) as pdf:
            num_pages = len(pdf.pages)

            if num_pages == 0:
                return _empty_result(source_name, "PDF has zero pages.")

            for page_number, page in enumerate(pdf.pages, start=1):
                # extract_text() returns None if a page has no extractable
                # text (e.g. it's a scanned image with no OCR layer)
                page_text = page.extract_text()
                pages_data.append({
                    "page_number": page_number,
                    "text": page_text.strip() if page_text else "",
                })

    except Exception as e:
        # Covers corrupt files, password-protected PDFs, non-PDF files
        # renamed to .pdf, etc. We deliberately catch broadly here because
        # pdfplumber/pypdf can raise many different underlying exception
        # types depending on what's wrong with the file.
        return {
            "source_name": source_name,
            "status": "error",
            "message": f"Could not read this PDF. It may be corrupted, "
                       f"password-protected, or not a valid PDF file. "
                       f"(Details: {e})",
            "full_text": "",
            "pages": [],
            "num_pages": 0,
        }

    full_text = "\n\n".join(p["text"] for p in pages_data if p["text"])

    if not full_text.strip():
        # This is the "scanned PDF with no extractable text" case from
        # the error-handling requirements. We don't crash — we tell the
        # user clearly what the problem is and what to do about it.
        return _empty_result(
            source_name,
            "No extractable text was found. This usually means the PDF "
            "is a scanned image without a text layer (needs OCR, which "
            "this tool does not currently perform).",
            num_pages=len(pages_data),
        )

    return {
        "source_name": source_name,
        "status": "ok",
        "message": f"Extracted text from {len(pages_data)} page(s).",
        "full_text": full_text,
        "pages": pages_data,
        "num_pages": len(pages_data),
    }


def _empty_result(source_name: str, message: str, num_pages: int = 0) -> dict:
    """Small helper to avoid repeating this shape everywhere."""
    return {
        "source_name": source_name,
        "status": "empty",
        "message": message,
        "full_text": "",
        "pages": [],
        "num_pages": num_pages,
    }
