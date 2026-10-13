"""
Metadata Extraction Tool
-------------------------
Extracts title, authors, and publication year from a paper's first page
using the LLM. Kept as its own small, fast tool (rather than folded into
the full paper analysis) because metadata is useful early - e.g. to
label a paper nicely in the UI before the heavier analysis step runs.
"""

from agent.llm_client import call_llm_json, LLMError
from agent.prompts import metadata_prompt


def extract_metadata(extraction_result: dict) -> dict:
    """
    Args:
        extraction_result: dict from pdf_extractor.extract_text_from_pdf()
                            (must have status "ok")

    Returns:
        {
            "status": "ok" | "error",
            "message": str,
            "source_name": str,
            "title": str or None,
            "authors": str or None,
            "year": str or None,
        }
    """
    source_name = extraction_result["source_name"]

    if extraction_result["status"] != "ok" or not extraction_result["pages"]:
        return {
            "status": "error",
            "message": "No extracted text available to pull metadata from.",
            "source_name": source_name,
            "title": None, "authors": None, "year": None,
        }

    first_page_text = extraction_result["pages"][0]["text"]

    try:
        system, user = metadata_prompt(source_name, first_page_text)
        result = call_llm_json(system, user)

        return {
            "status": "ok",
            "message": "Metadata extracted.",
            "source_name": source_name,
            "title": result.get("title"),
            "authors": result.get("authors"),
            "year": result.get("year"),
        }

    except LLMError as e:
        # Graceful degradation: metadata failing shouldn't block the
        # rest of the pipeline, so we return a clear error status
        # rather than raising and crashing the whole analysis.
        return {
            "status": "error",
            "message": str(e),
            "source_name": source_name,
            "title": None, "authors": None, "year": None,
        }
