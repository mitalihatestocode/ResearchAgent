"""
Text Chunker
------------
Takes the page-by-page output from pdf_extractor.py and splits it into
smaller overlapping "chunks" suitable for embedding and retrieval.

WHY CHUNK AT ALL? (viva point)
Embedding models and retrievers work best on small, focused pieces of
text (a paragraph-ish amount), not entire papers. If we embedded whole
papers, a query about "dataset" would retrieve the ENTIRE paper (no
better than just dumping it into the LLM prompt - defeats the purpose
of RAG). Chunking lets us retrieve just the 3-5 most relevant snippets
across ALL papers, keeping the LLM's context focused and grounded.

WHY OVERLAP CHUNKS? So we don't cut a sentence/idea in half at a chunk
boundary and lose meaning - the overlap keeps some shared context
between consecutive chunks.

Each chunk keeps metadata (which paper, which page) so that later, when
we show retrieved evidence to the user, we can say exactly WHERE it
came from - this is what makes the final report's source references
possible.
"""

CHUNK_SIZE = 800       # characters per chunk (rough, not token-exact - fine for a student project)
CHUNK_OVERLAP = 150     # characters shared between consecutive chunks


def chunk_paper(extraction_result: dict) -> list[dict]:
    """
    Split one paper's extracted pages into overlapping chunks.

    Args:
        extraction_result: the dict returned by pdf_extractor.extract_text_from_pdf()
                            (must have status "ok" and a "pages" list)

    Returns:
        A list of chunk dictionaries:
        {
            "chunk_id": "sourcename_p3_c0",
            "source_name": "paper1.pdf",
            "page_number": 3,
            "text": "..."
        }
    """
    if extraction_result["status"] != "ok":
        return []

    source_name = extraction_result["source_name"]
    chunks = []

    for page in extraction_result["pages"]:
        page_text = page["text"]
        page_number = page["page_number"]

        if not page_text:
            continue  # skip pages with no text (already handled upstream, but safe to double check)

        page_chunks = _split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": f"{source_name}_p{page_number}_c{i}",
                "source_name": source_name,
                "page_number": page_number,
                "text": chunk_text,
            })

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple sliding-window splitter. Not linguistically fancy (doesn't
    respect sentence boundaries) - deliberately kept simple for a
    beginner-friendly, explainable implementation. Good enough for
    research paper text, and easy to defend in viva: "we use a
    fixed-size sliding window with overlap to avoid cutting context
    at chunk boundaries."
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # slide forward, keeping overlap

    return chunks
