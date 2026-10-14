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
of RAG). Chunking lets us retrieve just the few most relevant snippets
across ALL papers, keeping the LLM's context focused and grounded.

HOW WE CHUNK (improvements over a naive fixed-size window):
1. SENTENCE-AWARE: chunks are packed from whole sentences up to
   CHUNK_SIZE characters, so we never cut a sentence in half.
2. OVERLAP: the last sentence(s) of each chunk (~CHUNK_OVERLAP chars)
   are repeated at the start of the next one, so an idea spanning a
   boundary is still retrievable.
3. SECTION-AWARE: common paper headings (Abstract, Method, Results,
   Limitations, ...) are detected and stored as chunk metadata. This
   lets the UI show "page 4, Results" and lets retrieval prefer the
   right section.
4. REFERENCES SKIPPED: the bibliography is excluded from the index -
   it is full of other papers' titles that would pollute retrieval.
5. INJECTION SCAN: every chunk is scanned for prompt-injection text and
   flagged (see safety/guardrails.py).

Each chunk keeps metadata (which paper, which page, which section) so
answers can cite exactly WHERE evidence came from.
"""

import re

from safety.guardrails import scan_for_injection

CHUNK_SIZE = 900        # max characters per chunk
CHUNK_OVERLAP = 150     # approx. characters carried over between consecutive chunks
MIN_CHUNK_CHARS = 40    # drop tiny fragments (page numbers, stray headers)

SECTION_NAMES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "literature review": "Related Work",
    "method": "Methodology",
    "methods": "Methodology",
    "methodology": "Methodology",
    "approach": "Methodology",
    "proposed method": "Methodology",
    "materials and methods": "Methodology",
    "dataset": "Data",
    "datasets": "Data",
    "data": "Data",
    "experiments": "Experiments",
    "experiment": "Experiments",
    "experimental setup": "Experiments",
    "evaluation": "Evaluation",
    "results": "Results",
    "results and discussion": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "limitations and future work": "Limitations",
    "threats to validity": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion",
    "future work": "Future Work",
    "references": "References",
    "bibliography": "References",
    "acknowledgments": "Acknowledgements",
    "acknowledgements": "Acknowledgements",
}
SKIP_SECTIONS = {"References"}

# Optional numbering like "3", "3.1", "IV." followed by a heading word
_HEADING_RE = re.compile(r"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?([A-Za-z][A-Za-z &]{2,40}?)\s*:?\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def detect_section(line: str):
    """Returns a canonical section name if the line looks like a heading, else None."""
    if len(line) > 60:
        return None
    match = _HEADING_RE.match(line)
    if not match:
        return None
    return SECTION_NAMES.get(match.group(1).strip().lower())


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _pack_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily packs whole sentences into chunks with sentence-level overlap."""
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        # A single very long "sentence" (tables, equations) is hard-split
        while len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(sentence[:chunk_size])
            sentence = sentence[chunk_size - overlap:]

        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # carry trailing sentences forward as overlap
            carried, carried_len = [], 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, current_len = carried, carried_len

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_paper(extraction_result: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split one paper's extracted pages into overlapping, section-tagged chunks.

    Args:
        extraction_result: the dict returned by pdf_extractor.extract_text_from_pdf()
                            (must have status "ok" and a "pages" list)

    Returns:
        A list of chunk dictionaries:
        {
            "chunk_id": "paper1.pdf_p3_c0",
            "source_name": "paper1.pdf",
            "page_number": 3,
            "section": "Results",
            "text": "...",
            "injection_flags": [...]   # non-empty if suspicious text was found
        }
    """
    if extraction_result["status"] != "ok":
        return []

    source_name = extraction_result["source_name"]
    chunks = []
    current_section = "Front Matter"

    for page in extraction_result["pages"]:
        page_text = page["text"]
        page_number = page["page_number"]
        if not page_text:
            continue

        # Split the page into (section, text) blocks at detected headings
        blocks, buffer = [], []
        block_section = current_section
        for line in page_text.splitlines():
            heading = detect_section(line)
            if heading:
                if buffer:
                    blocks.append((block_section, "\n".join(buffer)))
                    buffer = []
                current_section = heading
                block_section = heading
            else:
                buffer.append(line)
        if buffer:
            blocks.append((block_section, "\n".join(buffer)))

        chunk_index = 0
        for section, block_text in blocks:
            if section in SKIP_SECTIONS:
                continue
            for chunk_text in _pack_sentences(split_sentences(block_text), chunk_size, overlap):
                if len(chunk_text) < MIN_CHUNK_CHARS:
                    continue
                chunks.append({
                    "chunk_id": f"{source_name}_p{page_number}_c{chunk_index}",
                    "source_name": source_name,
                    "page_number": page_number,
                    "section": section,
                    "text": chunk_text,
                    "injection_flags": scan_for_injection(chunk_text),
                })
                chunk_index += 1

    return chunks
