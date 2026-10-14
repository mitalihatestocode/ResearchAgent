"""Document ingestion and RAG tests (TC01, TC07, TC12, TC13)."""

import io

from rag.pdf_extractor import extract_text_from_pdf
from rag.chunker import chunk_paper, detect_section, _pack_sentences
from rag.retriever import retrieve_relevant_chunks, merge_retrievals
from tests.pdf_factory import make_pdf, PAPER_A_PAGES


def test_tc01_valid_pdf_is_extracted_page_by_page():
    """TC01: a valid paper is extracted with correct page count and text."""
    result = extract_text_from_pdf(io.BytesIO(make_pdf(PAPER_A_PAGES)), "paper_a.pdf")
    assert result["status"] == "ok"
    assert result["num_pages"] == 2
    assert "MIMIC-III" in result["pages"][1]["text"]


def test_tc07_scanned_pdf_without_text_layer_reports_empty():
    """TC07: a scanned (image-only) PDF gives a clear 'empty' status, not a crash."""
    result = extract_text_from_pdf(io.BytesIO(make_pdf([[], []])), "scanned.pdf")
    assert result["status"] == "empty"
    assert "OCR" in result["message"]


def test_tc07b_corrupt_file_reports_error():
    result = extract_text_from_pdf(io.BytesIO(b"this is not a pdf"), "corrupt.pdf")
    assert result["status"] == "error"
    assert result["pages"] == []


def test_chunker_detects_sections_and_keeps_page_numbers(papers):
    chunks = chunk_paper(papers["paper_a.pdf"])
    sections = {c["section"] for c in chunks}
    assert {"Abstract", "Methodology", "Data", "Results", "Limitations"} <= sections
    dataset_chunk = next(c for c in chunks if "MIMIC-III" in c["text"])
    assert dataset_chunk["page_number"] == 2
    assert dataset_chunk["section"] == "Data"


def test_chunker_never_splits_sentences_and_overlaps():
    sentences = [f"Sentence number {i} talks about retrieval evaluation." for i in range(40)]
    chunks = _pack_sentences(sentences, chunk_size=300, overlap=80)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith(".")              # every chunk ends on a sentence boundary
        assert len(chunk) <= 300
    first_tail = chunks[0].split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1]  # overlap carried into the next chunk


def test_references_section_is_not_indexed():
    pages = [["Results", "Accuracy improved to 90 percent on the test set.",
              "References", "Smith J. Some cited paper title about other things. 2019."]]
    result = extract_text_from_pdf(io.BytesIO(make_pdf(pages)), "refs.pdf")
    texts = " ".join(c["text"] for c in chunk_paper(result))
    assert "Accuracy improved" in texts
    assert "Smith J." not in texts


def test_detect_section_handles_numbering():
    assert detect_section("3.1 Experimental Setup") == "Experiments"
    assert detect_section("IV. RESULTS") == "Results"
    assert detect_section("We report results in Table 2 for all settings.") is None


def test_tc12_hybrid_retrieval_finds_exact_dataset_name(store):
    """TC12: hybrid search ranks the chunk containing an exact dataset name first."""
    result = retrieve_relevant_chunks(store, "Which experiments used MIMIC-III?", top_k=3, mode="hybrid")
    assert result["status"] == "ok"
    top = result["chunks"][0]
    assert "MIMIC-III" in top["text"]
    assert top["source_name"] == "paper_a.pdf" and top["page_number"] == 2
    assert {"similarity_score", "bm25_score", "fused_score", "section"} <= set(top)


def test_retrieval_source_filter_restricts_to_one_paper(store):
    result = retrieve_relevant_chunks(store, "dataset", top_k=5, source_filter=["paper_b.pdf"])
    assert result["chunks"]
    assert all(c["source_name"] == "paper_b.pdf" for c in result["chunks"])


def test_tc13_relevance_threshold_rejects_off_topic_queries(store):
    """TC13: off-topic queries return no evidence instead of weakly related text."""
    result = retrieve_relevant_chunks(store, "weather forecast for Mumbai tomorrow", top_k=3,
                                      min_relevance=0.2)
    assert result["status"] == "low_relevance"
    assert result["chunks"] == []


def test_merge_retrievals_deduplicates(store):
    r1 = retrieve_relevant_chunks(store, "dataset used", top_k=4)
    r2 = retrieve_relevant_chunks(store, "dataset corpus", top_k=4)
    merged = merge_retrievals([r1, r2], top_k=10)
    ids = [c["chunk_id"] for c in merged]
    assert len(ids) == len(set(ids))
