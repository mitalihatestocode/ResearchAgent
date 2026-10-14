"""
Integration test with the REAL embedding model and REAL papers (TC12/TC13
on production settings). Skipped automatically when the sample papers have
not been downloaded (python scripts/download_sample_papers.py).

Verifies that the relevance threshold used in production cleanly separates
on-topic from off-topic questions on real research papers.
"""

import os

import pytest

from rag.retriever import MIN_RELEVANCE, retrieve_relevant_chunks

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_papers")
SAMPLES = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".pdf")) if os.path.isdir(SAMPLE_DIR) else []

pytestmark = pytest.mark.skipif(len(SAMPLES) < 3 or os.environ.get("SKIP_REAL_MODEL") == "1",
                                reason="sample papers not downloaded")


@pytest.fixture(scope="module")
def real_model():
    from sentence_transformers import SentenceTransformer
    from rag.embeddings import _MODEL_NAME
    return SentenceTransformer(_MODEL_NAME)


@pytest.fixture(autouse=True)
def use_real_model(fake_embeddings, real_model, monkeypatch):
    """Runs after conftest's autouse fake and swaps the real model back in."""
    import rag.embeddings
    monkeypatch.setattr(rag.embeddings, "_model", real_model)


@pytest.fixture(scope="module")
def real_store(real_model):
    import rag.embeddings
    from rag.pdf_extractor import extract_text_from_pdf
    from rag.chunker import chunk_paper
    from rag.vector_store import VectorStore

    previous = rag.embeddings._model
    rag.embeddings._model = real_model
    store = VectorStore()
    for name in SAMPLES:
        result = extract_text_from_pdf(os.path.join(SAMPLE_DIR, name), name)
        assert result["status"] == "ok"
        chunks = chunk_paper(result)
        assert len({c["section"] for c in chunks}) >= 4, f"section detection failed for {name}"
        store.add_chunks(chunks)
    rag.embeddings._model = previous
    return store


@pytest.mark.parametrize("query", [
    "What datasets were used for evaluation?",
    "What is the retrieval evaluator in Corrective RAG?",
    "How does Self-RAG use reflection tokens?",
])
def test_on_topic_questions_pass_threshold(real_store, query):
    result = retrieve_relevant_chunks(real_store, query, top_k=3, min_relevance=MIN_RELEVANCE)
    assert result["status"] == "ok", result["message"]


@pytest.mark.parametrize("query", [
    "What's the weather in Mumbai today?",
    "recipe for chocolate cake",
    "Who won the cricket world cup?",
])
def test_off_topic_questions_are_rejected(real_store, query):
    result = retrieve_relevant_chunks(real_store, query, top_k=3, min_relevance=MIN_RELEVANCE)
    assert result["status"] == "low_relevance", result["message"]


def test_keyword_search_finds_exact_benchmark_name(real_store):
    result = retrieve_relevant_chunks(real_store, "PopQA", top_k=3, mode="keyword")
    assert any("PopQA" in c["text"] for c in result["chunks"])


def test_two_column_pages_keep_full_width_title():
    """Regression (found in live UI testing): the centred title must not be split by column detection."""
    from rag.pdf_extractor import extract_text_from_pdf
    result = extract_text_from_pdf(os.path.join(SAMPLE_DIR, "corrective_rag_2024.pdf"), "crag.pdf")
    assert result["pages"][0]["text"].startswith("Corrective Retrieval Augmented Generation")
