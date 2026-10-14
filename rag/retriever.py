"""
RAG Retrieval Tool
-------------------
This is the tool the agent calls whenever it needs actual evidence from
the uploaded papers to answer a question. It wraps the vector store's
hybrid search with the standard "tool" shape used throughout this
project: takes a clear input, returns a clear structured output, never
throws an unhandled exception.

This is deliberately the ONLY way paper content reaches the LLM - we
never dump full paper text into a prompt. This is what makes the system
"genuine RAG" rather than a chatbot with paper text pasted in.

RELEVANCE THRESHOLD (safety/viva point):
If the best chunk's cosine similarity is below MIN_RELEVANCE, we report
"no relevant evidence" instead of passing weakly related text to the LLM.
This is what stops off-topic questions ("what's the weather?") from
producing a confident-sounding but ungrounded answer.
"""

from rag.vector_store import VectorStore

# Calibrated on real papers with all-MiniLM-L6-v2: off-topic questions
# ("weather in Mumbai", "cake recipe") score 0.08-0.25, on-topic
# questions 0.4-0.6. The planner's scope check handles the grey zone.
MIN_RELEVANCE = 0.28


def retrieve_relevant_chunks(
    vector_store: VectorStore,
    query: str,
    top_k: int = 5,
    source_filter: list[str] = None,
    mode: str = "hybrid",
    min_relevance: float = None,
) -> dict:
    """
    Retrieve the most relevant paper chunks for a given query.

    Args:
        vector_store: the VectorStore built from the uploaded papers
        query: natural language question (e.g. "What dataset was used?")
        top_k: how many chunks to retrieve
        source_filter: optional - restrict retrieval to specific paper(s)
        mode: "hybrid" | "dense" | "keyword"
        min_relevance: optional cosine threshold; chunks below it are
                       dropped. None = no threshold (used for the broad
                       per-topic queries inside paper analysis).

    Returns:
        {
            "status": "ok" | "empty" | "low_relevance",
            "message": str,
            "query": str,
            "max_similarity": float,
            "chunks": [ {chunk_id, source_name, page_number, section, text,
                         similarity_score, bm25_score, fused_score}, ... ]
        }
    """
    if vector_store is None or vector_store.is_empty():
        return {
            "status": "empty",
            "message": "No papers have been processed into the knowledge base yet.",
            "query": query, "max_similarity": 0.0, "chunks": [],
        }

    chunks = vector_store.search(query, top_k=top_k, source_filter=source_filter, mode=mode)

    if not chunks:
        return {
            "status": "empty",
            "message": "No relevant content was found for this query in the uploaded papers.",
            "query": query, "max_similarity": 0.0, "chunks": [],
        }

    max_similarity = max(c["similarity_score"] for c in chunks)

    if min_relevance is not None:
        relevant = [c for c in chunks if c["similarity_score"] >= min_relevance]
        if not relevant:
            return {
                "status": "low_relevance",
                "message": f"The closest passages in the uploaded papers are only weakly related "
                           f"to this question (best similarity {max_similarity:.2f} < "
                           f"{min_relevance:.2f}), so they were not used as evidence.",
                "query": query, "max_similarity": max_similarity, "chunks": [],
            }
        chunks = relevant

    return {
        "status": "ok",
        "message": f"Retrieved {len(chunks)} relevant chunk(s) (best similarity {max_similarity:.2f}).",
        "query": query, "max_similarity": max_similarity, "chunks": chunks,
    }


def merge_retrievals(results: list[dict], top_k: int) -> list[dict]:
    """
    Merge chunks from several retrievals (e.g. the original query plus
    rewritten queries), de-duplicating by chunk_id and keeping the best
    score for each chunk.
    """
    best = {}
    for result in results:
        for chunk in result.get("chunks", []):
            existing = best.get(chunk["chunk_id"])
            if existing is None or chunk["similarity_score"] > existing["similarity_score"]:
                best[chunk["chunk_id"]] = chunk
    return sorted(best.values(), key=lambda c: -c["similarity_score"])[:top_k]
