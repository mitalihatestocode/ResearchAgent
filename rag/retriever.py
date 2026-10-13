"""
RAG Retrieval Tool
-------------------
This is the tool the agent calls whenever it needs actual evidence from
the uploaded papers to answer a question. It wraps the vector store's
similarity search with the standard "tool" shape used throughout this
project: takes a clear input, returns a clear structured output, never
throws an unhandled exception.

This is deliberately the ONLY way paper content reaches the LLM later -
we never dump full paper text into a prompt. This is what makes the
system "genuine RAG" rather than a chatbot with paper text pasted in.
"""

from rag.vector_store import VectorStore


def retrieve_relevant_chunks(
    vector_store: VectorStore,
    query: str,
    top_k: int = 5,
    source_filter: list[str] = None,
) -> dict:
    """
    Retrieve the most relevant paper chunks for a given query.

    Args:
        vector_store: the VectorStore built from the uploaded papers
        query: natural language question (e.g. "What dataset was used?")
        top_k: how many chunks to retrieve
        source_filter: optional - restrict retrieval to specific paper(s)

    Returns:
        {
            "status": "ok" | "empty",
            "message": str,
            "query": str,
            "chunks": [ {chunk_id, source_name, page_number, text, similarity_score}, ... ]
        }
    """
    if vector_store is None or vector_store.is_empty():
        return {
            "status": "empty",
            "message": "No papers have been processed into the knowledge base yet.",
            "query": query,
            "chunks": [],
        }

    chunks = vector_store.search(query, top_k=top_k, source_filter=source_filter)

    if not chunks:
        # This is the "empty retrieval results" error-handling case -
        # can happen if source_filter excludes everything, or (rarely)
        # if the query is very unusual. We surface this clearly rather
        # than silently returning nothing to the LLM.
        return {
            "status": "empty",
            "message": "No relevant content was found for this query in the "
                       "uploaded papers.",
            "query": query,
            "chunks": [],
        }

    return {
        "status": "ok",
        "message": f"Retrieved {len(chunks)} relevant chunk(s).",
        "query": query,
        "chunks": chunks,
    }
