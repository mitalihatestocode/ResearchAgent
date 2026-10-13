"""
Vector Store (FAISS)
---------------------
FAISS stores vectors and can very quickly find "the vectors most similar
to this query vector." It does NOT store our metadata (paper name, page
number, chunk text) - it only knows about numbers. So we keep a parallel
Python list of chunk metadata, where list index == FAISS vector index.
This is a standard, simple pattern and easy to explain in viva.

We use IndexFlatIP (Inner Product). Because our embeddings are
normalized (see embeddings.py), inner product is equivalent to cosine
similarity - a standard way to measure "how similar in meaning."

This class is deliberately kept in-memory only for now (rebuilt each
time papers are uploaded in a session) rather than persisted to disk -
simpler for a student project, and Streamlit's session_state keeps it
alive for the duration of the user's session.
"""

import faiss
import numpy as np

from rag.embeddings import embed_texts


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunk_metadata = []  # parallel list: chunk_metadata[i] describes vector i

    def add_chunks(self, chunks: list[dict]):
        """
        Embed a list of chunks and add them to the index.

        Args:
            chunks: list of dicts from chunker.chunk_paper(), each with
                    "chunk_id", "source_name", "page_number", "text"
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)
        vectors = np.array(vectors).astype("float32")

        self.index.add(vectors)
        self.chunk_metadata.extend(chunks)

    def search(self, query: str, top_k: int = 5, source_filter: list[str] = None) -> list[dict]:
        """
        Find the top_k chunks most similar in meaning to the query.

        Args:
            query: the user's natural language question
            top_k: how many chunks to return
            source_filter: optional list of paper source_names to restrict
                            the search to (e.g. only search paper1.pdf).
                            If None, searches across all uploaded papers.

        Returns:
            List of chunk dicts (same shape as input chunks) each with
            an added "similarity_score" field, sorted best-first.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = embed_texts([query]).astype("float32")

        # If filtering by source, we need to search more broadly then
        # filter, since FAISS's basic IndexFlatIP doesn't support
        # filtering natively. Simple and fine at our scale (a handful
        # of papers, a few hundred chunks).
        search_k = self.index.ntotal if source_filter else top_k
        scores, indices = self.index.search(query_vector, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunk_metadata[idx]

            if source_filter and chunk["source_name"] not in source_filter:
                continue

            result = dict(chunk)  # copy so we don't mutate stored metadata
            result["similarity_score"] = float(score)
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def is_empty(self) -> bool:
        return self.index.ntotal == 0
