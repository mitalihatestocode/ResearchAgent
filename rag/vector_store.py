"""
Vector Store (FAISS + BM25 hybrid)
-----------------------------------
FAISS stores vectors and can very quickly find "the vectors most similar
to this query vector." It does NOT store our metadata (paper name, page
number, chunk text) - it only knows about numbers. So we keep a parallel
Python list of chunk metadata, where list index == FAISS vector index.

We use IndexFlatIP (Inner Product). Because our embeddings are
normalized (see embeddings.py), inner product is equivalent to cosine
similarity - a standard way to measure "how similar in meaning."

HYBRID SEARCH (viva point):
Alongside FAISS we keep a BM25 keyword index over the same chunks.
search() supports three modes:
  - "dense":   embeddings only (semantic similarity)
  - "keyword": BM25 only (exact terms - model names, dataset names, numbers)
  - "hybrid":  both, fused with Reciprocal Rank Fusion (RRF):
               score = sum over rankers of 1 / (RRF_K + rank)
RRF is the standard way to merge rankings whose raw scores are on
different scales (cosine is 0..1, BM25 is unbounded), with no tuning.

The store is in-memory (rebuilt per session) - Streamlit's session_state
keeps it alive for the duration of the user's session.
"""

import faiss
import numpy as np

from rag.embeddings import embed_texts
from rag.bm25 import BM25Index

RRF_K = 60
SEARCH_MODES = ("hybrid", "dense", "keyword")


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.bm25 = BM25Index()
        self.chunk_metadata = []  # parallel list: chunk_metadata[i] describes vector i

    def add_chunks(self, chunks: list[dict], vectors=None):
        """
        Embed a list of chunks and add them to both indexes.

        Args:
            chunks: list of dicts from chunker.chunk_paper(), each with
                    "chunk_id", "source_name", "page_number", "text"
            vectors: optional precomputed embeddings (e.g. loaded from a
                     user's saved library) - skips re-embedding entirely

        Returns:
            the embedding matrix used, so callers can persist it
        """
        if not chunks:
            return np.zeros((0, self.index.d), dtype="float32")

        texts = [c["text"] for c in chunks]
        if vectors is None or len(vectors) != len(chunks):
            vectors = embed_texts(texts)
        vectors = np.asarray(vectors).astype("float32")

        self.index.add(vectors)
        self.bm25.add(texts)
        self.chunk_metadata.extend(chunks)
        return vectors

    def search(self, query: str, top_k: int = 5, source_filter: list[str] = None, mode: str = "hybrid") -> list[dict]:
        """
        Find the top_k chunks most relevant to the query.

        Args:
            query: the user's natural language question
            top_k: how many chunks to return
            source_filter: optional list of paper source_names to restrict
                            the search to. If None, searches all papers.
            mode: "hybrid" | "dense" | "keyword"

        Returns:
            List of chunk dicts (same shape as input chunks) each with added
            "similarity_score" (cosine), "bm25_score" and "fused_score",
            sorted best-first.
        """
        n = self.index.ntotal
        if n == 0:
            return []
        if mode not in SEARCH_MODES:
            mode = "hybrid"

        # The corpus is small (a few papers -> a few hundred chunks), so we
        # score every chunk exactly; this also makes source filtering trivial.
        query_vector = np.array(embed_texts([query])).astype("float32")
        dense_scores, dense_ids = self.index.search(query_vector, n)
        cosine = {int(i): float(s) for s, i in zip(dense_scores[0], dense_ids[0]) if i != -1}
        bm25_scores = self.bm25.scores(query)

        allowed = [
            i for i in range(n)
            if not source_filter or self.chunk_metadata[i]["source_name"] in source_filter
        ]
        if not allowed:
            return []

        dense_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -cosine.get(i, 0.0)), start=1)}
        keyword_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -bm25_scores[i]), start=1)}

        fused = {}
        for i in allowed:
            if mode == "dense":
                fused[i] = cosine.get(i, 0.0)
            elif mode == "keyword":
                fused[i] = bm25_scores[i]
            else:
                score = 1.0 / (RRF_K + dense_rank[i])
                if bm25_scores[i] > 0:  # a chunk with zero keyword overlap gets no keyword vote
                    score += 1.0 / (RRF_K + keyword_rank[i])
                fused[i] = score

        ranked = sorted(allowed, key=lambda i: -fused[i])[:top_k]

        results = []
        for i in ranked:
            result = dict(self.chunk_metadata[i])  # copy so we don't mutate stored metadata
            result["similarity_score"] = cosine.get(i, 0.0)
            result["bm25_score"] = float(bm25_scores[i])
            result["fused_score"] = float(fused[i])
            results.append(result)
        return results

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def stats(self) -> dict:
        per_paper = {}
        for chunk in self.chunk_metadata:
            per_paper[chunk["source_name"]] = per_paper.get(chunk["source_name"], 0) + 1
        return {"total_chunks": self.index.ntotal, "chunks_per_paper": per_paper}
