"""
Manual RAG check (no LLM needed): PDF -> extract -> chunk -> embed -> index
-> retrieve, printing the top passages with dense, BM25 and fused scores.

Usage:
    python scripts/try_retriever.py data/sample_papers/*.pdf --query "What dataset was used?" --mode hybrid
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pdf_extractor import extract_text_from_pdf
from rag.chunker import chunk_paper
from rag.vector_store import VectorStore
from rag.retriever import retrieve_relevant_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+")
    parser.add_argument("--query", default="What dataset or methodology was used?")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "keyword"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    store = VectorStore()
    for path in args.pdfs:
        result = extract_text_from_pdf(path, os.path.basename(path))
        chunks = chunk_paper(result)
        store.add_chunks(chunks)
        print(f"{path}: {result['status']}, {result['num_pages']} pages, {len(chunks)} chunks")

    result = retrieve_relevant_chunks(store, args.query, top_k=args.top_k, mode=args.mode)
    print(f"\nQuery: {args.query} [{args.mode}]\n{result['message']}\n")
    for i, c in enumerate(result["chunks"], start=1):
        print(f"--- {i}. {c['source_name']} p.{c['page_number']} ({c['section']}) "
              f"cosine={c['similarity_score']:.3f} bm25={c['bm25_score']:.2f} fused={c['fused_score']:.4f}")
        print(c["text"][:300], "\n")


if __name__ == "__main__":
    main()
