"""
Quick manual test for the full Phase 2 RAG pipeline:
PDF -> extract -> chunk -> embed -> FAISS index -> retrieve

Usage:
    python tests/test_retriever.py path/to/paper1.pdf path/to/paper2.pdf ...
    python tests/test_retriever.py path/to/paper1.pdf --query "What dataset was used?"

If you don't pass --query, it defaults to a generic test question.
Note: first run will download the embedding model (~80MB), so it may
take a minute. Subsequent runs are fast.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pdf_extractor import extract_text_from_pdf
from rag.chunker import chunk_paper
from rag.vector_store import VectorStore
from rag.retriever import retrieve_relevant_chunks


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python tests/test_retriever.py paper1.pdf [paper2.pdf ...] [--query \"...\"]")
        sys.exit(1)

    query = "What dataset or methodology was used?"
    if "--query" in args:
        idx = args.index("--query")
        query = args[idx + 1]
        pdf_paths = args[:idx]
    else:
        pdf_paths = args

    print("Building knowledge base...")
    store = VectorStore()
    total_chunks = 0

    for path in pdf_paths:
        result = extract_text_from_pdf(path, os.path.basename(path))
        print(f"  {path}: status={result['status']}, pages={result['num_pages']}")

        if result["status"] == "ok":
            chunks = chunk_paper(result)
            store.add_chunks(chunks)
            total_chunks += len(chunks)
            print(f"    -> {len(chunks)} chunks added")

    print(f"\nTotal chunks in knowledge base: {total_chunks}")
    print(f"Query: {query}\n")

    result = retrieve_relevant_chunks(store, query, top_k=3)
    print(f"Status: {result['status']} - {result['message']}\n")

    for i, chunk in enumerate(result["chunks"], start=1):
        print(f"--- Result {i} (score={chunk['similarity_score']:.3f}) ---")
        print(f"Source: {chunk['source_name']}, page {chunk['page_number']}")
        print(chunk["text"][:300])
        print()


if __name__ == "__main__":
    main()
