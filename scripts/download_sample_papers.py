"""
Downloads three open-access arXiv papers on a shared topic (retrieval-
augmented generation) into data/sample_papers/ for demos and manual tests.

PDFs are git-ignored (we don't redistribute papers in the repo); run this
once after cloning:

    python scripts/download_sample_papers.py
"""

import os
import ssl
import sys
import urllib.request

SAMPLES = {
    "rag_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401",     # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "self_rag_2023.pdf": "https://arxiv.org/pdf/2310.11511",      # Self-RAG: Learning to Retrieve, Generate, and Critique
    "corrective_rag_2024.pdf": "https://arxiv.org/pdf/2401.15884",  # Corrective Retrieval Augmented Generation
}

TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_papers")


def main():
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    os.makedirs(TARGET_DIR, exist_ok=True)
    for name, url in SAMPLES.items():
        path = os.path.join(TARGET_DIR, name)
        if os.path.exists(path):
            print(f"already present: {name}")
            continue
        print(f"downloading {name} from {url} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/2.0 (academic project)"})
        try:
            with urllib.request.urlopen(request, timeout=60, context=context) as response, open(path, "wb") as f:
                f.write(response.read())
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
    print(f"Sample papers are in {TARGET_DIR}")


if __name__ == "__main__":
    main()
