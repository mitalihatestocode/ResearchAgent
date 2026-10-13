"""
Quick manual test for the full Phase 3 agent: plan -> execute tools.

Requires a GROQ_API_KEY in a .env file at the project root (copy
.env.example -> .env and fill in your key from https://console.groq.com/keys).

Usage:
    python tests/test_agent.py path/to/paper1.pdf path/to/paper2.pdf --query "Compare the datasets used"
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag.pdf_extractor import extract_text_from_pdf
from rag.chunker import chunk_paper
from rag.vector_store import VectorStore
from agent.orchestrator import run_agent
from agent.llm_client import LLMError


def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: python tests/test_agent.py paper1.pdf [paper2.pdf ...] --query "..."')
        sys.exit(1)

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    query = "Analyze these papers and summarize their methodology and dataset."
    if "--query" in args:
        idx = args.index("--query")
        query = args[idx + 1]
        pdf_paths = args[:idx]
    else:
        pdf_paths = args

    print("Extracting and indexing papers...")
    papers = {}
    store = VectorStore()

    for path in pdf_paths:
        name = os.path.basename(path)
        result = extract_text_from_pdf(path, name)
        print(f"  {name}: status={result['status']}, pages={result['num_pages']}")
        if result["status"] == "ok":
            papers[name] = result
            store.add_chunks(chunk_paper(result))

    if not papers:
        print("No papers were successfully extracted. Stopping.")
        sys.exit(1)

    print(f"\nQuery: {query}\n")
    print("Running agent (planning + executing)...\n")

    try:
        result = run_agent(query, papers, store)
    except LLMError as e:
        print(f"Agent failed: {e}")
        sys.exit(1)

    plan = result["plan"]
    print(f"Plan status: {plan['status']}")
    print(f"Reasoning: {plan['reasoning']}")
    print(f"Steps: {' -> '.join(plan['steps'])}\n")

    step_results = result["step_results"]

    if "extract_metadata" in step_results:
        print("--- METADATA ---")
        for name, meta in step_results["extract_metadata"].items():
            print(f"{name}: title={meta.get('title')}, authors={meta.get('authors')}, year={meta.get('year')}")
        print()

    if "analyze_papers" in step_results:
        print("--- PAPER ANALYSIS ---")
        paper_infos = result["paper_infos"]
        for name, analysis in step_results["analyze_papers"].items():
            if analysis["status"] != "ok":
                print(f"{name}: FAILED - {analysis['message']}")
                continue
            info = paper_infos[name]
            print(f"\n{name}:")
            for label, value in info.to_display_dict().items():
                print(f"  {label}: {value}")
        print()

    if "compare_papers" in step_results:
        print("--- COMPARISON ---")
        comparison = step_results["compare_papers"]
        print(f"Status: {comparison['status']} - {comparison['message']}")
        if comparison.get("table"):
            for row in comparison["table"]["rows"]:
                print(f"  {row['parameter']}: {row['values']}")
        if comparison["status"] == "ok":
            print(f"Similarities: {comparison['similarities']}")
            print(f"Differences: {comparison['differences']}")
        print()

    if "identify_research_gaps" in step_results:
        print("--- RESEARCH GAPS ---")
        gaps = step_results["identify_research_gaps"]
        print(f"Status: {gaps['status']} - {gaps['message']}")
        for i, gap in enumerate(gaps.get("research_gaps", []), start=1):
            print(f"  {i}. {gap['direction']} (based on: {gap.get('based_on', 'N/A')})")
        print()

    if "generate_report" in step_results:
        print("--- FINAL REPORT ---")
        print(step_results["generate_report"])
        print()

    if "rag_retrieval" in step_results:
        print("--- RETRIEVED EVIDENCE ---")
        retrieval = step_results["rag_retrieval"]
        for chunk in retrieval.get("chunks", []):
            print(f"[{chunk['source_name']} p{chunk['page_number']}, score={chunk['similarity_score']:.2f}]")
            print(f"  {chunk['text'][:200]}")


if __name__ == "__main__":
    main()
