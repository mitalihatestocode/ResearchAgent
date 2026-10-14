"""
Run the full agent from the command line (no Streamlit) and print the
plan, the step trace and the final answer. Handy for debugging and for
recording demo output.

Requires GROQ_API_KEY in .env (copy .env.example -> .env).

Usage:
    python scripts/run_agent_cli.py data/sample_papers/*.pdf --query "Compare these papers and identify research gaps"
    python scripts/run_agent_cli.py paper1.pdf paper2.pdf --query "What dataset did paper 1 use?" --report out.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag.pdf_extractor import extract_text_from_pdf
from rag.chunker import chunk_paper
from rag.vector_store import VectorStore
from agent.orchestrator import run_agent

ICON = {"ok": "[ok]", "warning": "[warn]", "error": "[ERR]", "skipped": "[skip]", "blocked": "[BLOCKED]", "info": "[info]"}


def build_knowledge_base(pdf_paths):
    papers, store = {}, VectorStore()
    for path in pdf_paths:
        name = os.path.basename(path)
        result = extract_text_from_pdf(path, name)
        print(f"  {name}: {result['status']} - {result['message']}")
        if result["status"] == "ok":
            papers[name] = result
            store.add_chunks(chunk_paper(result))
    return papers, store


def main():
    parser = argparse.ArgumentParser(description="Run ResearchAgent from the command line")
    parser.add_argument("pdfs", nargs="+", help="PDF files to analyse")
    parser.add_argument("--query", default="Compare these papers and identify potential research gaps.")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "keyword"])
    parser.add_argument("--report", help="write the generated report (if any) to this Markdown file")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")

    print("Building knowledge base...")
    papers, store = build_knowledge_base(args.pdfs)
    if not papers:
        sys.exit("No papers were successfully extracted.")

    print(f"\nQuery: {args.query}\n")
    result = run_agent(args.query, papers, store, settings={"retrieval_mode": args.mode},
                       on_event=lambda e: print(f"{ICON.get(e['status'], '-'):10} {e['step']:24} {e['detail']}"))

    print("\n" + "=" * 70 + "\nFINAL ANSWER\n" + "=" * 70)
    print(result["final_answer"])

    verification = result["step_results"].get("verify_answer")
    if verification and verification.get("verdict"):
        print(f"\nVerification: {verification['verdict']} (revised: {verification.get('revised', False)})")
    answer = result["step_results"].get("answer_question")
    if answer and answer.get("sources"):
        print("\nSources:")
        for sid, src in answer["sources"].items():
            print(f"  [{sid}] {src['source_name']} p.{src['page_number']} ({src['section']})")

    u = result["usage"]
    print(f"\n{u['calls']} LLM calls, {u['prompt_tokens'] + u['completion_tokens']} tokens, {result['duration_s']}s total")

    if args.report and "generate_report" in result["step_results"]:
        with open(args.report, "w") as f:
            f.write(result["step_results"]["generate_report"])
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
