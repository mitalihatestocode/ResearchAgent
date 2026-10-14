"""
Run the RAG benchmark.

    python -m evaluation.run_eval --retrieval-only   # no LLM: compares dense / keyword / hybrid retrieval
    python -m evaluation.run_eval                    # + full agent runs with the real LLM (needs GROQ_API_KEY)

Writes evaluation/results.json (shown in the app's Evaluation tab) and
docs/evaluation_results.md (for the report).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from evaluation.metrics import retrieval_metrics, answer_metrics, mean

BENCHMARK = os.path.join(ROOT, "evaluation", "benchmark.json")
RESULTS_JSON = os.path.join(ROOT, "evaluation", "results.json")
RESULTS_MD = os.path.join(ROOT, "docs", "evaluation_results.md")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample_papers")
TOP_K = 6


def load_benchmark() -> dict:
    with open(BENCHMARK) as f:
        return json.load(f)


def build_store():
    from rag.pdf_extractor import extract_text_from_pdf
    from rag.chunker import chunk_paper
    from rag.vector_store import VectorStore
    papers, store = {}, VectorStore()
    for name in sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".pdf")):
        result = extract_text_from_pdf(os.path.join(SAMPLE_DIR, name), name)
        if result["status"] == "ok":
            papers[name] = result
            store.add_chunks(chunk_paper(result))
    return papers, store


def evaluate_retrieval(store, items, modes=("dense", "keyword", "hybrid"), k=TOP_K) -> dict:
    """
    Retrieval quality per search mode - no LLM involved. Two settings:
      unscoped : search all papers (hardest)
      scoped   : restricted to the paper the question names - what the agent
                 does when the planner sets target_papers
    """
    positives = [i for i in items if i["expected"]]
    summary = {}
    for scope in ("unscoped", "scoped"):
        for mode in modes:
            rows = []
            for item in positives:
                source_filter = [item["paper"]] if scope == "scoped" else None
                chunks = store.search(item["question"], top_k=k, mode=mode, source_filter=source_filter)
                rows.append({"id": item["id"], **retrieval_metrics(chunks, item)})
            summary[f"{mode} ({scope})"] = {
                "hit_at_k": mean(r["hit"] for r in rows),
                "mrr": mean(r["reciprocal_rank"] for r in rows),
                "context_recall": mean(r["context_recall"] for r in rows),
                "per_item": rows,
            }
    return {"k": k, "n_items": len(positives), "modes": summary}


def evaluate_agent(papers, store, items, on_progress=print) -> dict:
    from agent.orchestrator import run_agent
    cache, rows = {}, []
    for item in items:
        start = time.time()
        try:
            result = run_agent(item["question"], papers, store, cache=cache)
            metrics = answer_metrics(result, item)
            answer = result["final_answer"]
            steps = result["effective_steps"]
        except Exception as e:  # keep evaluating the rest
            metrics, answer, steps = {"exact": False, "fact_recall": 0.0, "error": str(e)}, "", []
        rows.append({"id": item["id"], "question": item["question"], "steps": steps,
                     "answer": answer[:1200], **metrics})
        on_progress(f"{item['id']}: exact={metrics.get('exact')} recall={metrics.get('fact_recall')} "
                    f"({time.time() - start:.1f}s)")
    answerable = [r for r, i in zip(rows, items) if i["expected"]]
    unanswerable = [r for r, i in zip(rows, items) if i.get("expect_refusal")]
    return {
        "n_items": len(rows),
        "answer_accuracy": mean(r["exact"] for r in answerable),
        "mean_fact_recall": mean(r["fact_recall"] for r in answerable),
        "citation_validity": mean(r.get("citation_valid") for r in answerable),
        "faithfulness": mean(r.get("faithful") for r in answerable),
        "refusal_accuracy": mean(r.get("refusal_correct") for r in unanswerable),
        "mean_latency_s": mean(r.get("duration_s") for r in rows),
        "mean_llm_calls": mean(r.get("llm_calls") for r in rows),
        "mean_tokens": mean(r.get("tokens") for r in rows),
        "per_item": rows,
    }


def write_markdown(results: dict):
    r = results["retrieval"]
    lines = ["# Evaluation Results", "",
             f"_Run {results['timestamp']} · benchmark `{results['benchmark']}` · "
             f"{r['n_items']} answerable + {results['n_unanswerable']} unanswerable questions · "
             f"papers: the 3 sample arXiv papers_", "",
             f"## Retrieval quality (top-{r['k']}, no LLM)", "",
             "| Search mode | Hit@k | MRR | Context recall |", "|---|---|---|---|"]
    for mode, m in r["modes"].items():
        lines.append(f"| {mode} | {m['hit_at_k']:.2f} | {m['mrr']:.2f} | {m['context_recall']:.2f} |")
    lines += ["", "_Hit@k: a chunk from the right paper containing an expected fact is in the top k. "
              "MRR: mean reciprocal rank of the first such chunk. Context recall: share of expected facts "
              "present in the retrieved chunks._", ""]

    a = results.get("agent")
    if a:
        lines += ["## End-to-end answer quality (real LLM)", "",
                  "| Metric | Value |", "|---|---|",
                  f"| Answer accuracy (all expected facts present) | {a['answer_accuracy']:.0%} |",
                  f"| Mean fact recall | {a['mean_fact_recall']:.0%} |",
                  f"| Citation validity (code check) | {a['citation_validity']:.0%} |",
                  f"| Faithfulness (verifier: supported) | {a['faithfulness']:.0%} |",
                  f"| Correct refusals on unanswerable / out-of-scope | {a['refusal_accuracy']:.0%} |",
                  f"| Mean latency | {a['mean_latency_s']:.1f}s |",
                  f"| Mean LLM calls / tokens per question | {a['mean_llm_calls']:.1f} / {a['mean_tokens']:.0f} |",
                  "", "### Per question", "", "| ID | Question | Exact | Fact recall | Citations ok | Faithful | Plan |",
                  "|---|---|---|---|---|---|---|"]
        for row in a["per_item"]:
            fmt = lambda v: "–" if v is None else ("✅" if v is True else ("❌" if v is False else f"{v:.0%}"))
            lines.append(f"| {row['id']} | {row['question']} | {fmt(row.get('exact'))} | {fmt(row.get('fact_recall'))} | "
                         f"{fmt(row.get('citation_valid'))} | {fmt(row.get('faithful'))} | {' → '.join(row['steps']) or '—'} |")
    os.makedirs(os.path.dirname(RESULTS_MD), exist_ok=True)
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM-based end-to-end evaluation")
    args = parser.parse_args()

    bench = load_benchmark()
    items = bench["items"]
    print("Indexing sample papers...")
    papers, store = build_store()
    if len(papers) < 3:
        sys.exit("Sample papers missing - run: python scripts/download_sample_papers.py")

    results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "benchmark": bench["name"],
               "n_unanswerable": sum(1 for i in items if i.get("expect_refusal")),
               "retrieval": evaluate_retrieval(store, items)}
    for mode, m in results["retrieval"]["modes"].items():
        print(f"retrieval [{mode:7}] hit@{TOP_K}={m['hit_at_k']:.2f} mrr={m['mrr']:.2f} recall={m['context_recall']:.2f}")

    if not args.retrieval_only:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("GROQ_API_KEY not set - use --retrieval-only or add it to .env")
        results["agent"] = evaluate_agent(papers, store, items)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    write_markdown(results)
    print(f"Saved {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    main()
