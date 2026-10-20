"""Measures search and answer quality on tests/benchmark.json: 12 questions with known answers (checked against the
papers' text) and 3 questions the agent should decline. Needs the 3 sample papers (click "Load 3 sample papers" in
the app once) and, for the answer part, GROQ_API_KEY in .env.

    python tests/evaluate.py                 # search + answers (about 5 minutes on the free tier)
    python tests/evaluate.py --search-only   # search quality only, no API key needed"""
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

import rag  # noqa: E402
from agent import LLM, run_agent  # noqa: E402

REFUSAL = re.compile(r"not (explicitly )?(report|mention|state|provide|specif|disclose)|do(es)? not (report|mention|state|"
                     r"provide|specify|include|contain)|no (information|mention|data|details?|figures?)|couldn't find"
                     r"|won't guess|doesn't seem to be about|not (available|found|given|disclosed)|unavailable")


def norm(text):
    """Lower-case, and turn the special hyphens/spaces language models like to use into normal ones."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[‐-―−]", "-", re.sub(r"[   ]", " ", text)).lower()


def recall(text, expected):
    """Share of expected facts found; each fact is a list of accepted spellings."""
    hits = [any(re.search(r"(?<![a-z0-9])" + re.escape(norm(a)) + r"(?![a-z0-9])", norm(text)) for a in group)
            for group in expected]
    return sum(hits) / len(hits) if hits else 1.0


def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    with open(os.path.join(ROOT, "tests", "benchmark.json")) as f:
        items = json.load(f)["items"]
    folder = os.path.join(ROOT, "data", "sample_papers")
    papers, index = {}, rag.Index()
    for name in {i["paper"] for i in items if i["paper"]}:
        papers[name] = rag.read_pdf(os.path.join(folder, name), name)
        index.add(rag.chunk(papers[name]))

    lines = ["# Evaluation results", "", "## Search quality (top 6 passages, no LLM)", "",
             "| Search | All papers: hit@6 | Paper named in the question: hit@6 |", "|---|---|---|"]
    answerable = [i for i in items if i["expected"]]
    for mode in ("dense", "keyword", "hybrid"):
        scores = []
        for scoped in (False, True):
            hits = [any(recall(c["text"], i["expected"]) > 0 for c in index.search(
                i["question"], 6, [i["paper"]] if scoped else None, mode) if c["paper"] == i["paper"]) for i in answerable]
            scores.append(f"{sum(hits) / len(hits):.2f}")
        lines.append(f"| {mode} | {scores[0]} | {scores[1]} |")
    if "--search-only" not in sys.argv:
        rows, cache = [], {}
        for i in items:
            result = run_agent(i["question"], papers, index, LLM(), cache=cache)
            sources = result["results"].get("answer_question", {}).get("sources", {})
            cited_ok = all(s in sources for s in re.findall(r"\[(S\d+)\]", result["answer"]))
            verdict = result["results"].get("verify_answer", {}).get("verdict")
            score = recall(result["answer"], i["expected"]) if i["expected"] else float(bool(REFUSAL.search(norm(result["answer"]))))
            rows.append((i, score, cited_ok, verdict, result["seconds"], result["answer"]))
            print(f"{i['id']}: {score:.2f} ({result['seconds']}s)")
        ans = [r for r in rows if r[0]["expected"]]
        refuse = [r for r in rows if not r[0]["expected"]]
        lines += ["", "## Answers (real LLM)", "", "| Measure | Result |", "|---|---|",
                  f"| Fully correct answers | {sum(r[1] == 1 for r in ans)}/{len(ans)} |",
                  f"| Average share of expected facts found | {sum(r[1] for r in ans) / len(ans):.0%} |",
                  f"| Answers whose citations all exist | {sum(r[2] for r in ans)}/{len(ans)} |",
                  f"| Answers the fact-checker marked as supported | {sum(r[3] == 'supported' for r in ans)}/{len(ans)} |",
                  f"| Correctly declined (not in the papers / off topic) | {sum(r[1] == 1 for r in refuse)}/{len(refuse)} |",
                  f"| Average time per question | {sum(r[4] for r in rows) / len(rows):.1f}s |",
                  "", "| ID | Question | Facts found | Verdict | Answer (start) |", "|---|---|---|---|---|"]
        lines += [f"| {i['id']} | {i['question']} | {score:.0%} | {verdict or '-'} | "
                  f"{' '.join(answer.split())[:160].replace('|', '/')} |" for i, score, _, verdict, _, answer in rows]
    with open(os.path.join(ROOT, "docs", "evaluation_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
