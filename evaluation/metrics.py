"""
Evaluation metrics for the RAG benchmark (evaluation/benchmark.json).

Each benchmark item lists expected facts as groups of acceptable aliases,
e.g. [["Natural Questions", "NQ"], ["TriviaQA"]] - a fact counts as found
if ANY alias appears (case-insensitive, word-boundary match).

RETRIEVAL metrics (no LLM - cheap, deterministic):
  hit@k          : did any of the top-k chunks from the correct paper contain
                   at least one expected fact?
  context recall : share of expected facts present in the top-k chunks
  reciprocal rank: 1 / rank of the first chunk containing an expected fact

ANSWER metrics (full agent run):
  fact recall    : share of expected facts present in the final answer
  exact          : all expected facts present
  citation valid : every [S#] cited exists in the evidence (code check)
  faithful       : the verifier judged the answer "supported"
  refusal correct: for unanswerable / out-of-scope items, the agent declined
                   instead of inventing an answer
"""

import re
import unicodedata

# LLMs often emit typographic variants: non-breaking hyphens (RAG\u2011Token),
# en/em dashes and narrow no-break spaces (150\u202fk). Normalise before matching.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u2007\u202f\u2009\u200a"), " ")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_DASHES).translate(_SPACES).lower()

REFUSAL_PATTERNS = [
    r"\bnot (explicitly )?(report|reported|mention|mentioned|state|stated|provide|provided|specif)",
    r"\bdo(es)? not (report|mention|state|provide|specify|include|contain)",
    r"\b(could|can) ?not find\b", r"\bno (information|mention|data|details?)\b",
    r"doesn't appear to be about the uploaded research papers", r"\bwon't guess\b",
    r"\bnot (available|found|given|disclosed|included)\b", r"\bunavailable\b",
]


def _contains(text: str, alias: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(normalize(alias)) + r"(?![a-z0-9])"
    return re.search(pattern, normalize(text)) is not None


def facts_found(text: str, expected: list[list[str]]) -> list[bool]:
    return [any(_contains(text, alias) for alias in group) for group in expected]


def fact_recall(text: str, expected: list[list[str]]) -> float:
    if not expected:
        return 1.0
    found = facts_found(text, expected)
    return sum(found) / len(found)


def is_refusal(text: str) -> bool:
    lowered = normalize(text)
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def mentions_number(text: str) -> bool:
    """Used for 'not reported' items: an invented figure is a hallucination."""
    return re.search(r"(\$\s?\d|\d[\d,.]*\s?(usd|dollars|kg|tonnes?|tons?|co2|gpu[- ]hours))", (text or "").lower()) is not None


def retrieval_metrics(chunks: list[dict], item: dict) -> dict:
    """chunks: ranked retrieval results (each with source_name, text)."""
    expected = item["expected"]
    relevant_rank = None
    context = []
    for rank, chunk in enumerate(chunks, start=1):
        if item.get("paper") and chunk["source_name"] != item["paper"]:
            continue
        context.append(chunk["text"])
        if relevant_rank is None and any(facts_found(chunk["text"], expected)):
            relevant_rank = rank
    joined = "\n".join(context)
    return {
        "hit": relevant_rank is not None,
        "reciprocal_rank": 1.0 / relevant_rank if relevant_rank else 0.0,
        "context_recall": fact_recall(joined, expected),
    }


def answer_metrics(result: dict, item: dict) -> dict:
    answer = result.get("final_answer", "")
    verification = result.get("step_results", {}).get("verify_answer") or {}
    citation = verification.get("citation_check") or {}
    metrics = {
        "fact_recall": fact_recall(answer, item["expected"]),
        "citation_valid": bool(citation.get("ok")) if citation else None,
        "faithful": verification.get("verdict") == "supported" if verification.get("verdict") else None,
        "llm_calls": result.get("usage", {}).get("calls", 0),
        "tokens": result.get("usage", {}).get("prompt_tokens", 0) + result.get("usage", {}).get("completion_tokens", 0),
        "duration_s": result.get("duration_s", 0.0),
    }
    if item.get("expect_refusal"):
        metrics["refusal_correct"] = is_refusal(answer) and not mentions_number(answer)
        metrics["exact"] = metrics["refusal_correct"]
    else:
        metrics["exact"] = metrics["fact_recall"] == 1.0
    return metrics


def mean(values) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else 0.0
