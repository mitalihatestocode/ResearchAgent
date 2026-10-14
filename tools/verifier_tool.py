"""
Answer Verifier (self-reflection)
---------------------------------
After the agent drafts an answer, this tool audits it BEFORE the user
sees it. Two independent checks:

1. DETERMINISTIC CITATION CHECK (plain Python, cannot be fooled):
   - every [S#] cited in the answer must exist in the evidence set
   - a substantive answer with no citations at all is flagged

2. LLM-AS-JUDGE CHECK (agent/prompts.py: verifier_prompt):
   a separate, temperature-0 call that checks each claim against the
   evidence and lists unsupported claims.

If either check fails, the orchestrator sends the feedback back to the
answer tool for ONE revision pass (a "reflexion" loop), then re-verifies.
The final verdict and confidence are shown to the user next to the answer.
"""

import re

from agent.llm_client import call_llm_json, LLMError
from agent.prompts import verifier_prompt, TEMPERATURE

CITATION_RE = re.compile(r"\[(S\d+)\]")


def check_citations(answer: str, valid_ids: set[str]) -> dict:
    cited = set(CITATION_RE.findall(answer))
    invalid = sorted(cited - valid_ids)
    # "not found / not reported" style answers legitimately have no citations
    honest_no_answer = bool(re.search(r"(do(es)? not|could not|cannot|not) (report|find|mention|state|contain)", answer, re.I))
    missing = not cited and not honest_no_answer and len(answer) > 80
    return {
        "cited_ids": sorted(cited),
        "invalid_ids": invalid,
        "missing_citations": missing,
        "ok": not invalid and not missing,
    }


def verify_answer(question: str, answer: str, evidence_text: str, valid_ids: set[str]) -> dict:
    """
    Returns:
        {
            "status": "ok" | "error",
            "verdict": "supported" | "partially_supported" | "unsupported" | "unverified",
            "confidence": float,
            "unsupported_claims": [str, ...],
            "citation_check": {...},
            "feedback": str,       # what to fix, empty if nothing
            "reasoning": str,
        }
    """
    citation_check = check_citations(answer, valid_ids)

    feedback_lines = []
    if citation_check["invalid_ids"]:
        feedback_lines.append(f"These citation ids do not exist in the evidence: {', '.join(citation_check['invalid_ids'])}.")
    if citation_check["missing_citations"]:
        feedback_lines.append("The answer makes claims without any [S#] citations.")

    try:
        system, user = verifier_prompt(question, answer, evidence_text)
        judgement = call_llm_json(system, user, temperature=TEMPERATURE["verifier"])
    except LLMError as e:
        return {
            "status": "error",
            "verdict": "unverified",
            "confidence": 0.0,
            "unsupported_claims": [],
            "citation_check": citation_check,
            "feedback": " ".join(feedback_lines),
            "reasoning": f"Verifier unavailable: {e}",
        }

    verdict = judgement.get("verdict", "unverified")
    if verdict not in ("supported", "partially_supported", "unsupported"):
        verdict = "unverified"
    unsupported = [str(c) for c in (judgement.get("unsupported_claims") or []) if str(c).strip()]
    try:
        confidence = max(0.0, min(1.0, float(judgement.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    # Deterministic failures override an optimistic LLM verdict
    if not citation_check["ok"] and verdict == "supported":
        verdict = "partially_supported"

    for claim in unsupported:
        feedback_lines.append(f"Unsupported claim: {claim}")

    return {
        "status": "ok",
        "verdict": verdict,
        "confidence": confidence,
        "unsupported_claims": unsupported,
        "citation_check": citation_check,
        "feedback": "\n".join(feedback_lines),
        "reasoning": judgement.get("reasoning", ""),
    }
