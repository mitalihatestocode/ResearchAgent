"""
Grounded Answer Tool
--------------------
Writes the natural-language answer to the user's question, using ONLY
the evidence retrieved by the RAG step (and, when available, the
structured per-paper analysis). Every claim must carry an [S#] citation
that maps back to a specific paper + page.

When the retrieval step found nothing relevant, this tool does NOT call
the LLM at all - it returns an honest "not found" answer. That is a
deliberate safety choice: with no evidence, any LLM output would be
ungrounded by definition.
"""

import json

from agent.llm_client import call_llm_json, LLMError
from agent.prompts import answer_prompt, format_evidence, TEMPERATURE
from safety.guardrails import sanitize_evidence

NO_EVIDENCE_ANSWER = (
    "I could not find information relevant to this question in the uploaded papers, "
    "so I won't guess. Try rephrasing the question, or ask about the papers' "
    "datasets, methods, results or limitations."
)


def build_evidence(chunks: list[dict]) -> tuple[str, dict]:
    """Returns (evidence_text_for_prompt, {"S1": chunk, ...})."""
    safe_chunks = [dict(c, text=sanitize_evidence(c["text"])) for c in chunks]
    id_map = {f"S{i}": c for i, c in enumerate(chunks, start=1)}
    return format_evidence(safe_chunks, id_prefix="S"), id_map


def structured_context(paper_infos: list) -> str:
    if not paper_infos:
        return ""
    return json.dumps([info.to_display_dict() for info in paper_infos], indent=1)[:6000]


def answer_question(question: str, chunks: list[dict], paper_infos: list = None,
                    history: list[dict] = None, feedback: str = "") -> dict:
    """
    Args:
        question: the user's question
        chunks: retrieved evidence chunks (from rag_retrieval)
        paper_infos: optional PaperInfo list for extra structured context
        history: previous conversation turns (for follow-up questions)
        feedback: verifier feedback when this is a revision pass

    Returns:
        {
            "status": "ok" | "no_evidence" | "error",
            "message": str,
            "answer": str (markdown with [S#] citations),
            "answerable": "yes" | "partial" | "no",
            "sources": {"S1": {source_name, page_number, section, text}, ...},
            "evidence_text": str,   # kept for the verifier
        }
    """
    if not chunks and not paper_infos:
        return {
            "status": "no_evidence",
            "message": "No relevant evidence was retrieved, so no LLM answer was generated.",
            "answer": NO_EVIDENCE_ANSWER,
            "answerable": "no",
            "sources": {},
            "evidence_text": "",
        }

    evidence_text, id_map = build_evidence(chunks)

    try:
        system, user = answer_prompt(
            question, evidence_text or "(no passages retrieved)",
            structured_context(paper_infos), history, feedback,
        )
        result = call_llm_json(system, user, temperature=TEMPERATURE["answer"])
    except LLMError as e:
        return {
            "status": "error", "message": str(e), "answer": "",
            "answerable": "no", "sources": {}, "evidence_text": evidence_text,
        }

    answer = str(result.get("answer", "")).strip()
    if not answer:
        return {
            "status": "error", "message": "The LLM returned an empty answer.", "answer": "",
            "answerable": "no", "sources": {}, "evidence_text": evidence_text,
        }

    return {
        "status": "ok",
        "message": f"Answer generated from {len(chunks)} evidence passage(s).",
        "answer": answer,
        "answerable": result.get("answerable", "yes"),
        "sources": {
            sid: {k: c.get(k) for k in ("source_name", "page_number", "section", "text", "similarity_score")}
            for sid, c in id_map.items()
        },
        "evidence_text": evidence_text,
    }
