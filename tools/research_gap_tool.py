"""
Research Gap Analysis Tool
-----------------------------
Takes author-reported limitations (already extracted per-paper in
Phase 3) and the differences identified during comparison (Phase 4's
comparison_tool), and asks the LLM to suggest potential research
directions.

CRITICAL SAFETY FRAMING (rubric requirement):
Every suggestion here is explicitly a "potential research direction",
never a claimed discovery or guaranteed novel contribution. The LLM
cannot verify novelty against the full body of existing literature, so
we never let it - or the UI - imply otherwise. This is enforced both in
the prompt (agent/prompts.py: research_gap_prompt) and in how results
are labeled when displayed.
"""

from schemas.paper_schema import PaperInfo
from agent.llm_client import call_llm_json, LLMError
from agent.prompts import research_gap_prompt


def identify_research_gaps(paper_infos: list[PaperInfo], comparison_differences: str = "") -> dict:
    """
    Args:
        paper_infos: list of analyzed PaperInfo objects (their
                     "limitations" field must be populated from Phase 3)
        comparison_differences: the "differences" narrative from
                                 comparison_tool.compare_papers(), if
                                 available (adds useful context but not
                                 required)

    Returns:
        {
            "status": "ok" | "error" | "no_limitations_found",
            "message": str,
            "author_reported_limitations": [ {source_name, limitation}, ... ],
            "research_gaps": [ {"direction": str, "based_on": str}, ... ],
        }
    """
    limitations_entries = [
        {"source_name": info.source_name, "limitation": info.limitations}
        for info in paper_infos
        if info.limitations and "Not explicitly reported" not in info.limitations
    ]

    if not limitations_entries:
        # Graceful handling of the "missing information" case: if no
        # paper reported any limitations, we can't ground gap suggestions
        # in anything real, so we say so rather than inventing gaps.
        return {
            "status": "no_limitations_found",
            "message": "None of the analyzed papers had explicitly reported "
                       "limitations to base research gap suggestions on.",
            "author_reported_limitations": [],
            "research_gaps": [],
        }

    limitations_summary = "\n".join(
        f"- [{e['source_name']}] {e['limitation']}" for e in limitations_entries
    )

    try:
        system, user = research_gap_prompt(limitations_summary, comparison_differences or "Not provided.")
        result = call_llm_json(system, user)

        gaps = result.get("research_gaps", [])
        # Basic shape check - drop anything malformed rather than crash
        # the UI on a missing key.
        valid_gaps = [
            g for g in gaps
            if isinstance(g, dict) and "direction" in g
        ]

        return {
            "status": "ok",
            "message": f"Identified {len(valid_gaps)} potential research direction(s) "
                       f"based on {len(limitations_entries)} reported limitation(s).",
            "author_reported_limitations": limitations_entries,
            "research_gaps": valid_gaps,
        }

    except LLMError as e:
        return {
            "status": "error",
            "message": str(e),
            "author_reported_limitations": limitations_entries,
            "research_gaps": [],
        }
