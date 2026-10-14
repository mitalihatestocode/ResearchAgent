"""
Research Gap Analysis Tool
-----------------------------
Takes author-reported limitations (already extracted per paper by the
analysis tool) and the differences identified by the comparison tool,
and asks the LLM to suggest potential research directions.

CRITICAL SAFETY FRAMING (rubric requirement):
Every suggestion here is explicitly a "potential research direction",
never a claimed discovery or guaranteed novel contribution. This is
enforced in THREE independent layers:
  1. the prompt (agent/prompts.py: research_gap_prompt)
  2. a deterministic output guardrail (safety/guardrails.py:
     soften_overclaims) that rewrites words like "novel" or "first-ever"
  3. the search_related_work tool, which checks each gap against
     external literature so the user can see existing work
"""

from schemas.paper_schema import PaperInfo
from agent.llm_client import call_llm_json, LLMError
from agent.prompts import research_gap_prompt, TEMPERATURE
from safety.guardrails import soften_overclaims


def identify_research_gaps(paper_infos: list[PaperInfo], comparison_differences: str = "") -> dict:
    """
    Args:
        paper_infos: list of analyzed PaperInfo objects (their
                     "limitations" field must be populated)
        comparison_differences: the "differences" narrative from
                                 comparison_tool.compare_papers(), if
                                 available (adds useful context but not
                                 required)

    Returns:
        {
            "status": "ok" | "error" | "no_limitations_found",
            "message": str,
            "author_reported_limitations": [ {source_name, limitation}, ... ],
            "research_gaps": [ {"direction", "based_on", "search_query"}, ... ],
            "guardrail_changes": [str, ...],   # overclaims that were softened
        }
    """
    limitations_entries = [
        {"source_name": info.source_name, "limitation": info.limitations}
        for info in paper_infos
        if info.is_reported("limitations")
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
            "guardrail_changes": [],
        }

    limitations_summary = "\n".join(
        f"- [{e['source_name']}] {e['limitation']}" for e in limitations_entries
    )

    try:
        system, user = research_gap_prompt(limitations_summary, comparison_differences or "Not provided.")
        result = call_llm_json(system, user, temperature=TEMPERATURE["gaps"])
    except LLMError as e:
        return {
            "status": "error",
            "message": str(e),
            "author_reported_limitations": limitations_entries,
            "research_gaps": [],
            "guardrail_changes": [],
        }

    gaps = result.get("research_gaps", [])
    if not isinstance(gaps, list):
        gaps = []

    valid_gaps, guardrail_changes = [], []
    for gap in gaps:
        # Basic shape check - drop anything malformed rather than crash the UI
        if not isinstance(gap, dict) or not str(gap.get("direction", "")).strip():
            continue
        direction, changes = soften_overclaims(str(gap["direction"]))
        based_on, more_changes = soften_overclaims(str(gap.get("based_on", "N/A")))
        guardrail_changes.extend(changes + more_changes)
        valid_gaps.append({
            "direction": direction,
            "based_on": based_on,
            "search_query": str(gap.get("search_query") or direction)[:120],
        })

    return {
        "status": "ok",
        "message": f"Identified {len(valid_gaps)} potential research direction(s) "
                   f"based on {len(limitations_entries)} reported limitation(s).",
        "author_reported_limitations": limitations_entries,
        "research_gaps": valid_gaps,
        "guardrail_changes": guardrail_changes,
    }
