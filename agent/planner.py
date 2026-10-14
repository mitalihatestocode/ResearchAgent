"""
Planner
-------
This is the core of the agent's "planning" behavior (rubric: Agent
Architecture & Planning). Given the user's request, the Planner asks
the LLM to decide which tools are actually needed - rather than the
orchestrator always running a fixed hardcoded sequence.

The planner sees the tool registry's descriptions (agent/tool_registry.py)
plus few-shot examples, and returns:
  - reasoning      : why these tools (shown to the user)
  - in_scope       : whether the request is about the papers at all
  - intent         : factual_question / comparison / research_gaps / ...
  - target_papers  : which papers the question is about ([] = all)
  - steps          : ordered tool names, each with a "why"

Every field is validated: unknown tool names and unknown paper names are
dropped, so a hallucinated plan can never reach the executor.

FALLBACK: if the LLM is unavailable or returns garbage, a rule-based
keyword planner (heuristic_plan) produces a sensible plan instead, so
the agent degrades gracefully rather than stopping.
"""

import re

from agent.llm_client import call_llm_json, LLMError
from agent.prompts import planning_prompt, TEMPERATURE
from agent.tool_registry import planner_catalog, TOOLS_BY_NAME

INTENTS = {"factual_question", "paper_summary", "comparison", "research_gaps",
           "related_work", "full_review", "out_of_scope"}


def heuristic_plan(user_query: str, reason: str = "") -> dict:
    """Keyword-based fallback planner used when the LLM planner is unavailable."""
    q = user_query.lower()
    steps, intent = [], "factual_question"

    if re.search(r"\b(gap|gaps|future work|open problem|research direction|new idea)", q):
        steps = ["extract_metadata", "analyze_papers", "compare_papers", "identify_research_gaps", "generate_report"]
        intent = "research_gaps"
    elif re.search(r"\b(compare|comparison|contrast|differ|difference|similar)", q):
        steps = ["extract_metadata", "analyze_papers", "compare_papers"]
        intent = "comparison"
    elif re.search(r"\b(related|recent papers|literature|other papers|similar work)", q):
        steps = ["extract_metadata", "search_related_work"]
        intent = "related_work"
    elif re.search(r"\b(report|review|summari[sz]e|overview|analy[sz]e)", q):
        steps = ["extract_metadata", "analyze_papers", "generate_report"]
        intent = "full_review"
    else:
        steps = ["rag_retrieval", "answer_question"]

    return {
        "status": "fallback",
        "reasoning": f"Rule-based fallback plan{f' ({reason})' if reason else ''}: matched keywords for intent '{intent}'.",
        "in_scope": True,
        "intent": intent,
        "target_papers": [],
        "steps": steps,
        "step_reasons": {},
    }


def _normalise_steps(raw_steps) -> tuple[list[str], dict]:
    """Accepts [{"tool":..,"why":..}] or ["tool", ...]; keeps known tools only."""
    steps, reasons = [], {}
    for item in raw_steps or []:
        if isinstance(item, dict):
            name, why = str(item.get("tool", "")).strip(), str(item.get("why", "")).strip()
        else:
            name, why = str(item).strip(), ""
        if name in TOOLS_BY_NAME and TOOLS_BY_NAME[name].planner_visible and name not in steps:
            steps.append(name)
            if why:
                reasons[name] = why
    return steps, reasons


def create_plan(user_query: str, paper_names: list[str], history: list[dict] = None) -> dict:
    """
    Args:
        user_query: the user's natural-language request
        paper_names: list of uploaded paper source_names available
        history: previous conversation turns [{"query", "answer"}]

    Returns:
        {
            "status": "ok" | "fallback",
            "reasoning": str,
            "in_scope": bool,
            "intent": str,
            "target_papers": [str, ...],   # subset of paper_names, [] = all
            "steps": [str, ...],           # subset of the tool registry
            "step_reasons": {step: why},
        }
    """
    try:
        system, user = planning_prompt(user_query, paper_names, planner_catalog(), history)
        plan = call_llm_json(system, user, temperature=TEMPERATURE["planner"])
    except LLMError as e:
        return heuristic_plan(user_query, f"planner LLM unavailable: {str(e)[:120]}")

    steps, reasons = _normalise_steps(plan.get("steps"))
    in_scope = plan.get("in_scope", True) is not False
    intent = plan.get("intent") if plan.get("intent") in INTENTS else ("factual_question" if in_scope else "out_of_scope")

    # Only keep target papers that really exist (case-insensitive match)
    lookup = {name.lower(): name for name in paper_names}
    targets = []
    for name in plan.get("target_papers") or []:
        match = lookup.get(str(name).strip().lower())
        if match and match not in targets:
            targets.append(match)
    if len(targets) == len(paper_names):
        targets = []

    if in_scope and not steps:
        return heuristic_plan(user_query, "planner returned no valid steps")

    return {
        "status": "ok",
        "reasoning": str(plan.get("reasoning", "")),
        "in_scope": in_scope,
        "intent": intent if in_scope else "out_of_scope",
        "target_papers": targets,
        "steps": steps if in_scope else [],
        "step_reasons": reasons,
    }
