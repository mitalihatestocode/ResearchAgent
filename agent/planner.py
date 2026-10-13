"""
Planner
-------
This is the core of the agent's "planning" behavior (rubric: Agent
Architecture & Planning). Given the user's request, the Planner asks
the LLM to decide which steps are actually needed - rather than the
orchestrator always running a fixed hardcoded sequence.

AVAILABLE_STEPS is the vocabulary the planner is allowed to choose from.
Keeping it as an explicit list (rather than letting the LLM invent step
names) keeps the plan safely executable - the orchestrator only knows
how to run steps from this exact list.
"""

from agent.llm_client import call_llm_json, LLMError
from agent.prompts import planning_prompt

AVAILABLE_STEPS = [
    "extract_metadata",        # pull title/authors/year for each paper
    "analyze_papers",          # structured extraction (dataset, methodology, etc.) per paper
    "rag_retrieval",           # a direct retrieval to answer a narrow factual question
    "compare_papers",          # compare structured info across 2+ papers (table + AI narrative)
    "identify_research_gaps",  # suggest potential research directions from limitations/differences
    "generate_report",         # assemble everything into one final structured report
]

# Used if the LLM call fails or returns something unusable - the agent
# should still be able to do SOMETHING useful rather than fully stop.
DEFAULT_FALLBACK_PLAN = {
    "reasoning": "Fallback plan (planner LLM call failed) - running standard analysis.",
    "steps": ["extract_metadata", "analyze_papers"],
}


def create_plan(user_query: str, paper_names: list[str]) -> dict:
    """
    Args:
        user_query: the user's natural-language request
        paper_names: list of uploaded paper source_names available

    Returns:
        {
            "status": "ok" | "fallback",
            "reasoning": str,
            "steps": [str, ...]   # subset of AVAILABLE_STEPS, in order
        }
    """
    try:
        system, user = planning_prompt(user_query, paper_names, AVAILABLE_STEPS)
        plan = call_llm_json(system, user)

        steps = plan.get("steps", [])
        # Safety check: only keep steps the orchestrator actually knows
        # how to execute. Silently drops anything the LLM hallucinated
        # rather than crashing.
        valid_steps = [s for s in steps if s in AVAILABLE_STEPS]

        if not valid_steps:
            return {
                "status": "fallback",
                "reasoning": "Planner returned no valid steps - using default plan.",
                "steps": DEFAULT_FALLBACK_PLAN["steps"],
            }

        return {
            "status": "ok",
            "reasoning": plan.get("reasoning", ""),
            "steps": valid_steps,
        }

    except LLMError as e:
        return {
            "status": "fallback",
            "reasoning": f"{DEFAULT_FALLBACK_PLAN['reasoning']} ({e})",
            "steps": DEFAULT_FALLBACK_PLAN["steps"],
        }
