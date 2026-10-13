"""
Research Agent Orchestrator
-----------------------------
This is the agent's "body" - it takes a user query, asks the Planner
what steps are needed, then actually executes those steps by calling
the right tools in order. This is where "planning" and "tool use"
become a real working multi-step process rather than separate pieces.

DELIBERATELY NOT a black-box autonomous loop: the plan is decided once
up front, then executed step by step, with every step's result kept
so the UI can show the full trace (what was planned, what ran, what
came back). This is what makes the agent's behavior demonstrable and
explainable in viva rather than a mystery.

DEPENDENCY ENFORCEMENT (viva point):
compare_papers, identify_research_gaps, and generate_report all need
structured PaperInfo data to work with - which only exists once
extract_metadata + analyze_papers have run. Rather than trusting the
planner LLM to always remember this ordering, the orchestrator enforces
it directly: if a downstream step is requested, its prerequisites are
silently added. This keeps the agent's execution safe even if the
planning step is imperfect - a deliberate, explainable safety net.
"""

from agent.planner import create_plan
from tools.metadata_tool import extract_metadata
from tools.analysis_tool import analyze_paper
from tools.comparison_tool import compare_papers
from tools.research_gap_tool import identify_research_gaps
from tools.report_tool import generate_report
from rag.retriever import retrieve_relevant_chunks
from schemas.paper_schema import PaperInfo

# Steps are always executed in this fixed order, regardless of the
# order the planner LLM listed them in - later steps depend on earlier
# ones, so this ordering is a correctness requirement, not a preference.
CANONICAL_ORDER = [
    "extract_metadata",
    "analyze_papers",
    "rag_retrieval",
    "compare_papers",
    "identify_research_gaps",
    "generate_report",
]

# Steps that require PaperInfo (i.e. need extract_metadata + analyze_papers
# to have run first) - their prerequisites are auto-inserted if missing.
NEEDS_PAPER_INFO = {"compare_papers", "identify_research_gaps", "generate_report"}


def run_agent(user_query: str, papers: dict, vector_store) -> dict:
    """
    Args:
        user_query: the user's natural language request
        papers: dict of source_name -> extraction_result (from pdf_extractor),
                only successfully-extracted ("ok") papers should be passed in
        vector_store: the built VectorStore for RAG retrieval

    Returns:
        {
            "plan": { "status", "reasoning", "steps" },   # the ORIGINAL plan from the planner
            "effective_steps": [str, ...],                 # plan steps + auto-inserted prerequisites
            "step_results": { step_name: result, ... },
            "paper_infos": {source_name: PaperInfo, ...},  # empty if not built
        }
    """
    paper_names = list(papers.keys())
    plan = create_plan(user_query, paper_names)

    effective_steps = set(plan["steps"])
    if effective_steps & NEEDS_PAPER_INFO:
        effective_steps.add("extract_metadata")
        effective_steps.add("analyze_papers")

    # Execute in canonical order, skipping anything not requested
    ordered_steps = [s for s in CANONICAL_ORDER if s in effective_steps]

    step_results = {}
    paper_infos = {}

    for step in ordered_steps:
        if step == "extract_metadata":
            step_results["extract_metadata"] = {
                name: extract_metadata(result) for name, result in papers.items()
            }

        elif step == "analyze_papers":
            step_results["analyze_papers"] = {
                name: analyze_paper(name, vector_store) for name in paper_names
            }

            # Build PaperInfo objects now that both metadata and analysis exist
            metadata_results = step_results.get("extract_metadata", {})
            for name in paper_names:
                analysis = step_results["analyze_papers"][name]
                if analysis["status"] == "ok":
                    meta = metadata_results.get(name, {"status": "error"})
                    paper_infos[name] = PaperInfo.from_tool_outputs(name, meta, analysis)

        elif step == "rag_retrieval":
            step_results["rag_retrieval"] = retrieve_relevant_chunks(
                vector_store, user_query, top_k=5
            )

        elif step == "compare_papers":
            infos = list(paper_infos.values())
            step_results["compare_papers"] = compare_papers(infos)

        elif step == "identify_research_gaps":
            infos = list(paper_infos.values())
            comparison = step_results.get("compare_papers", {})
            differences = comparison.get("differences", "") if comparison.get("status") == "ok" else ""
            step_results["identify_research_gaps"] = identify_research_gaps(infos, differences)

        elif step == "generate_report":
            infos = list(paper_infos.values())
            report_text = generate_report(
                infos,
                comparison_result=step_results.get("compare_papers"),
                gap_result=step_results.get("identify_research_gaps"),
            )
            step_results["generate_report"] = report_text

        # Unknown steps are silently ignored here as a safety net -
        # create_plan() already filters to AVAILABLE_STEPS, but this
        # keeps the orchestrator robust even if that ever changes.

    return {
        "plan": plan,
        "effective_steps": ordered_steps,
        "step_results": step_results,
        "paper_infos": paper_infos,
    }
