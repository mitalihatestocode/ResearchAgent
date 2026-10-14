"""
Tool Registry
-------------
The single list of tools the agent can use. Each entry describes:
  - name / description / when_to_use : shown to the PLANNER LLM, so it
    picks tools from their descriptions (this is the same pattern used by
    function-calling and LangChain-style agents)
  - requires : hard prerequisites. The orchestrator auto-inserts these
    if the planner forgets them (dependency resolution)
  - kind : "local" (pure Python), "llm" (calls the LLM), "external_api"

Adding a tool = adding one ToolSpec here + one handler in orchestrator.py.
The order of TOOLS is the canonical execution order (a valid topological
order of the dependency graph).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    when_to_use: str
    requires: tuple = ()
    kind: str = "llm"
    planner_visible: bool = True  # verify_answer is added by policy, not chosen


TOOLS = [
    ToolSpec(
        name="extract_metadata",
        description="Extract title, authors and year of each paper from its first page",
        when_to_use="the user needs paper identities, related-work search, or any full analysis",
    ),
    ToolSpec(
        name="analyze_papers",
        description="Structured RAG extraction per paper: research problem, objectives, dataset, methodology, models, metrics, results, limitations (with page sources)",
        when_to_use="summaries of whole papers, comparison, research gaps, or reports",
    ),
    ToolSpec(
        name="rag_retrieval",
        description="Hybrid (semantic + keyword) search over the uploaded papers for passages relevant to the question; rewrites the query and retries if evidence is weak",
        when_to_use="any specific question that must be answered from the papers' text",
        kind="local",
    ),
    ToolSpec(
        name="answer_question",
        description="Write a concise answer to the user's question using only retrieved evidence, with [S#] page citations",
        when_to_use="the user asked a question that needs a direct written answer",
        requires=("rag_retrieval",),
    ),
    ToolSpec(
        name="verify_answer",
        description="Self-check: audits the drafted answer against the evidence and triggers one revision if claims are unsupported",
        when_to_use="always added automatically after answer_question",
        requires=("answer_question",),
        planner_visible=False,
    ),
    ToolSpec(
        name="compare_papers",
        description="Comparison table across papers (dataset, methods, models, metrics, results, limitations) plus similarities/differences synthesis",
        when_to_use="the user asks to compare or contrast two or more papers",
        requires=("extract_metadata", "analyze_papers"),
    ),
    ToolSpec(
        name="identify_research_gaps",
        description="Suggest potential research directions grounded in author-reported limitations and cross-paper differences",
        when_to_use="the user asks about research gaps, future work, open problems or new ideas",
        requires=("extract_metadata", "analyze_papers"),
    ),
    ToolSpec(
        name="search_related_work",
        description="Search external scholarly databases (arXiv, OpenAlex) for related publications; when research gaps exist, runs a prior-work check for each gap",
        when_to_use="the user wants related/recent literature, or after identify_research_gaps to check whether gaps are already explored",
        requires=("extract_metadata",),
        kind="external_api",
    ),
    ToolSpec(
        name="generate_report",
        description="Assemble everything produced so far into one structured, downloadable Markdown report",
        when_to_use="the user asks for a report, literature review or full analysis",
        requires=("extract_metadata", "analyze_papers"),
        kind="local",
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}
TOOL_NAMES = [t.name for t in TOOLS]


def planner_catalog() -> list[dict]:
    """Tool descriptions in the shape the planning prompt expects."""
    return [
        {"name": t.name, "description": t.description, "when_to_use": t.when_to_use}
        for t in TOOLS if t.planner_visible
    ]


def resolve_dependencies(requested: list[str]) -> tuple[list[str], list[str]]:
    """
    Expands a requested set of tools with their (transitive) prerequisites
    and returns them in canonical execution order.

    Returns:
        (ordered_steps, auto_added_steps)
    """
    needed = {name for name in requested if name in TOOLS_BY_NAME}
    frontier = list(needed)
    while frontier:
        for dep in TOOLS_BY_NAME[frontier.pop()].requires:
            if dep not in needed:
                needed.add(dep)
                frontier.append(dep)

    ordered = [name for name in TOOL_NAMES if name in needed]
    auto_added = [name for name in ordered if name not in requested]
    return ordered, auto_added
