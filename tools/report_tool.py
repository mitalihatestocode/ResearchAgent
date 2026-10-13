"""
Report Generator
------------------
Assembles the final structured report from everything the earlier tools
already produced (metadata, per-paper analysis, comparison, research
gaps). Deliberately NOT another LLM call - by this point every piece of
content already exists and is already labeled/grounded; re-generating
it through the LLM would just add another chance to drift from the
source data for no benefit.

This is also where the rubric's labeling requirement is enforced in one
place: every section is explicitly tagged as either
  - "Reported by authors" (facts extracted from the papers)
  - "AI-generated analysis" (comparison synthesis)
  - "Potential research direction" (gap suggestions - never "discovery")
"""

from schemas.paper_schema import PaperInfo


def generate_report(
    paper_infos: list[PaperInfo],
    comparison_result: dict = None,
    gap_result: dict = None,
) -> str:
    """
    Args:
        paper_infos: list of analyzed PaperInfo objects
        comparison_result: output of comparison_tool.compare_papers(), or None
        gap_result: output of research_gap_tool.identify_research_gaps(), or None

    Returns:
        A single Markdown string - the final report, ready to display in
        Streamlit (st.markdown) or offer as a download.
    """
    lines = ["# ResearchAgent — Analysis Report\n"]

    # ---- Section 1: Paper Overview ----
    lines.append("## Paper Overview")
    lines.append("_Source: Reported by authors (extracted from each paper)_\n")
    for info in paper_infos:
        lines.append(f"**{info.source_name}**")
        lines.append(f"- Title: {info.title}")
        lines.append(f"- Authors: {info.authors}")
        lines.append(f"- Year: {info.year}\n")

    # ---- Section 2: Individual Paper Analysis ----
    lines.append("## Individual Paper Analysis")
    lines.append("_Source: Reported by authors (extracted via retrieval-grounded analysis)_\n")
    for info in paper_infos:
        lines.append(f"### {info.source_name}")
        d = info.to_display_dict()
        for field in ["Research Problem", "Objectives", "Dataset", "Methodology",
                      "Models", "Evaluation Metrics", "Results", "Limitations"]:
            lines.append(f"- **{field}:** {d[field]}")
        lines.append("")

    # ---- Section 3: Comparison ----
    if comparison_result and comparison_result.get("status") == "ok":
        lines.append("## Comparison Across Papers")
        lines.append("_Table: Reported by authors (pivoted from individual analysis, not re-generated)_\n")

        table = comparison_result["table"]
        header = "| Parameter | " + " | ".join(table["papers"]) + " |"
        separator = "|---" * (len(table["papers"]) + 1) + "|"
        lines.append(header)
        lines.append(separator)
        for row in table["rows"]:
            values = " | ".join(row["values"][p] for p in table["papers"])
            lines.append(f"| {row['parameter']} | {values} |")
        lines.append("")

        lines.append("**Similarities** _(AI-generated analysis)_")
        lines.append(comparison_result.get("similarities", "") + "\n")
        lines.append("**Differences** _(AI-generated analysis)_")
        lines.append(comparison_result.get("differences", "") + "\n")

    # ---- Section 4: Research Gaps ----
    if gap_result and gap_result.get("status") == "ok":
        lines.append("## Potential Research Gaps")
        lines.append(
            "_These are AI-generated suggestions based on author-reported "
            "limitations and paper differences. They are potential "
            "directions worth further investigation, NOT confirmed or "
            "verified novel contributions._\n"
        )

        lines.append("**Author-reported limitations these are based on:**")
        for entry in gap_result.get("author_reported_limitations", []):
            lines.append(f"- [{entry['source_name']}] {entry['limitation']}")
        lines.append("")

        lines.append("**Potential research directions:**")
        for i, gap in enumerate(gap_result.get("research_gaps", []), start=1):
            lines.append(f"{i}. {gap['direction']}")
            lines.append(f"   - _Based on: {gap.get('based_on', 'N/A')}_")
        lines.append("")

    # ---- Section 5: Evidence note ----
    lines.append("## Evidence Note")
    lines.append(
        "All factual fields above were extracted using retrieval-grounded "
        "analysis (RAG) against the uploaded papers - the LLM only saw "
        "retrieved excerpts, not full documents, and was instructed to "
        "state clearly when information was not found rather than guess."
    )

    return "\n".join(lines)
