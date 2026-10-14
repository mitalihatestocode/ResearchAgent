"""
Report Generator
------------------
Assembles the final structured report from everything the earlier tools
already produced (metadata, per-paper analysis, comparison, research
gaps, prior-work check, verified answer). Deliberately NOT another LLM
call - by this point every piece of content already exists and is
already labeled/grounded; re-generating it through the LLM would just
add another chance to drift from the source data for no benefit.

This is also where the labeling requirement is enforced in one place:
every section is explicitly tagged as either
  - "Reported by authors" (facts extracted from the papers)
  - "AI-generated analysis" (comparison synthesis, answers)
  - "Potential research direction" (gap suggestions - never "discovery")
  - "External source" (literature search results)
"""

from datetime import datetime

from schemas.paper_schema import PaperInfo, ANALYSIS_FIELDS, FIELD_LABELS
from safety.output_security import md_link


def _cell(text) -> str:
    """Make a value safe for a Markdown table cell."""
    return str(text or "").replace("|", "/").replace("\n", " ").strip()


def generate_report(
    paper_infos: list[PaperInfo],
    comparison_result: dict = None,
    gap_result: dict = None,
    answer_result: dict = None,
    verification: dict = None,
    related_work: dict = None,
    plan: dict = None,
    query: str = "",
) -> str:
    """
    Returns:
        A single Markdown string - the final report, ready to display in
        Streamlit (st.markdown) or offer as a download.
    """
    lines = ["# ResearchAgent — Analysis Report", ""]
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
                 f"{len(paper_infos)} paper(s) analysed_")
    if query:
        lines.append(f"\n**User request:** {query}")
    lines.append("")

    # ---- Answer to the user's question ----
    if answer_result and answer_result.get("status") == "ok":
        lines.append("## Answer")
        verdict = (verification or {}).get("verdict")
        tag = "AI-generated analysis, grounded in cited passages"
        if verdict and verdict not in ("not_applicable", "disabled"):
            tag += f"; self-verification verdict: {verdict.replace('_', ' ')}"
        lines.append(f"_{tag}_\n")
        lines.append(answer_result["answer"] + "\n")
        if answer_result.get("sources"):
            lines.append("**Sources**")
            for sid, src in answer_result["sources"].items():
                lines.append(f"- [{sid}] {src['source_name']}, page {src['page_number']} ({src.get('section') or 'Unknown'})")
            lines.append("")

    # ---- Section 1: Paper Overview ----
    lines.append("## 1. Paper Overview")
    lines.append("_Source: Reported by authors (extracted from each paper)_\n")
    lines.append("| File | Title | Authors | Year | Fields found |")
    lines.append("|---|---|---|---|---|")
    for info in paper_infos:
        lines.append(f"| {_cell(info.source_name)} | {_cell(info.title or 'Not found')} | "
                     f"{_cell(info.authors or 'Not found')} | {_cell(info.year or 'Not found')} | "
                     f"{round(info.completeness() * 100)}% |")
    lines.append("")

    # ---- Section 2: Individual Paper Analysis ----
    lines.append("## 2. Individual Paper Analysis")
    lines.append("_Source: Reported by authors (extracted via retrieval-grounded analysis; page references in brackets)_\n")
    for info in paper_infos:
        lines.append(f"### {info.display_name}")
        lines.append(f"_File: {info.source_name}_\n")
        for field in ANALYSIS_FIELDS:
            refs = info.field_sources.get(field)
            ref_text = f" _[{', '.join(refs)}]_" if refs else ""
            lines.append(f"- **{FIELD_LABELS[field]}:** {getattr(info, field)}{ref_text}")
        lines.append("")

    # ---- Section 3: Comparison ----
    if comparison_result and comparison_result.get("table"):
        lines.append("## 3. Comparison Across Papers")
        lines.append("_Table: Reported by authors (pivoted from individual analysis, not re-generated)_\n")

        table = comparison_result["table"]
        lines.append("| Parameter | " + " | ".join(_cell(p) for p in table["papers"]) + " |")
        lines.append("|---" * (len(table["papers"]) + 1) + "|")
        for row in table["rows"]:
            values = " | ".join(_cell(row["values"][p]) for p in table["papers"])
            lines.append(f"| {row['parameter']} | {values} |")
        lines.append("")

        if comparison_result.get("status") == "ok":
            lines.append("**Similarities** _(AI-generated analysis)_\n")
            lines.append(comparison_result.get("similarities", "") + "\n")
            lines.append("**Differences** _(AI-generated analysis)_\n")
            lines.append(comparison_result.get("differences", "") + "\n")
            if comparison_result.get("key_takeaway"):
                lines.append("**Key takeaway** _(AI-generated analysis)_\n")
                lines.append(comparison_result["key_takeaway"] + "\n")

    # ---- Section 4: Research Gaps ----
    if gap_result and gap_result.get("status") == "ok":
        lines.append("## 4. Potential Research Gaps")
        lines.append(
            "_These are AI-generated suggestions based on author-reported "
            "limitations and paper differences. They are potential "
            "directions worth further investigation, NOT confirmed or "
            "verified novel contributions._\n"
        )

        lines.append("**Author-reported limitations these are based on:**\n")
        for entry in gap_result.get("author_reported_limitations", []):
            lines.append(f"- [{entry['source_name']}] {entry['limitation']}")
        lines.append("")

        lines.append("**Potential research directions:**\n")
        for i, gap in enumerate(gap_result.get("research_gaps", []), start=1):
            lines.append(f"{i}. **{gap['direction']}**")
            lines.append(f"   - _Based on:_ {gap.get('based_on', 'N/A')}")
            prior = gap.get("prior_work")
            if prior:
                if prior.get("results"):
                    lines.append(f"   - _Prior-work check ({prior.get('provider')}, external source):_ "
                                 f"possibly related existing work —")
                    for r in prior["results"]:
                        lines.append(f"     - {md_link(r['title'], r['url'])} ({r['year']})")
                else:
                    lines.append(f"   - _Prior-work check:_ {prior.get('message', 'no results')}")
        lines.append("")
        if gap_result.get("guardrail_changes"):
            lines.append(f"_Safety note: {len(gap_result['guardrail_changes'])} overclaiming phrase(s) "
                         f"were automatically softened by the output guardrail._\n")

    # ---- Section 5: Related work (when searched without gaps) ----
    if related_work and related_work.get("mode") == "related_work":
        lines.append("## 5. Related Publications (external search)")
        lines.append("_Source: arXiv / OpenAlex APIs. Listed for further reading; not analysed by the agent._\n")
        for search in related_work.get("searches", []):
            lines.append(f"**Search:** {search['query']}")
            for r in search.get("results", []):
                lines.append(f"- {md_link(r['title'], r['url'])} — {r['authors']} ({r['year']}, {r['source']})")
            if not search.get("results"):
                lines.append(f"- {search.get('message')}")
            lines.append("")

    # ---- Section 6: Methodology / agent trace ----
    lines.append("## Methodology & Evidence Note")
    if plan:
        lines.append(f"- **Agent plan ({plan.get('status')}):** {' → '.join(plan.get('steps', []))}")
        lines.append(f"- **Planner reasoning:** {plan.get('reasoning', '')}")
    lines.append(
        "- All factual fields were extracted using retrieval-grounded analysis (hybrid BM25 + "
        "embedding search) over the uploaded papers. The LLM only saw retrieved excerpts, not "
        "full documents, and was instructed to state clearly when information was not found "
        "rather than guess."
    )
    lines.append("- Fields marked \"Not explicitly reported in the provided paper.\" were not found in the retrieved evidence.")
    lines.append("- This report is decision support for a human researcher; verify key claims against the original papers.")

    return "\n".join(lines)
