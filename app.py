"""
ResearchAgent - Phase 3
------------------------
Adds: LLM integration (Groq), the Planner, and the Metadata/Analysis
tools. The agent can now take a real question, decide a plan, and
produce structured per-paper analysis grounded in retrieved evidence.

Comparison, limitation analysis, research gap identification, and the
final report are NOT wired in yet - Phase 4.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from a local .env file, if present

from rag.pdf_extractor import extract_text_from_pdf
from rag.chunker import chunk_paper
from rag.vector_store import VectorStore
from agent.orchestrator import run_agent
from agent.llm_client import LLMError

st.set_page_config(page_title="ResearchAgent", layout="wide")

# session_state holds extracted papers across Streamlit reruns.
# Streamlit reruns the whole script on every interaction, so anything
# we want to "remember" (like already-processed papers) must live here.
if "papers" not in st.session_state:
    st.session_state.papers = {}  # source_name -> extraction result dict

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # built once papers are processed

if "indexed_papers" not in st.session_state:
    st.session_state.indexed_papers = set()  # tracks which papers are already in the index

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload 2-5 research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Skip papers we've already processed this session
            if uploaded_file.name in st.session_state.papers:
                continue

            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                result = extract_text_from_pdf(uploaded_file, uploaded_file.name)
                st.session_state.papers[uploaded_file.name] = result

    if st.session_state.papers:
        st.divider()
        if st.button("Clear all papers"):
            st.session_state.papers = {}
            st.session_state.vector_store = None
            st.session_state.indexed_papers = set()
            st.rerun()

# ---------------- Main page ----------------
st.title("ResearchAgent")
st.caption("An Agentic AI System for Research Paper Analysis, Comparison and Research Gap Identification")

st.subheader("Uploaded Papers")

if not st.session_state.papers:
    st.info("Upload PDF research papers from the sidebar to get started.")
else:
    for name, result in st.session_state.papers.items():
        status = result["status"]

        if status == "ok":
            icon = "✅"
        elif status == "empty":
            icon = "⚠️"
        else:
            icon = "❌"

        with st.expander(f"{icon} {name} — {result['message']}"):
            st.write(f"**Status:** {status}")
            st.write(f"**Pages:** {result['num_pages']}")

            if status == "ok":
                preview = result["full_text"][:1000]
                st.text_area(
                    "Text preview (first 1000 characters)",
                    preview,
                    height=200,
                    key=f"preview_{name}",
                )

    ok_papers = [r for r in st.session_state.papers.values() if r["status"] == "ok"]
    st.divider()
    st.write(f"**{len(ok_papers)} of {len(st.session_state.papers)} paper(s) successfully processed.**")

    if len(ok_papers) < 2:
        st.warning(
            "You need at least 2 successfully processed papers to run a "
            "comparison later. Upload more, or check the papers marked "
            "⚠️/❌ above."
        )

    # ---------------- Phase 2: Build knowledge base (chunk + embed + index) ----------------
    st.divider()
    st.subheader("Knowledge Base (RAG)")

    newly_ok_papers = [
        (name, result) for name, result in st.session_state.papers.items()
        if result["status"] == "ok" and name not in st.session_state.indexed_papers
    ]

    if newly_ok_papers:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()

        with st.spinner(
            f"Building knowledge base: chunking and embedding "
            f"{len(newly_ok_papers)} paper(s)... (first run downloads the "
            f"embedding model, may take a minute)"
        ):
            for name, result in newly_ok_papers:
                chunks = chunk_paper(result)
                st.session_state.vector_store.add_chunks(chunks)
                st.session_state.indexed_papers.add(name)

    if st.session_state.vector_store and not st.session_state.vector_store.is_empty():
        st.success(
            f"Knowledge base ready — {st.session_state.vector_store.index.ntotal} "
            f"chunk(s) indexed from {len(st.session_state.indexed_papers)} paper(s)."
        )

        # ---------------- Phase 3: Research Agent (plan + execute) ----------------
        st.divider()
        st.subheader("Ask the Research Agent")

        import os
        if not os.environ.get("GROQ_API_KEY"):
            st.warning(
                "No GROQ_API_KEY found. Copy `.env.example` to `.env`, add your "
                "key from https://console.groq.com/keys, then restart the app "
                "to use the agent."
            )
        else:
            user_query = st.text_input(
                "e.g. 'Compare the datasets used in these papers' or "
                "'What methodology did each paper use?'"
            )

            if st.button("Run Agent") and user_query:
                ok_paper_dict = {
                    name: result for name, result in st.session_state.papers.items()
                    if result["status"] == "ok"
                }

                with st.spinner("Agent is planning and executing..."):
                    try:
                        agent_result = run_agent(
                            user_query, ok_paper_dict, st.session_state.vector_store
                        )
                    except LLMError as e:
                        st.error(f"Agent could not complete: {e}")
                        agent_result = None

                if agent_result:
                    plan = agent_result["plan"]
                    step_results = agent_result["step_results"]

                    # Show the plan itself - this is the visible proof of
                    # "planning" for the demo/viva: the examiner can see the
                    # agent decided a DIFFERENT set of steps depending on
                    # the question asked.
                    st.markdown("**Agent's Plan**")
                    if plan["status"] == "fallback":
                        st.caption(f"⚠️ {plan['reasoning']}")
                    else:
                        st.caption(plan["reasoning"])
                    st.code(" → ".join(plan["steps"]), language=None)

                    # ---- Metadata results ----
                    if "extract_metadata" in step_results:
                        st.markdown("**Metadata**")
                        for name, meta in step_results["extract_metadata"].items():
                            if meta["status"] == "ok":
                                st.write(
                                    f"📄 **{name}** — {meta['title'] or 'Not found'} "
                                    f"({meta['year'] or 'year unknown'}), "
                                    f"{meta['authors'] or 'authors unknown'}"
                                )
                            else:
                                st.write(f"📄 **{name}** — ⚠️ {meta['message']}")

                    # ---- Structured paper analysis ----
                    if "analyze_papers" in step_results:
                        st.markdown("**Paper Analysis**")
                        paper_infos = agent_result["paper_infos"]

                        for name, analysis in step_results["analyze_papers"].items():
                            if analysis["status"] != "ok":
                                st.error(f"{name}: {analysis['message']}")
                                continue

                            paper_info = paper_infos[name]

                            with st.expander(f"📊 {name} — structured analysis"):
                                for field_label, value in paper_info.to_display_dict().items():
                                    st.markdown(f"**{field_label}:** {value}")

                                st.caption(
                                    f"Grounded in {sum(len(e['chunks']) for e in analysis['evidence'])} "
                                    f"retrieved excerpt(s) across {len(analysis['evidence'])} topic queries."
                                )

                    # ---- Comparison across papers ----
                    if "compare_papers" in step_results:
                        comparison = step_results["compare_papers"]
                        st.markdown("**Comparison Across Papers**")

                        if comparison["status"] == "insufficient_papers":
                            st.info(comparison["message"])
                        else:
                            table = comparison["table"]
                            if table:
                                import pandas as pd
                                rows = {
                                    row["parameter"]: [row["values"][p] for p in table["papers"]]
                                    for row in table["rows"]
                                }
                                df = pd.DataFrame(rows, index=table["papers"]).T
                                st.dataframe(df, use_container_width=True)
                                st.caption("_Table: Reported by authors — pivoted directly from structured analysis, not re-generated._")

                            if comparison["status"] == "ok":
                                st.markdown("**Similarities** _(AI-generated analysis)_")
                                st.write(comparison["similarities"])
                                st.markdown("**Differences** _(AI-generated analysis)_")
                                st.write(comparison["differences"])
                            elif comparison["status"] == "error":
                                st.warning(f"Comparison table shown above, but narrative synthesis failed: {comparison['message']}")

                    # ---- Research gaps ----
                    if "identify_research_gaps" in step_results:
                        gaps = step_results["identify_research_gaps"]
                        st.markdown("**Potential Research Gaps**")

                        if gaps["status"] == "no_limitations_found":
                            st.info(gaps["message"])
                        elif gaps["status"] == "error":
                            st.warning(f"Could not generate research gaps: {gaps['message']}")
                        else:
                            st.caption(
                                "_These are AI-generated suggestions based on author-reported "
                                "limitations. They are potential directions worth further "
                                "investigation, NOT confirmed novel contributions._"
                            )
                            with st.expander("Author-reported limitations these are based on"):
                                for entry in gaps["author_reported_limitations"]:
                                    st.write(f"**[{entry['source_name']}]** {entry['limitation']}")

                            for i, gap in enumerate(gaps["research_gaps"], start=1):
                                st.markdown(f"**{i}. {gap['direction']}**")
                                st.caption(f"Based on: {gap.get('based_on', 'N/A')}")

                    # ---- Final report ----
                    if "generate_report" in step_results:
                        st.markdown("**Final Report**")
                        report_text = step_results["generate_report"]
                        with st.expander("📄 View full report", expanded=False):
                            st.markdown(report_text)
                        st.download_button(
                            "Download report as Markdown",
                            data=report_text,
                            file_name="research_agent_report.md",
                            mime="text/markdown",
                        )

                    # ---- Direct retrieval results (for narrow factual questions) ----
                    if "rag_retrieval" in step_results:
                        retrieval = step_results["rag_retrieval"]
                        st.markdown("**Retrieved Evidence**")
                        if retrieval["status"] == "empty":
                            st.info(retrieval["message"])
                        else:
                            for chunk in retrieval["chunks"]:
                                with st.expander(
                                    f"📄 {chunk['source_name']} — page {chunk['page_number']} "
                                    f"(similarity: {chunk['similarity_score']:.2f})"
                                ):
                                    st.write(chunk["text"])
