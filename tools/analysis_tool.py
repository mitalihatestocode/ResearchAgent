"""
Paper Analysis Tool
---------------------
Converts one paper into the structured PaperInfo shape (research problem,
dataset, methodology, etc.) by:
  1. Running several targeted RAG retrievals against THIS paper only
     (one per topic - e.g. "dataset used", "evaluation metrics") so we
     gather focused evidence for each field instead of one vague search.
  2. Sending all that retrieved evidence to the LLM in a single call to
     produce the structured JSON.

This is the tool where RAG and the LLM genuinely work together: the LLM
never sees the raw paper, only what retrieval decided was relevant -
this is what keeps answers grounded and lets us cite page numbers.
"""

from rag.retriever import retrieve_relevant_chunks
from agent.llm_client import call_llm_json, LLMError
from agent.prompts import paper_analysis_prompt

# One retrieval query per structured field we need. Multiple targeted
# queries retrieve much more relevant evidence than a single generic
# "summarize this paper" query would.
TOPIC_QUERIES = [
    "research problem being addressed",
    "objectives and goals of the study",
    "dataset used in the study",
    "methodology and approach used",
    "machine learning models or algorithms used",
    "evaluation metrics used to measure performance",
    "results and findings",
    "limitations mentioned by the authors",
]


def analyze_paper(source_name: str, vector_store, chunks_per_topic: int = 2) -> dict:
    """
    Args:
        source_name: which uploaded paper to analyze (used to filter retrieval)
        vector_store: the built VectorStore containing all papers' chunks
        chunks_per_topic: how many chunks to retrieve per topic query

    Returns:
        {
            "status": "ok" | "error",
            "message": str,
            "source_name": str,
            "extracted_fields": dict,   # raw LLM output matching PaperInfo fields
            "evidence": [ {topic, chunks: [...]} , ... ]  # for showing sources in UI
        }
    """
    evidence = []
    context_blocks = []

    for topic in TOPIC_QUERIES:
        retrieval = retrieve_relevant_chunks(
            vector_store, topic, top_k=chunks_per_topic, source_filter=[source_name]
        )

        if retrieval["status"] == "ok":
            evidence.append({"topic": topic, "chunks": retrieval["chunks"]})
            for chunk in retrieval["chunks"]:
                context_blocks.append(
                    f"[Topic: {topic} | Page {chunk['page_number']}]\n{chunk['text']}"
                )
        else:
            evidence.append({"topic": topic, "chunks": []})

    if not context_blocks:
        return {
            "status": "error",
            "message": f"No retrievable content found for {source_name}. "
                       f"The paper may not have been indexed correctly.",
            "source_name": source_name,
            "extracted_fields": {},
            "evidence": evidence,
        }

    combined_context = "\n\n".join(context_blocks)

    try:
        system, user = paper_analysis_prompt(source_name, combined_context)
        extracted_fields = call_llm_json(system, user)

        return {
            "status": "ok",
            "message": f"Analyzed {source_name} using {len(context_blocks)} retrieved excerpt(s).",
            "source_name": source_name,
            "extracted_fields": extracted_fields,
            "evidence": evidence,
        }

    except LLMError as e:
        return {
            "status": "error",
            "message": str(e),
            "source_name": source_name,
            "extracted_fields": {},
            "evidence": evidence,
        }
