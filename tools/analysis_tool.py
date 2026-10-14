"""
Paper Analysis Tool
---------------------
Converts one paper into the structured PaperInfo shape (research problem,
dataset, methodology, etc.) by:
  1. Running several targeted hybrid RAG retrievals against THIS paper
     only (one per field - e.g. "dataset used", "evaluation metrics") so
     we gather focused evidence for each field instead of one vague search.
  2. De-duplicating the retrieved chunks and labelling them E1, E2, ...
  3. Sending that evidence to the LLM in a single call to produce the
     structured JSON, including which evidence ids support each field.
  4. Mapping evidence ids back to page numbers/sections, so every field
     in the UI can say "Source: p.4 (Results)".

This is the tool where RAG and the LLM genuinely work together: the LLM
never sees the raw paper, only what retrieval decided was relevant.
"""

from rag.retriever import retrieve_relevant_chunks
from agent.llm_client import call_llm_json, LLMError
from agent.prompts import paper_analysis_prompt, format_evidence, PAPER_FIELDS, TEMPERATURE
from safety.guardrails import sanitize_evidence

# One retrieval query per structured field. The keyword-rich phrasing
# helps the BM25 half of hybrid search hit the right section.
TOPIC_QUERIES = {
    "research_problem": "research problem motivation challenge addressed in this paper",
    "objectives": "objectives goals contributions of this study we propose",
    "dataset": "dataset corpus data collection benchmark samples used for experiments",
    "methodology": "methodology proposed approach framework pipeline architecture",
    "models": "machine learning models algorithms baselines architecture used",
    "evaluation_metrics": "evaluation metrics accuracy precision recall F1 score measure performance",
    "results": "results findings performance improvement outperforms table",
    "limitations": "limitations weaknesses future work drawbacks threats to validity",
}


# Token budget: Groq's free tier allows ~8k tokens/minute per model, so the
# evidence sent per paper is capped (MAX_EVIDENCE_CHUNKS x MAX_CHUNK_CHARS
# ~= 2.5k tokens) while still covering every field.
MAX_EVIDENCE_CHUNKS = 14
MAX_CHUNK_CHARS = 650


def analyze_paper(source_name: str, vector_store, chunks_per_topic: int = 2, mode: str = "hybrid") -> dict:
    """
    Args:
        source_name: which uploaded paper to analyze (used to filter retrieval)
        vector_store: the built VectorStore containing all papers' chunks
        chunks_per_topic: how many chunks to retrieve per topic query
        mode: retrieval mode ("hybrid" | "dense" | "keyword")

    Returns:
        {
            "status": "ok" | "error",
            "message": str,
            "source_name": str,
            "extracted_fields": dict,          # raw LLM output matching PaperInfo fields
            "field_sources": {field: ["p.3 (Results)", ...]},
            "evidence": [ {topic, chunks: [...]} , ... ]  # for showing sources in UI
        }
    """
    evidence = []
    unique_chunks = {}  # chunk_id -> chunk, preserves first-seen order

    for field, query in TOPIC_QUERIES.items():
        retrieval = retrieve_relevant_chunks(
            vector_store, query, top_k=chunks_per_topic, source_filter=[source_name], mode=mode
        )
        evidence.append({"topic": field, "chunks": retrieval["chunks"]})
        for chunk in retrieval["chunks"]:
            unique_chunks.setdefault(chunk["chunk_id"], chunk)

    if not unique_chunks:
        return {
            "status": "error",
            "message": f"No retrievable content found for {source_name}. "
                       f"The paper may not have been indexed correctly.",
            "source_name": source_name,
            "extracted_fields": {},
            "field_sources": {},
            "evidence": evidence,
        }

    # unique_chunks is in topic order, so truncating keeps every field's best evidence
    selected = list(unique_chunks.values())[:MAX_EVIDENCE_CHUNKS]
    ordered = sorted(selected, key=lambda c: (c["page_number"], c["chunk_id"]))
    safe_chunks = [dict(c, text=sanitize_evidence(c["text"])[:MAX_CHUNK_CHARS]) for c in ordered]
    evidence_text = format_evidence(safe_chunks, id_prefix="E")
    id_to_chunk = {f"E{i}": c for i, c in enumerate(ordered, start=1)}

    try:
        system, user = paper_analysis_prompt(source_name, evidence_text)
        extracted_fields = call_llm_json(system, user, temperature=TEMPERATURE["analysis"])
    except LLMError as e:
        return {
            "status": "error",
            "message": str(e),
            "source_name": source_name,
            "extracted_fields": {},
            "field_sources": {},
            "evidence": evidence,
        }

    # Map the evidence ids the LLM cited back to real pages. Ids that do
    # not exist (hallucinated citations) are silently dropped here.
    field_sources = {}
    evidence_ids = extracted_fields.get("evidence_ids") or {}
    if isinstance(evidence_ids, dict):
        for field in PAPER_FIELDS:
            refs = []
            for eid in evidence_ids.get(field, []) or []:
                chunk = id_to_chunk.get(str(eid).strip())
                if chunk:
                    ref = f"p.{chunk['page_number']} ({chunk.get('section', 'Unknown')})"
                    if ref not in refs:
                        refs.append(ref)
            if refs:
                field_sources[field] = refs

    return {
        "status": "ok",
        "message": f"Analyzed {source_name} using {len(ordered)} unique retrieved excerpt(s).",
        "source_name": source_name,
        "extracted_fields": extracted_fields,
        "field_sources": field_sources,
        "evidence": evidence,
    }
