# ResearchAgent — Architecture

## Flow
```
User
  |
  v
Streamlit UI
  |
  v
Research Agent (orchestrator.py)
  |
  v
Planner (planner.py) --> produces an explicit JSON plan based on the
  |                       user's specific question (this is the
  |                       "autonomous decision-making" part: the agent
  |                       decides WHICH steps/tools are needed, rather
  |                       than always running the same fixed pipeline)
  v
Orchestrator enforces step dependencies, then executes in canonical order:
  |
  |-- extract_metadata        (title/authors/year per paper)
  |-- analyze_papers           (structured extraction per paper, via
  |                              multi-topic RAG retrieval + LLM)
  |-- rag_retrieval             (direct retrieval for narrow factual asks)
  |-- compare_papers            (deterministic table + AI narrative)
  |-- identify_research_gaps    (LLM suggestions grounded in reported limitations)
  |-- generate_report           (deterministic assembly of everything above)
  v
Streamlit UI (renders agent trace + results + downloadable report)
```

Only the steps the Planner actually selected (plus any auto-inserted
prerequisites) run for a given question - a narrow question like "what
dataset did paper 1 use" runs a much smaller subset than "compare these
papers and find research gaps."

## Why "Plan then Execute" instead of a free-roaming LangChain Agent/CrewAI?

1. **Explainability for viva.** We can point to the exact JSON plan the
   LLM produced for a specific question and walk through why each step
   was or wasn't included. A black-box autonomous agent loop is much
   harder to defend under questioning.
2. **Demo reliability.** A live demo in front of an examiner should not
   risk an agent looping unpredictably or calling the wrong tool
   repeatedly. An explicit plan is deterministic once generated.
3. **Still genuinely agentic.** The system still exhibits: planning
   (LLM decides the steps), tool selection (LLM decides which tools),
   and multi-step autonomous execution (no hardcoded if/else deciding
   the workflow — the LLM does).

## Why FAISS?
Local, no server/infra needed, well documented, free, and standard
enough that comparing it to alternatives (Chroma, Pinecone) is an easy
viva answer: "FAISS runs locally with no external dependency, which
matters for a self-contained student project."

## Why local embeddings (sentence-transformers) instead of an API?
No rate limits, no cost, works offline once downloaded, and keeps the
RAG pipeline independent of the LLM provider (so retrieval quality
isn't affected by which LLM API we use for generation).

## Structured paper representation
Each paper is converted into a Pydantic model (see `schemas/paper_schema.py`,
added in a later phase) with fields matching the FA rubric's comparison
requirements (dataset, methodology, models, metrics, results,
limitations, etc.). This structured form is what the Comparison Tool
and Research Gap Tool operate on — not raw text — which is why the
comparison table can be generated reliably rather than re-summarized
from scratch every time.
