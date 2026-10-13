# Project Notes — Problem Definition & Agent Suitability

## Why is this problem suited to an agentic approach (not a simple chatbot)?

A single-prompt chatbot (User -> Prompt -> LLM -> Answer) fails at this
task for concrete reasons:

1. **Multiple independent sub-tasks.** Extracting structured data from
   each paper, comparing across papers, and identifying gaps are
   different kinds of work needing different prompts/context. Doing
   this in one prompt means either truncating papers to fit context or
   losing structure in the output.

2. **Variable workflow per question.** "What datasets were used?" needs
   far less work than "generate a full literature review with research
   gaps" — a single fixed prompt/pipeline either overworks simple
   questions or underworks complex ones. An agent that plans first can
   right-size the work.

3. **Grounding requirement.** To avoid hallucinated findings, answers
   must be traceable to retrieved paper content, not generated from the
   LLM's general knowledge. That needs a retrieval step BEFORE
   generation (RAG), which a plain prompt-answer loop doesn't have room
   for.

4. **Multi-document reasoning.** Comparison and gap-finding require
   holding structured facts from N papers simultaneously and reasoning
   across them — this is naturally a multi-step process (extract each
   paper -> structure each -> compare structured data -> analyze gaps),
   not a single generation step.

## Agentic AI characteristics present in this system
- **Planning:** LLM-generated step list per query (see `planner.py`)
- **Tool use:** 5+ modular tools, invoked conditionally
- **Retrieval/RAG:** FAISS-backed retrieval with source metadata
- **Multi-step execution:** orchestrator runs the plan step by step
- **Autonomous decision-making:** which tools run depends on the plan,
  which depends on the LLM's read of the user's specific question
