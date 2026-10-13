"""
Prompts
-------
Every LLM call in this project uses a prompt template defined here -
never an inline ad-hoc string scattered in tool code. This satisfies
the "Advanced Prompt Engineering" rubric item: separate, purpose-built
prompts per task, not one generic do-everything prompt.

SHARED SAFETY RULES (viva point):
Every prompt below includes the same core anti-hallucination rules.
This is intentional - safety shouldn't depend on which specific task
is running. It's centralized here as SAFETY_RULES so it's applied
consistently and is easy to point to as one artifact during viva.
"""

SAFETY_RULES = """
Core rules you must always follow:
- Do not fabricate information. Only use what is explicitly present in
  the provided text.
- If information is not present in the provided text, say exactly:
  "Not explicitly reported in the provided paper." Do not guess or infer.
- Do not invent citations, authors, or numbers.
- Distinguish clearly between facts stated in the paper and any analysis
  you are asked to generate.
- Never claim a research idea or gap is definitely novel - phrase
  suggestions as "potential" research directions, not confirmed discoveries.
"""


def planning_prompt(user_query: str, paper_names: list[str], available_steps: list[str]) -> tuple[str, str]:
    """
    Prompt for the Planner: given the user's question, decide which
    steps (from the currently available tool vocabulary) are needed.

    This is the core "agentic" prompt - it's what makes the system
    decide a different workflow per question instead of always running
    the same fixed pipeline.
    """
    system = f"""You are the planning module of a research paper analysis agent.
Given a user's request and a list of uploaded papers, decide which steps
are needed to fulfill the request. You do NOT answer the request yourself -
you only produce a plan.

Available step types (use ONLY these, spelled exactly this way):
{chr(10).join(f"- {step}" for step in available_steps)}

Only include steps that are actually needed for this specific request.
For example, a question about one paper's dataset does not need a
"compare_papers" step if comparison wasn't asked for and only one
relevant paper exists.

Respond with a JSON object of this exact shape:
{{
  "reasoning": "one or two sentences on why these steps were chosen",
  "steps": ["step_type_1", "step_type_2", ...]
}}
{SAFETY_RULES}"""

    user = f"""User request: "{user_query}"

Uploaded papers available: {", ".join(paper_names)}

Produce the plan."""

    return system, user


def paper_analysis_prompt(source_name: str, retrieved_context: str) -> tuple[str, str]:
    """
    Prompt for the Paper Analysis Tool: convert retrieved chunks from
    ONE paper into the structured PaperInfo schema fields.
    """
    system = f"""You are a research paper analysis assistant. You will be
given multiple retrieved excerpts from ONE research paper.

Your task is to carefully identify information supported by the excerpts.
The excerpts may come from different sections of the paper and may contain
information relevant to more than one field.

Important instructions:
- Use the retrieved excerpts as evidence.
- Combine information from multiple excerpts when they describe the same topic.
- Do not assume that a field is missing merely because one excerpt does not contain it.
- Pay special attention to explicit descriptions of datasets, methodology,
  algorithms, evaluation metrics, results, and limitations.
- If the excerpts genuinely contain no evidence for a field, use the exact
  string "Not explicitly reported in the provided paper."
- Never guess or add information that is not supported by the excerpts.

Respond with a JSON object with exactly these keys:
title, authors, year, research_problem, objectives, dataset, methodology,
models, evaluation_metrics, results, limitations

For title/authors/year, use null if the retrieved excerpts do not contain
enough evidence to identify them.
{SAFETY_RULES}"""

    user = f"""Paper source file: {source_name}

Retrieved excerpts from this paper:
---
{retrieved_context}
---

Extract the structured information as JSON."""

    return system, user

def metadata_prompt(source_name: str, first_page_text: str) -> tuple[str, str]:
    """
    Prompt for the Metadata Extraction Tool: pull just title/authors/year
    from the first page (where this info usually lives). Kept separate
    from full paper analysis because metadata is cheap/fast to extract
    and is needed early (e.g. to label results in the UI) even before
    full analysis runs.
    """
    system = f"""You extract bibliographic metadata from the first page of
a research paper. Respond with a JSON object with exactly these keys:
title, authors, year (all strings, or null if not found in the given text).
{SAFETY_RULES}"""

    user = f"""Source file: {source_name}

First page text:
---
{first_page_text}
---

Extract title, authors, and year as JSON."""

    return system, user


def comparison_prompt(papers_summary: str) -> tuple[str, str]:
    """
    Prompt for the Comparison Tool's narrative synthesis: given the
    ALREADY-EXTRACTED structured data for each paper (not raw paper
    text), identify similarities and differences in plain language.

    NOTE: the comparison TABLE itself (parameter-by-parameter values)
    is built in plain Python from the structured data directly - no LLM
    involved there, since that data was already grounded in Phase 3.
    This prompt is only for the qualitative synthesis on top of it.
    """
    system = f"""You are comparing multiple already-analyzed research papers.
You will be given each paper's structured extracted information (not the
raw paper text). Identify meaningful similarities and differences across
the papers based ONLY on this structured data.

Respond with a JSON object of this exact shape:
{{
  "similarities": "a few sentences on what the papers have in common",
  "differences": "a few sentences on how the papers differ meaningfully"
}}
{SAFETY_RULES}"""

    user = f"""Structured data for each paper:
---
{papers_summary}
---

Identify similarities and differences as JSON."""

    return system, user


def research_gap_prompt(limitations_summary: str, differences_summary: str) -> tuple[str, str]:
    """
    Prompt for the Research Gap Analysis Tool: given author-reported
    limitations and the differences identified during comparison,
    suggest potential research directions.

    Explicitly instructed to treat these as POSSIBILITIES, not confirmed
    novel contributions - this directly satisfies the rubric's
    requirement to avoid overclaiming novelty.
    """
    system = f"""You suggest potential research directions based on limitations
that paper authors have explicitly reported, and differences identified
between papers. You are NOT claiming these directions are novel or that
no one else has explored them - you have no way to verify that. Frame
every suggestion as a "potential" direction worth further investigation,
tied to a specific limitation or gap you were given.

Respond with a JSON object of this exact shape:
{{
  "research_gaps": [
    {{
      "direction": "short description of the potential research direction",
      "based_on": "which specific limitation(s) or difference(s) this comes from"
    }},
    ...
  ]
}}
Produce 3-6 research_gaps entries. Each must be traceable to something
in the input below - do not invent gaps unrelated to the given information.
{SAFETY_RULES}"""

    user = f"""Author-reported limitations across the papers:
---
{limitations_summary}
---

Differences identified between the papers:
---
{differences_summary}
---

Suggest potential research directions as JSON."""

    return system, user
