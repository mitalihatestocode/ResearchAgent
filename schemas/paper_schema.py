"""
Paper Schema
------------
Defines the structured shape every paper gets converted into after
analysis. Using a Pydantic model (instead of loose dicts) gives us:

1. A single source of truth for what fields a "paper" has - matches
   the FA rubric's comparison parameters exactly.
2. Automatic validation - if the LLM's output is missing a field or
   has the wrong type, we catch it immediately instead of crashing
   later during comparison.
3. Easy conversion to/from JSON for LLM prompts and Streamlit display.

NOT_FOUND_TEXT is the standard string used whenever information isn't
available in the paper - this is a safety requirement from the rubric
(never guess/fabricate; say clearly when something wasn't found).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional

NOT_FOUND_TEXT = "Not explicitly reported in the provided paper."


class PaperInfo(BaseModel):
    source_name: str  # which uploaded file this came from - always required

    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None

    research_problem: str = NOT_FOUND_TEXT
    objectives: str = NOT_FOUND_TEXT
    dataset: str = NOT_FOUND_TEXT
    methodology: str = NOT_FOUND_TEXT
    models: str = NOT_FOUND_TEXT
    evaluation_metrics: str = NOT_FOUND_TEXT
    results: str = NOT_FOUND_TEXT
    limitations: str = NOT_FOUND_TEXT

    def to_display_dict(self) -> dict:
        """Flat dict for showing in a Streamlit comparison table."""
        return {
            "Source File": self.source_name,
            "Title": self.title or NOT_FOUND_TEXT,
            "Authors": self.authors or NOT_FOUND_TEXT,
            "Year": self.year or NOT_FOUND_TEXT,
            "Research Problem": self.research_problem,
            "Objectives": self.objectives,
            "Dataset": self.dataset,
            "Methodology": self.methodology,
            "Models": self.models,
            "Evaluation Metrics": self.evaluation_metrics,
            "Results": self.results,
            "Limitations": self.limitations,
        }



    @field_validator(
        "research_problem",
        "objectives",
        "dataset",
        "methodology",
        "models",
        "evaluation_metrics",
        "results",
        "limitations",
        mode="before",
    )
    @classmethod
    def convert_list_to_string(cls, value):
        if isinstance(value, list):
            return "; ".join(str(item) for item in value)
        return value

    @classmethod
    def from_tool_outputs(cls, source_name: str, metadata_result: dict, analysis_result: dict) -> "PaperInfo":
        """
        Build a PaperInfo safely from the raw dicts returned by
        metadata_tool.extract_metadata() and analysis_tool.analyze_paper().

        Deliberately defensive: either tool may have failed (status
        "error") or the LLM may have omitted a key. We fall back to
        Pydantic's own defaults (NOT_FOUND_TEXT / None) rather than
        letting a missing key crash the UI.
        """
        fields = analysis_result.get("extracted_fields", {}) or {}

        kwargs = {"source_name": source_name}

        if metadata_result.get("status") == "ok":
            kwargs["title"] = metadata_result.get("title")
            kwargs["authors"] = metadata_result.get("authors")
            kwargs["year"] = metadata_result.get("year")

        for field_name in [
            "research_problem", "objectives", "dataset", "methodology",
            "models", "evaluation_metrics", "results", "limitations",
        ]:
            value = fields.get(field_name)
            if value:  # only override the default if the LLM actually gave something
                kwargs[field_name] = value

        return cls(**kwargs)
