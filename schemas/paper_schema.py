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

ANALYSIS_FIELDS = [
    "research_problem", "objectives", "dataset", "methodology",
    "models", "evaluation_metrics", "results", "limitations",
]

FIELD_LABELS = {
    "research_problem": "Research Problem",
    "objectives": "Objectives",
    "dataset": "Dataset",
    "methodology": "Methodology",
    "models": "Models",
    "evaluation_metrics": "Evaluation Metrics",
    "results": "Results",
    "limitations": "Limitations",
}


def _to_text(value):
    """LLMs sometimes return lists/dicts/numbers for text fields - normalise them."""
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


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

    # field name -> ["p.3 (Results)", ...] - where each field's evidence came from
    field_sources: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("title", "authors", "year", mode="before")
    @classmethod
    def normalise_metadata(cls, value):
        return _to_text(value)

    @field_validator(*ANALYSIS_FIELDS, mode="before")
    @classmethod
    def convert_list_to_string(cls, value):
        text = _to_text(value)
        return text if text and text.strip() else NOT_FOUND_TEXT

    @property
    def display_name(self) -> str:
        return self.title or self.source_name

    def is_reported(self, field_name: str) -> bool:
        return "not explicitly reported" not in getattr(self, field_name).lower()

    def completeness(self) -> float:
        """Share of analysis fields that were actually found in the paper (0..1)."""
        return sum(self.is_reported(f) for f in ANALYSIS_FIELDS) / len(ANALYSIS_FIELDS)

    def to_display_dict(self) -> dict:
        """Flat dict for showing in a Streamlit comparison table."""
        data = {
            "Source File": self.source_name,
            "Title": self.title or NOT_FOUND_TEXT,
            "Authors": self.authors or NOT_FOUND_TEXT,
            "Year": self.year or NOT_FOUND_TEXT,
        }
        for field in ANALYSIS_FIELDS:
            data[FIELD_LABELS[field]] = getattr(self, field)
        return data

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

        # The analysis prompt also asks for title/authors/year as a fallback
        for key in ("title", "authors", "year"):
            if not kwargs.get(key) and fields.get(key):
                kwargs[key] = fields.get(key)

        for field_name in ANALYSIS_FIELDS:
            value = fields.get(field_name)
            if value:  # only override the default if the LLM actually gave something
                kwargs[field_name] = value

        kwargs["field_sources"] = analysis_result.get("field_sources", {}) or {}
        return cls(**kwargs)
