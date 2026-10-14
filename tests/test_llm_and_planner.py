"""LLM client reliability and planner tests (TC09, TC11, TC17)."""

import pytest

from agent import llm_client
from agent.llm_client import call_llm_json, parse_json_response, LLMError
from agent.planner import create_plan, heuristic_plan
from agent.tool_registry import resolve_dependencies

PAPERS = ["paper_a.pdf", "paper_b.pdf"]


def test_parse_json_handles_fences_and_surrounding_prose():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure! Here it is: {"a": 2} Hope this helps.') == {"a": 2}


def test_tc11_missing_api_key_gives_clear_error(monkeypatch):
    """TC11: no GROQ_API_KEY -> a clear, actionable error instead of a crash."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
        llm_client.call_llm("system", "user")


def test_transient_errors_are_retried_then_succeed(fake_llm):
    class RateLimited(Exception):
        status_code = 429

    attempts = {"n": 0}

    def flaky(user):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimited("429 too many requests")
        return {"ok": True}

    fake_llm.overrides["metadata"] = flaky
    result = call_llm_json("You extract bibliographic metadata", "x")
    assert result == {"ok": True}
    assert attempts["n"] == 3


def test_invalid_json_raises_llm_error(fake_llm):
    fake_llm.overrides["metadata"] = "this is not json at all"
    with pytest.raises(LLMError, match="valid JSON"):
        call_llm_json("You extract bibliographic metadata", "x")


def test_tc09_plans_differ_by_question_type(fake_llm):
    """TC09: a narrow question gets a lean plan; a gap question gets the full pipeline."""
    narrow = create_plan("What dataset did paper 1 use?", PAPERS)
    broad = create_plan("Compare these papers and identify research gaps", PAPERS)

    assert narrow["steps"] == ["rag_retrieval", "answer_question"]
    assert narrow["target_papers"] == ["paper_a.pdf"]
    assert "identify_research_gaps" in broad["steps"]
    assert "compare_papers" not in narrow["steps"]
    assert fake_llm.calls[0]["temperature"] == 0.0  # planning is deterministic


def test_planner_drops_hallucinated_tools_and_papers(fake_llm):
    fake_llm.overrides["planner"] = {
        "reasoning": "x", "in_scope": True, "intent": "factual_question",
        "target_papers": ["imaginary.pdf", "PAPER_B.PDF"],
        "steps": [{"tool": "delete_files"}, {"tool": "rag_retrieval"}, "answer_question", {"tool": "verify_answer"}],
    }
    plan = create_plan("What did paper b find?", PAPERS)
    assert plan["steps"] == ["rag_retrieval", "answer_question"]  # unknown + policy-only tools removed
    assert plan["target_papers"] == ["paper_b.pdf"]


def test_tc17_planner_failure_falls_back_to_rule_based_plan(fake_llm):
    """TC17: when the planner LLM fails, a keyword-based plan keeps the agent working."""
    fake_llm.overrides["planner"] = RuntimeError("model unavailable")
    plan = create_plan("Compare the methodologies of these papers", PAPERS)
    assert plan["status"] == "fallback"
    assert "compare_papers" in plan["steps"]


def test_heuristic_plan_intents():
    assert heuristic_plan("find research gaps")["intent"] == "research_gaps"
    assert heuristic_plan("what is the F1 score?")["steps"] == ["rag_retrieval", "answer_question"]


def test_dependency_resolution_adds_prerequisites_in_order():
    ordered, auto_added = resolve_dependencies(["generate_report", "compare_papers"])
    assert ordered == ["extract_metadata", "analyze_papers", "compare_papers", "generate_report"]
    assert set(auto_added) == {"extract_metadata", "analyze_papers"}
