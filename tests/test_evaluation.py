"""Evaluation metric and benchmark-file tests."""

import json
import os

from evaluation.metrics import (fact_recall, facts_found, is_refusal, mentions_number,
                                retrieval_metrics, answer_metrics)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_benchmark_file_is_well_formed():
    with open(os.path.join(ROOT, "evaluation", "benchmark.json")) as f:
        bench = json.load(f)
    ids = [i["id"] for i in bench["items"]]
    assert len(ids) == len(set(ids)) >= 15
    for item in bench["items"]:
        assert item["question"].endswith("?")
        assert item["expected"] or item.get("expect_refusal")
        assert all(isinstance(group, list) and group for group in item["expected"])


def test_fact_matching_uses_aliases_and_word_boundaries():
    expected = [["Natural Questions", "NQ"], ["TriviaQA"]]
    assert facts_found("Evaluated on NQ and TriviaQA [S1].", expected) == [True, True]
    assert fact_recall("Uses TriviaQA only", expected) == 0.5
    assert facts_found("UNQUALIFIED results", [["NQ"]]) == [False]      # no substring matches


def test_refusal_detection_and_invented_numbers():
    assert is_refusal("The uploaded papers do not report the training cost.")
    assert is_refusal("This question doesn't appear to be about the uploaded research papers")
    assert not is_refusal("Self-RAG uses Contriever [S2].")
    assert mentions_number("Training cost $12,000 on A100s")
    assert not mentions_number("The paper does not report a cost.")


def test_retrieval_metrics_rank_only_chunks_from_the_right_paper():
    item = {"paper": "a.pdf", "expected": [["BART"]]}
    chunks = [{"source_name": "b.pdf", "text": "BART is used"},       # wrong paper: ignored
              {"source_name": "a.pdf", "text": "unrelated"},
              {"source_name": "a.pdf", "text": "We use BART-large"}]
    m = retrieval_metrics(chunks, item)
    assert m["hit"] and m["reciprocal_rank"] == 1 / 3 and m["context_recall"] == 1.0


def test_answer_metrics_for_answerable_and_unanswerable_items():
    good = {"final_answer": "It uses BART [S1].", "usage": {"calls": 3, "prompt_tokens": 10, "completion_tokens": 5},
            "step_results": {"verify_answer": {"verdict": "supported", "citation_check": {"ok": True}}}}
    m = answer_metrics(good, {"expected": [["BART"]]})
    assert m["exact"] and m["faithful"] and m["citation_valid"] and m["tokens"] == 15

    invented = {"final_answer": "Training cost about $50,000.", "step_results": {}}
    honest = {"final_answer": "The paper does not report the training cost.", "step_results": {}}
    item = {"expected": [], "expect_refusal": True}
    assert not answer_metrics(invented, item)["refusal_correct"]
    assert answer_metrics(honest, item)["refusal_correct"]


def test_unicode_hyphens_and_spaces_from_llms_still_match():
    """Regression (found in the live benchmark): 'RAG\u2011Token', 'Arc\u2011Challenge', '150\u202fk'."""
    assert facts_found("variants: RAG\u2011Token and RAG\u2011Sequence", [["RAG-Token"], ["RAG-Sequence"]]) == [True, True]
    assert facts_found("trained on 150\u202fk pairs", [["150k", "150 k"]]) == [True]
    assert facts_found("Arc\u2011Challenge", [["Arc-Challenge"]]) == [True]
