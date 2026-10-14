"""
Shared pytest fixtures.

The suite runs fully OFFLINE and deterministically:
  - FakeEmbedder replaces the sentence-transformers model with a hashed
    bag-of-words embedding (similar text -> similar vectors), so no model
    download is needed.
  - FakeLLM replaces the lowest-level Groq call (agent.llm_client._chat),
    so everything above it - retries, JSON parsing, planner validation,
    tools, orchestrator - runs for real. It routes on the system prompt
    to return a canned response per agent role.
  - External literature search is stubbed so tests never hit the network.
"""

import hashlib
import io
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.pdf_factory import make_pdf, PAPER_A_PAGES, PAPER_B_PAGES  # noqa: E402


# ---------------- fake embeddings ----------------
class FakeEmbedder:
    DIM = 384

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = np.zeros((len(texts), self.DIM), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) < 3:
                    continue
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
                vectors[row, bucket] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    import rag.embeddings
    import agent.orchestrator
    monkeypatch.setattr(rag.embeddings, "_model", FakeEmbedder())
    # The production threshold (0.28) is calibrated for all-MiniLM-L6-v2;
    # the hashed bag-of-words fake produces lower cosines for paraphrases.
    monkeypatch.setitem(agent.orchestrator.DEFAULT_SETTINGS, "min_relevance", 0.1)


@pytest.fixture(autouse=True)
def fake_prompt_guard(monkeypatch):
    """Deterministic stand-in for Llama Prompt Guard 2 (scores taken from real runs)."""
    import safety.prompt_guard

    def score(text):
        lowered = text.lower()
        attacks = ("pretend the safety policy", "ignore all previous", "developer message verbatim")
        return 0.999 if any(a in lowered for a in attacks) else 0.0004
    monkeypatch.setattr(safety.prompt_guard, "_score_with_groq", score)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import tools.literature_search_tool as lit

    def fake_search(query, max_results=3):
        return {
            "status": "ok", "message": "stubbed", "query": query, "provider": "arXiv",
            "results": [{"title": f"Existing work on {query}", "authors": "A. Author", "year": "2024",
                         "url": "http://arxiv.org/abs/0000.00000", "summary": "", "source": "arXiv"}],
        }
    monkeypatch.setattr(lit, "search_literature", fake_search)
    import agent.orchestrator
    monkeypatch.setattr(agent.orchestrator, "search_literature", fake_search)


# ---------------- fake LLM ----------------
class FakeLLM:
    """Routes each call by the role described in the system prompt."""

    def __init__(self):
        self.calls = []
        self.overrides = {}  # role -> response (dict, str, or Exception)

    def role_of(self, system: str) -> str:
        s = system.lower()
        if "planning module" in s:
            return "planner"
        if "bibliographic metadata" in s:
            return "metadata"
        if "meticulous research-paper analyst" in s:
            return "analysis"
        if "strict fact-checking reviewer" in s:
            return "verifier"
        if "careful research assistant answering" in s:
            return "answer"
        if "improve search queries" in s:
            return "rewrite"
        if "comparing multiple already-analyzed" in s:
            return "comparison"
        if "suggest potential research directions" in s:
            return "gaps"
        return "unknown"

    def __call__(self, model, system, user, temperature, json_mode):
        role = self.role_of(system)
        self.calls.append({"role": role, "system": system, "user": user, "temperature": temperature})
        response = self.overrides.get(role)
        if callable(response) and not isinstance(response, type):
            response = response(user)
        if isinstance(response, Exception):
            raise response
        if response is None:
            response = self.default(role, user)
        return response if isinstance(response, str) else json.dumps(response)

    def count(self, role):
        return sum(1 for c in self.calls if c["role"] == role)

    def default(self, role, user):
        if role == "planner":
            q = re.search(r'User request: "(.*)"', user).group(1).lower()
            if "gap" in q:
                steps = ["extract_metadata", "analyze_papers", "compare_papers",
                         "identify_research_gaps", "search_related_work", "generate_report"]
                return {"reasoning": "Gap analysis needs full structured analysis.", "in_scope": True,
                        "intent": "research_gaps", "target_papers": [],
                        "steps": [{"tool": s, "why": "needed"} for s in steps]}
            if "compare" in q:
                # Deliberately omit prerequisites to exercise dependency resolution
                return {"reasoning": "Comparison requested.", "in_scope": True, "intent": "comparison",
                        "target_papers": [], "steps": [{"tool": "compare_papers", "why": "compare"}]}
            if "weather" in q or "cricket" in q:
                return {"reasoning": "Unrelated to the papers.", "in_scope": False,
                        "intent": "out_of_scope", "target_papers": [], "steps": []}
            target = ["paper_a.pdf"] if "paper 1" in q else []
            return {"reasoning": "Narrow factual question.", "in_scope": True, "intent": "factual_question",
                    "target_papers": target,
                    "steps": [{"tool": "rag_retrieval", "why": "find evidence"},
                              {"tool": "answer_question", "why": "answer with citations"}]}
        if role == "metadata":
            if "paper_a" in user:
                return {"title": "Hybrid Retrieval for Clinical Question Answering",
                        "authors": "Asha Rao, Vikram Mehta", "year": "2023"}
            return {"title": "Dense Passage Retrieval for Legal Documents",
                    "authors": "Neha Kulkarni", "year": "Published in 2024"}
        if role == "analysis":
            if "paper_a" in user:
                return {"research_problem": "Clinical QA over long noisy notes",
                        "objectives": "Improve answer accuracy on medical questions",
                        "dataset": "MIMIC-III, 12,000 QA pairs",
                        "methodology": "Hybrid BM25 + dense retrieval with a generator",
                        "models": ["BioBERT", "BM25"],
                        "evaluation_metrics": "F1, exact match",
                        "results": "F1 78.4, EM 61.2",
                        "limitations": "English notes from a single hospital; latency not evaluated",
                        "evidence_ids": {"dataset": ["E1", "E99"], "limitations": ["E2"]}}
            return {"research_problem": "Finding precedent legal cases",
                    "objectives": "Not explicitly reported in the provided paper.",
                    "dataset": "COLIEE 2022, 4,400 cases",
                    "methodology": "Dual encoder fine-tuned with contrastive learning",
                    "models": "LegalBERT dual encoder",
                    "evaluation_metrics": "Recall@10",
                    "results": "Recall@10 0.83 vs 0.71 for BM25",
                    "limitations": "Only Canadian case law; large GPU memory",
                    "evidence_ids": {}}
        if role == "answer":
            return {"answer": "paper_a.pdf uses the MIMIC-III dataset with 12,000 QA pairs [S1].",
                    "answerable": "yes", "cited_ids": ["S1"]}
        if role == "verifier":
            return {"reasoning": "The claim matches S1.", "verdict": "supported",
                    "unsupported_claims": [], "confidence": 0.9}
        if role == "rewrite":
            return {"queries": ["dataset corpus used", "experimental data", "benchmark data"]}
        if role == "comparison":
            return {"similarities": "Both use retrieval.", "differences": "Different domains.",
                    "key_takeaway": "Hybrid retrieval suits noisy clinical text."}
        if role == "gaps":
            return {"research_gaps": [
                {"direction": "A novel, first-ever multilingual clinical QA benchmark",
                 "based_on": "paper_a.pdf: English notes only", "search_query": "multilingual clinical QA"},
                {"direction": "Cross-jurisdiction legal retrieval",
                 "based_on": "paper_b.pdf: only Canadian case law", "search_query": "cross-jurisdiction legal retrieval"},
                {"bad": "malformed entry without direction"},
            ]}
        return {}


@pytest.fixture
def fake_llm(monkeypatch):
    import agent.llm_client
    llm = FakeLLM()
    monkeypatch.setattr(agent.llm_client, "_chat", llm)
    monkeypatch.setattr(agent.llm_client, "BASE_BACKOFF_SECONDS", 0)
    return llm


# ---------------- papers / knowledge base ----------------
@pytest.fixture
def papers():
    from rag.pdf_extractor import extract_text_from_pdf
    return {
        "paper_a.pdf": extract_text_from_pdf(io.BytesIO(make_pdf(PAPER_A_PAGES)), "paper_a.pdf"),
        "paper_b.pdf": extract_text_from_pdf(io.BytesIO(make_pdf(PAPER_B_PAGES)), "paper_b.pdf"),
    }


@pytest.fixture
def store(papers):
    from rag.chunker import chunk_paper
    from rag.vector_store import VectorStore
    vs = VectorStore()
    for result in papers.values():
        vs.add_chunks(chunk_paper(result))
    return vs
