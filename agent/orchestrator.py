"""
Research Agent Orchestrator
-----------------------------
The agent's "body": it takes a user query, runs it through the input
guardrails, asks the Planner which tools are needed, then executes those
tools step by step, keeping a full trace of what happened.

AGENT LOOP (viva summary)
    guardrails -> plan -> resolve dependencies -> execute each step
      -> observe results -> self-correct where needed -> final answer

AGENTIC BEHAVIOURS IMPLEMENTED HERE
1. Planning            : the LLM chooses the tools per question (planner.py)
2. Dependency resolution: missing prerequisites are auto-inserted
                          (tool_registry.resolve_dependencies) and shown
                          in the trace as "auto-added"
3. Scope self-check    : if the planner says "out of scope" but retrieval
                          finds strongly relevant passages, the agent
                          overrides the planner and answers anyway (and
                          vice versa: weak evidence -> honest "not found")
4. Adaptive retrieval  : if evidence is weak, the LLM rewrites the query
                          into alternatives and retrieval is retried
5. Reflection          : every drafted answer is audited by the verifier;
                          unsupported claims trigger one revision pass
6. Memory              : conversation history (follow-up questions) and a
                          per-session cache of paper analyses, so papers
                          are not re-analysed for every question
7. Graceful degradation: every tool returns a status instead of raising;
                          one failing step never crashes the whole run

Every step appends a trace event {step, status, detail, duration_s,
auto_added, why} which the UI renders as a live timeline.
"""

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

from agent import llm_client
from agent.llm_client import call_llm_json, LLMError
from agent.planner import create_plan
from agent.prompts import query_rewrite_prompt, TEMPERATURE
from agent.tool_registry import resolve_dependencies
from rag.retriever import retrieve_relevant_chunks, merge_retrievals, MIN_RELEVANCE
from safety import prompt_guard
from safety.guardrails import validate_query
from safety.output_security import md_link
from schemas.paper_schema import PaperInfo
from tools.metadata_tool import extract_metadata
from tools.analysis_tool import analyze_paper
from tools.answer_tool import answer_question
from tools.verifier_tool import verify_answer
from tools.comparison_tool import compare_papers
from tools.research_gap_tool import identify_research_gaps
from tools.literature_search_tool import search_literature
from tools.report_tool import generate_report

DEFAULT_SETTINGS = {
    "retrieval_mode": "hybrid",     # hybrid | dense | keyword
    "top_k": 6,
    "min_relevance": MIN_RELEVANCE,  # cosine threshold for using a passage as evidence
    "enable_verification": True,
    "enable_query_rewrite": True,
    "enable_external_search": True,
    "enable_prompt_guard": True,    # ML injection classifier behind the regex guardrails
    "max_parallel_llm_calls": 3,
}

# Below this best-similarity the agent tries rewriting the query
WEAK_EVIDENCE_THRESHOLD = 0.35
# If the planner says "out of scope" but retrieval is at least this strong,
# the agent overrides the planner (scope self-check)
SCOPE_OVERRIDE_THRESHOLD = 0.45

OUT_OF_SCOPE_ANSWER = (
    "This question doesn't appear to be about the uploaded research papers, so I "
    "haven't answered it. I'm built to analyse, compare and find research gaps in "
    "the papers you upload. For example, try *\"What datasets were used?\"* or "
    "*\"Compare these papers and identify potential research gaps.\"*"
)


class AgentRun:
    def __init__(self, user_query, papers, vector_store, history=None, cache=None,
                 settings=None, on_event=None):
        self.query = user_query
        self.papers = papers
        self.paper_names = list(papers.keys())
        self.store = vector_store
        self.history = history or []
        self.cache = cache if cache is not None else {}
        self.cache.setdefault("metadata", {})
        self.cache.setdefault("analysis", {})
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.on_event = on_event

        self.trace = []
        self.security_events = []  # [{event, severity, detail}] - merged into the UI audit log
        self.step_results = {}
        self.paper_infos = {}
        self.plan = None
        self.target_papers = []

    # ---------------- trace helpers ----------------
    def emit(self, step, status, detail, started=None, **extra):
        event = {
            "step": step,
            "status": status,  # ok | warning | error | skipped | blocked | info
            "detail": detail,
            "duration_s": round(time.time() - started, 2) if started else 0.0,
            **extra,
        }
        self.trace.append(event)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass  # UI callback problems must never break the agent
        return event

    def security(self, event, severity, detail):
        self.security_events.append({"event": event, "severity": severity, "detail": detail})

    def _count_flagged(self, chunks):
        return sum(1 for c in chunks if c.get("injection_flags"))

    def _parallel(self, fn, items):
        workers = max(1, min(self.settings["max_parallel_llm_calls"], len(items)))
        # Each worker runs inside a copy of the caller's context, so the
        # session-scoped API key (llm_client.use_api_key) reaches every thread.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(contextvars.copy_context().run, fn, item) for item in items]
            return [f.result() for f in futures]

    def _scope_papers(self):
        return self.target_papers or self.paper_names

    # ---------------- step handlers ----------------
    def step_extract_metadata(self):
        names = self._scope_papers()
        todo = [n for n in names if n not in self.cache["metadata"]]
        for name, result in zip(todo, self._parallel(lambda n: extract_metadata(self.papers[n]), todo)):
            if result["status"] == "ok":
                self.cache["metadata"][name] = result
            else:
                self.step_results.setdefault("_metadata_errors", {})[name] = result
        results = {n: self.cache["metadata"].get(n) or self.step_results.get("_metadata_errors", {}).get(n)
                   for n in names}
        self.step_results["extract_metadata"] = results
        ok = sum(1 for r in results.values() if r and r["status"] == "ok")
        cached = len(names) - len(todo)
        status = "ok" if ok == len(names) else ("warning" if ok else "error")
        return status, f"Metadata for {ok}/{len(names)} paper(s)" + (f" ({cached} from cache)" if cached else "")

    def step_analyze_papers(self):
        names = self._scope_papers()
        mode = self.settings["retrieval_mode"]
        todo = [n for n in names if n not in self.cache["analysis"]]
        for name, result in zip(todo, self._parallel(lambda n: analyze_paper(n, self.store, mode=mode), todo)):
            if result["status"] == "ok":
                self.cache["analysis"][name] = result
            else:
                self.step_results.setdefault("_analysis_errors", {})[name] = result
        results = {n: self.cache["analysis"].get(n) or self.step_results.get("_analysis_errors", {}).get(n)
                   for n in names}
        self.step_results["analyze_papers"] = results

        metadata = self.step_results.get("extract_metadata", {})
        for name, analysis in results.items():
            if analysis and analysis["status"] == "ok":
                meta = metadata.get(name) or self.cache["metadata"].get(name) or {"status": "error"}
                self.paper_infos[name] = PaperInfo.from_tool_outputs(name, meta, analysis)

        flagged = sum(self._count_flagged(e["chunks"]) for r in results.values() if r for e in r.get("evidence", []))
        if flagged:
            self.security("document_injection_neutralised", "medium",
                          f"{flagged} evidence passage(s) contained instruction-like text; neutralised before the LLM saw them.")

        ok = len(self.paper_infos)
        cached = len(names) - len(todo)
        status = "ok" if ok == len(names) else ("warning" if ok else "error")
        return status, f"Structured analysis for {ok}/{len(names)} paper(s)" + (f" ({cached} from cache)" if cached else "")

    def _retrieval_query(self):
        # Short follow-ups ("what about its limitations?") borrow context from the previous turn
        if self.history and len(self.query.split()) <= 8:
            return f"{self.history[-1]['query']} {self.query}"
        return self.query

    def _retrieve(self, query):
        mode, top_k = self.settings["retrieval_mode"], self.settings["top_k"]
        results = [retrieve_relevant_chunks(self.store, query, top_k=top_k, source_filter=self.target_papers or None,
                                            mode=mode, min_relevance=self.settings["min_relevance"])]
        # Multi-paper questions: guarantee every paper gets a chance to contribute evidence
        scope = self._scope_papers()
        if len(scope) > 1:
            for name in scope:
                results.append(retrieve_relevant_chunks(self.store, query, top_k=2, source_filter=[name],
                                                        mode=mode, min_relevance=self.settings["min_relevance"]))
        return results

    def step_rag_retrieval(self):
        query = self._retrieval_query()
        results = self._retrieve(query)
        chunks = merge_retrievals(results, top_k=self.settings["top_k"] + len(self._scope_papers()))
        best = max((r["max_similarity"] for r in results), default=0.0)
        rewrites = []

        if self.settings["enable_query_rewrite"] and (not chunks or best < WEAK_EVIDENCE_THRESHOLD):
            try:
                system, user = query_rewrite_prompt(self.query, self.paper_names)
                rewrites = [str(q) for q in call_llm_json(system, user, temperature=TEMPERATURE["rewrite"]).get("queries", [])][:3]
            except LLMError:
                rewrites = []
            if rewrites:
                for rq in rewrites:
                    results.extend(self._retrieve(rq))
                chunks = merge_retrievals(results, top_k=self.settings["top_k"] + len(self._scope_papers()))
                best = max((r["max_similarity"] for r in results), default=0.0)

        self.step_results["rag_retrieval"] = {
            "status": "ok" if chunks else "low_relevance",
            "message": (f"Retrieved {len(chunks)} passage(s), best similarity {best:.2f}." if chunks
                        else f"No passage passed the relevance threshold ({self.settings['min_relevance']}); best similarity {best:.2f}."),
            "query": query,
            "rewritten_queries": rewrites,
            "max_similarity": best,
            "chunks": chunks,
        }
        detail = self.step_results["rag_retrieval"]["message"]
        flagged = self._count_flagged(chunks)
        if flagged:
            self.security("document_injection_neutralised", "medium",
                          f"{flagged} retrieved passage(s) contained instruction-like text; neutralised before the LLM saw them.")
            detail += f" {flagged} passage(s) had instruction-like text neutralised."
        if rewrites:
            detail += f" Self-correction: weak evidence, so the query was rewritten into {len(rewrites)} alternatives."
        return ("ok" if chunks else "warning"), detail

    def step_answer_question(self, feedback=""):
        retrieval = self.step_results.get("rag_retrieval", {})
        infos = list(self.paper_infos.values())
        result = answer_question(self.query, retrieval.get("chunks", []), infos, self.history, feedback)
        self.step_results["answer_question"] = result
        detail = result["message"]

        # Answerability-driven self-correction (added after the evaluation benchmark
        # showed honest-but-incomplete answers when the first search missed a passage):
        # if the agent's own draft says the evidence was insufficient, rewrite the
        # search around what is missing, retrieve more, and answer again.
        if (not feedback and result["status"] == "ok" and result.get("answerable") in ("no", "partial")
                and self.settings["enable_query_rewrite"] and retrieval.get("chunks")):
            improved = self._answer_with_more_evidence(result)
            if improved:
                result = improved
                detail = (f"Self-correction: first draft was '{self.step_results['rag_retrieval']['first_answerable']}', "
                          f"so the agent searched again with {len(retrieval.get('second_pass_queries', []))} rewritten "
                          f"queries and re-answered ({result['answerable']}). " + result["message"])
        status = {"ok": "ok", "no_evidence": "warning"}.get(result["status"], "error")
        return status, detail

    def _answer_with_more_evidence(self, first: dict):
        retrieval = self.step_results["rag_retrieval"]
        try:
            system, user = query_rewrite_prompt(self.query, self._scope_papers(), first["answer"])
            queries = [str(q) for q in call_llm_json(system, user, temperature=TEMPERATURE["rewrite"]).get("queries", [])][:3]
        except LLMError:
            return None
        if not queries:
            return None
        results = [{"chunks": retrieval["chunks"]}]
        for q in queries:
            results.extend(self._retrieve(q))
        chunks = merge_retrievals(results, top_k=self.settings["top_k"] + 2 * len(self._scope_papers()) + 2)
        if {c["chunk_id"] for c in chunks} <= {c["chunk_id"] for c in retrieval["chunks"]}:
            return None  # nothing new found - keep the honest first answer
        second = answer_question(self.query, chunks, list(self.paper_infos.values()), self.history)
        rank = {"yes": 2, "partial": 1, "no": 0}
        if second["status"] != "ok" or rank.get(second.get("answerable"), 0) <= rank.get(first.get("answerable"), 0):
            return None
        retrieval.update({"chunks": chunks, "second_pass_queries": queries, "first_answerable": first["answerable"]})
        self.step_results["answer_question"] = second
        return second

    def step_verify_answer(self):
        answer = self.step_results.get("answer_question", {})
        if answer.get("status") != "ok":
            self.step_results["verify_answer"] = {"status": "skipped", "verdict": "not_applicable"}
            return "skipped", "Nothing to verify (no LLM-generated answer)."
        if not self.settings["enable_verification"]:
            self.step_results["verify_answer"] = {"status": "skipped", "verdict": "disabled"}
            return "skipped", "Verification disabled in settings."

        valid_ids = set(answer["sources"].keys())
        first = verify_answer(self.query, answer["answer"], answer["evidence_text"], valid_ids)
        outcome = {"first_pass": first, "revised": False}

        if first["status"] == "ok" and first["verdict"] in ("partially_supported", "unsupported") and first["feedback"]:
            # Reflection: revise once using the verifier's feedback, then re-check
            original = answer
            self.step_answer_question(feedback=first["feedback"])
            revised = self.step_results["answer_question"]
            if revised.get("status") == "ok":
                second = verify_answer(self.query, revised["answer"], revised["evidence_text"], set(revised["sources"].keys()))
                outcome.update({"revised": True, "second_pass": second})
                final = second
            else:
                self.step_results["answer_question"] = original  # keep the original draft
                final = first
        else:
            final = first

        outcome.update({
            "status": final["status"],
            "verdict": final["verdict"],
            "confidence": final["confidence"],
            "unsupported_claims": final["unsupported_claims"],
            "citation_check": final["citation_check"],
        })
        self.step_results["verify_answer"] = outcome

        detail = f"Verdict: {final['verdict'].replace('_', ' ')} (confidence {final['confidence']:.2f})"
        if outcome["revised"]:
            self.security("unsupported_claims_revised", "low",
                          f"Verifier found {len(first['unsupported_claims'])} unsupported claim(s); the answer was revised.")
            detail = (f"First draft: {first['verdict'].replace('_', ' ')} -> revised using verifier feedback. "
                      + detail)
        status = "ok" if final["verdict"] == "supported" else "warning"
        return status, detail

    def step_compare_papers(self):
        result = compare_papers(list(self.paper_infos.values()), focus=self.query)
        self.step_results["compare_papers"] = result
        status = {"ok": "ok", "insufficient_papers": "warning"}.get(result["status"], "error")
        return status, result["message"]

    def step_identify_research_gaps(self):
        comparison = self.step_results.get("compare_papers", {})
        differences = comparison.get("differences", "") if comparison.get("status") == "ok" else ""
        result = identify_research_gaps(list(self.paper_infos.values()), differences)
        self.step_results["identify_research_gaps"] = result
        detail = result["message"]
        if result.get("guardrail_changes"):
            self.security("overclaim_softened", "low",
                          f"{len(result['guardrail_changes'])} overclaiming phrase(s) rewritten in research gaps.")
            detail += f" Output guardrail softened {len(result['guardrail_changes'])} overclaiming phrase(s)."
        status = {"ok": "ok", "no_limitations_found": "warning"}.get(result["status"], "error")
        return status, detail

    def step_search_related_work(self):
        if not self.settings["enable_external_search"]:
            self.step_results["search_related_work"] = {"status": "skipped", "mode": "disabled", "searches": []}
            return "skipped", "External literature search disabled in settings."

        gaps = self.step_results.get("identify_research_gaps", {})
        if gaps.get("status") == "ok" and gaps.get("research_gaps"):
            mode = "prior_work_check"
            queries = [(g["search_query"], g["direction"]) for g in gaps["research_gaps"][:4]]
        else:
            mode = "related_work"
            metadata = self.step_results.get("extract_metadata", {})
            titles = [m["title"] for m in metadata.values() if m and m.get("status") == "ok" and m.get("title")]
            queries = [(t, t) for t in titles[:3]] or [(self.query, self.query)]

        searches = self._parallel(lambda q: {**search_literature(q[0], max_results=3), "for": q[1]}, queries)
        if mode == "prior_work_check":
            for gap, search in zip(gaps["research_gaps"], searches):
                gap["prior_work"] = search

        self.step_results["search_related_work"] = {"status": "ok", "mode": mode, "searches": searches}
        found = sum(len(s["results"]) for s in searches)
        failed = sum(1 for s in searches if s["status"] == "error")
        detail = f"{'Prior-work check' if mode == 'prior_work_check' else 'Related-work search'}: {len(searches)} API search(es), {found} publication(s) found"
        if failed:
            detail += f", {failed} search(es) failed (network/API)"
        return ("ok" if failed < len(searches) else "error"), detail

    def step_generate_report(self):
        report = generate_report(
            list(self.paper_infos.values()),
            comparison_result=self.step_results.get("compare_papers"),
            gap_result=self.step_results.get("identify_research_gaps"),
            answer_result=self.step_results.get("answer_question"),
            verification=self.step_results.get("verify_answer"),
            related_work=self.step_results.get("search_related_work"),
            plan=self.plan,
            query=self.query,
        )
        self.step_results["generate_report"] = report
        return "ok", f"Report assembled ({len(report.split())} words)."

    # ---------------- main loop ----------------
    def run(self) -> dict:
        started_run = time.time()
        llm_client.reset_usage()

        # 1. Input guardrails
        guard = validate_query(self.query)
        if not guard["allowed"]:
            self.emit("input_guardrail", "blocked", guard["message"], category=guard["category"])
            severity = "high" if guard["category"] in ("prompt_injection", "misuse") else "low"
            self.security("query_blocked", severity, f"Input guardrail blocked a query (category: {guard['category']}).")
            return self._result(guard["message"], started_run, blocked=True, guardrail=guard)
        self.emit("input_guardrail", "ok", guard["message"])

        # 1b. ML injection classifier (catches paraphrases the regex rules miss)
        if self.settings["enable_prompt_guard"]:
            t = time.time()
            verdict = prompt_guard.classify(self.query)
            if verdict["malicious"]:
                message = ("This request was classified as a likely prompt-injection or jailbreak attempt "
                           "(Llama Prompt Guard 2), so it was blocked. Please ask a question about the papers.")
                self.emit("ml_guardrail", "blocked", verdict["message"], t, category="prompt_injection_ml")
                self.security("query_blocked", "high",
                              f"ML classifier blocked a query (category: prompt_injection_ml, {verdict['message']})")
                return self._result(message, started_run, blocked=True,
                                    guardrail={"allowed": False, "category": "prompt_injection_ml", "message": message})
            if verdict["status"] == "unavailable":
                self.security("prompt_guard_unavailable", "low", verdict["message"])
            self.emit("ml_guardrail", "ok" if verdict["status"] == "ok" else "warning", verdict["message"], t)

        # 2. Plan
        t = time.time()
        self.plan = create_plan(self.query, self.paper_names, self.history)
        self.target_papers = self.plan["target_papers"]
        plan_status = "ok" if self.plan["status"] == "ok" else "warning"
        self.emit("plan", plan_status, self.plan["reasoning"], t, intent=self.plan["intent"],
                  steps=self.plan["steps"], target_papers=self.target_papers)

        requested = list(self.plan["steps"])

        # 3. Scope self-check
        if not self.plan["in_scope"]:
            t = time.time()
            probe = retrieve_relevant_chunks(self.store, self.query, top_k=3, mode=self.settings["retrieval_mode"])
            if probe["max_similarity"] >= SCOPE_OVERRIDE_THRESHOLD:
                requested = ["rag_retrieval", "answer_question"]
                self.plan["intent"] = "factual_question"
                self.emit("scope_check", "warning",
                          f"Planner marked the request out of scope, but the papers contain closely related "
                          f"passages (similarity {probe['max_similarity']:.2f}); overriding and answering from evidence.", t)
            else:
                self.security("out_of_scope_declined", "info", "Out-of-scope request declined; no tools run.")
                self.emit("scope_check", "ok",
                          f"Confirmed out of scope (best similarity to the papers {probe['max_similarity']:.2f}). No tools run.", t)
                return self._result(OUT_OF_SCOPE_ANSWER, started_run)

        # Policies (applied regardless of what the planner chose):
        #  - every written answer is verified
        #  - every research-gap suggestion gets an external prior-work check
        if "answer_question" in requested and "verify_answer" not in requested:
            requested.append("verify_answer")
        if ("identify_research_gaps" in requested and "search_related_work" not in requested
                and self.settings["enable_external_search"]):
            requested.append("search_related_work")

        # 4. Dependency resolution
        ordered, auto_added = resolve_dependencies(requested)
        if auto_added:
            self.emit("dependency_resolution", "info",
                      f"Auto-added prerequisite/policy step(s): {', '.join(auto_added)}")
        self.effective_steps = ordered

        # 5. Execute
        for step in ordered:
            t = time.time()
            handler = getattr(self, f"step_{step}", None)
            if handler is None:
                self.emit(step, "skipped", "Unknown step ignored.", t)
                continue
            try:
                status, detail = handler()
            except Exception as e:  # last-resort safety net: one tool never crashes the run
                status, detail = "error", f"{type(e).__name__}: {e}"
            self.emit(step, status, detail, t, auto_added=step in auto_added,
                      why=self.plan.get("step_reasons", {}).get(step, ""))

        return self._result(self._compose_final_answer(), started_run)

    # ---------------- final answer ----------------
    def _compose_final_answer(self) -> str:
        answer = self.step_results.get("answer_question")
        if answer and answer.get("answer"):
            return answer["answer"]

        parts = []
        comparison = self.step_results.get("compare_papers")
        if comparison:
            if comparison["status"] == "ok":
                parts.append(f"**Comparison of {len(comparison['table']['papers'])} papers** (AI-generated synthesis)\n\n"
                             f"- **Similarities:** {comparison['similarities']}\n"
                             f"- **Differences:** {comparison['differences']}"
                             + (f"\n- **Takeaway:** {comparison['key_takeaway']}" if comparison.get("key_takeaway") else ""))
            else:
                parts.append(f"Comparison: {comparison['message']}")

        gaps = self.step_results.get("identify_research_gaps")
        if gaps:
            if gaps["status"] == "ok":
                lines = [f"{i}. {g['direction']} _(based on: {g['based_on']})_"
                         for i, g in enumerate(gaps["research_gaps"], start=1)]
                parts.append("**Potential research directions** (suggestions, not verified novel contributions)\n\n" + "\n".join(lines))
            else:
                parts.append(f"Research gaps: {gaps['message']}")

        related = self.step_results.get("search_related_work")
        if related and related.get("mode") == "related_work":
            lines = [f"- {md_link(r['title'], r['url'])} ({r['year']}, {r['source']})"
                     for s in related["searches"] for r in s["results"]]
            parts.append("**Related publications found via external search**\n\n" + ("\n".join(lines) or "None found."))

        if not parts and self.paper_infos:
            lines = []
            for info in self.paper_infos.values():
                label = f"**{info.display_name}**" + (f" ({info.source_name})" if info.title else "")
                lines.append(f"{label}\n"
                             f"- Problem: {info.research_problem}\n- Method: {info.methodology}\n"
                             f"- Dataset: {info.dataset}\n- Key results: {info.results}")
            parts.append("\n\n".join(lines))

        if "generate_report" in self.step_results:
            parts.append("_The full structured report is available in the **Report** tab._")

        return "\n\n".join(parts) or "The agent completed its plan but produced no answer text. See the trace for details."

    def _result(self, final_answer, started_run, blocked=False, guardrail=None) -> dict:
        return {
            "query": self.query,
            "blocked": blocked,
            "guardrail": guardrail,
            "plan": self.plan,
            "effective_steps": getattr(self, "effective_steps", []),
            "step_results": self.step_results,
            "paper_infos": self.paper_infos,
            "final_answer": final_answer,
            "trace": self.trace,
            "security_events": self.security_events,
            "usage": llm_client.get_usage(),
            "duration_s": round(time.time() - started_run, 2),
        }


def run_agent(user_query: str, papers: dict, vector_store, history: list = None,
              cache: dict = None, settings: dict = None, on_event=None) -> dict:
    """
    Args:
        user_query: the user's natural language request
        papers: dict of source_name -> extraction_result (only "ok" papers)
        vector_store: the built VectorStore for RAG retrieval
        history: previous turns [{"query", "answer"}] for follow-up questions
        cache: per-session dict reused across runs (metadata/analysis cache)
        settings: overrides for DEFAULT_SETTINGS
        on_event: optional callback(event_dict) for live progress in the UI

    Returns:
        {query, blocked, guardrail, plan, effective_steps, step_results,
         paper_infos, final_answer, trace, usage, duration_s}
    """
    return AgentRun(user_query, papers, vector_store, history, cache, settings, on_event).run()
