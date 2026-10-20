import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from groq import Groq

import tools
from rag import MIN_SCORE
from safety import check_question

MODELS = [os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), "openai/gpt-oss-120b"]


class LLMError(Exception):
    pass


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the reply contained no JSON")
    return json.loads(text[start:end + 1])


class LLM:

    def __init__(self, api_key=None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("No Groq API key: paste one in the sidebar or put GROQ_API_KEY in a .env file.")
        self.client = Groq(api_key=key, timeout=45, max_retries=0)
        self.calls = self.tokens = 0
        self.lock = threading.Lock()

    def ask(self, system, user, temperature=0.2):
        errors = []
        for attempt in range(6):
            model = MODELS[attempt % 2]
            try:
                reply = self.client.chat.completions.create(
                    model=model, temperature=temperature, max_completion_tokens=4096,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system + "\nReply with a single JSON object."},
                              {"role": "user", "content": user}],
                    **({"reasoning_effort": "low"} if "gpt-oss" in model else {}))
                with self.lock:
                    self.calls += 1
                    self.tokens += reply.usage.total_tokens if reply.usage else 0
                return parse_json(reply.choices[0].message.content or "")
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in (401, 403):
                    raise LLMError("Groq rejected the API key.")
                errors.append(f"{model}: {e}"[:300])
                if status == 429 and attempt % 2:
                    headers = getattr(getattr(e, "response", None), "headers", None) or {}
                    time.sleep(min(float(headers.get("retry-after", 5)), 20))
        raise LLMError(re.sub(r"gsk_\w+", "gsk_***", "The language model failed: " + " | ".join(errors[-2:])))

    def injection_score(self, text):
        try:
            reply = self.client.chat.completions.create(model="meta-llama/llama-prompt-guard-2-86m",
                                                        messages=[{"role": "user", "content": text[:2000]}])
            return float(reply.choices[0].message.content)
        except Exception:
            return None


TOOLS = {
    "extract_metadata": ("title, authors and year of each paper", []),
    "analyze_papers": ("structured summary of each paper: problem, objectives, dataset, method, models, metrics, "
                       "results, limitations", []),
    "rag_retrieval": ("search the papers for passages relevant to the question", []),
    "answer_question": ("write a short answer that cites the retrieved passages", ["rag_retrieval"]),
    "verify_answer": ("fact-check the answer against the passages and fix it once if needed", ["answer_question"]),
    "compare_papers": ("comparison table plus similarities and differences", ["extract_metadata", "analyze_papers"]),
    "identify_research_gaps": ("potential research directions from the limitations the authors state",
                               ["extract_metadata", "analyze_papers"]),
    "search_related_work": ("search arXiv/OpenAlex for related publications", ["extract_metadata"]),
    "generate_report": ("a downloadable Markdown report of everything found", ["extract_metadata", "analyze_papers"]),
}

EXAMPLES = """Request: "What dataset did paper 1 use?"  (papers: bert.pdf, roberta.pdf)
{"reasoning": "Narrow factual question about one paper: a search and a cited answer are enough.", "in_scope": true,
 "papers": ["bert.pdf"], "steps": ["rag_retrieval", "answer_question"]}

Request: "Compare these papers and identify potential research gaps."  (papers: a.pdf, b.pdf, c.pdf)
{"reasoning": "Needs a structured analysis of every paper, a comparison, gaps, a literature check and a report.",
 "in_scope": true, "papers": [], "steps": ["extract_metadata", "analyze_papers", "compare_papers",
 "identify_research_gaps", "search_related_work", "generate_report"]}

Request: "Find recent papers related to this work."  (papers: x.pdf)
{"reasoning": "External literature search, using the paper titles.", "in_scope": true, "papers": [],
 "steps": ["extract_metadata", "search_related_work"]}

Request: "What's the weather in Mumbai today?"  (papers: x.pdf)
{"reasoning": "Not about the papers, so no tool should run.", "in_scope": false, "papers": [], "steps": []}"""


def make_plan(llm, question, papers, history=()):
    tool_list = "\n".join(f"- {name}: {desc}" for name, (desc, _) in TOOLS.items() if name != "verify_answer")
    system = f"""You are the planning step of a research-paper agent. You don't answer; you choose which tools run.
Tools:
{tool_list}
Choose the SMALLEST set of tools that fully answers the request. Only compare or look for gaps when asked.
If the request names a paper ("paper 1", a title or a file name), put its file name in "papers" (papers are numbered
in the order listed); otherwise use []. If the request is not about the papers, set in_scope to false and steps to [].
Use the earlier conversation to understand follow-up questions. Write the reasoning first.
Return JSON with keys reasoning, in_scope, papers, steps.

Examples:
{EXAMPLES}"""
    user = (f"Earlier conversation:\n{tools.chat(history)}\n\nPapers:\n" +
            "\n".join(f"  paper {i}: {p}" for i, p in enumerate(papers, 1)) + f'\n\nRequest: "{question}"')
    try:
        out = llm.ask(system, user, 0.0)
    except Exception as e:
        return fallback_plan(question, f"the planner failed ({str(e)[:80]})")
    steps = [s.get("tool") if isinstance(s, dict) else s for s in out.get("steps") or []]
    steps = list(dict.fromkeys(s for s in steps if s in TOOLS and s != "verify_answer"))
    names = {p.lower(): p for p in papers}
    chosen = list(dict.fromkeys(names[str(p).lower()] for p in out.get("papers") or [] if str(p).lower() in names))
    in_scope = out.get("in_scope") is not False
    if in_scope and not steps:
        return fallback_plan(question, "the planner chose no valid tools")
    return {"source": "llm", "reasoning": str(out.get("reasoning", "")), "in_scope": in_scope,
            "papers": [] if len(chosen) == len(papers) else chosen, "steps": steps if in_scope else []}


def fallback_plan(question, why):
    rules = [(r"\b(gaps?|future work|open problems?|research directions?)\b",
              ["extract_metadata", "analyze_papers", "compare_papers", "identify_research_gaps", "generate_report"]),
             (r"\b(compare|comparison|contrast|differ|difference|similar)", ["extract_metadata", "analyze_papers", "compare_papers"]),
             (r"\b(related|recent papers|literature|other papers)\b", ["extract_metadata", "search_related_work"]),
             (r"\b(report|review|summari[sz]e|overview)", ["extract_metadata", "analyze_papers", "generate_report"])]
    steps = next((s for pattern, s in rules if re.search(pattern, question.lower())), ["rag_retrieval", "answer_question"])
    return {"source": "rules", "reasoning": f"Keyword-based plan because {why}.", "in_scope": True, "papers": [],
            "steps": steps}


def with_prerequisites(steps):
    wanted = set(steps)
    if "answer_question" in wanted:
        wanted.add("verify_answer")
    if "identify_research_gaps" in wanted:
        wanted.add("search_related_work")
    while True:
        missing = {need for step in wanted for need in TOOLS[step][1]} - wanted
        if not missing:
            return [step for step in TOOLS if step in wanted]
        wanted |= missing


OUT_OF_SCOPE = ("That question doesn't seem to be about the uploaded papers, so I haven't answered it. "
                "Try asking about their datasets, methods, results, limitations or research gaps.")
RANK = {"yes": 2, "partial": 1, "no": 0}


class Agent:
    def __init__(self, llm, papers, index, history=(), cache=None, on_step=None):
        self.llm, self.papers, self.index = llm, papers, index
        self.history = list(history)[-3:]
        self.cache = {} if cache is None else cache
        self.on_step = on_step
        self.trace, self.results, self.infos, self.steps, self.plan = [], {}, {}, [], None

    def log(self, step, status, detail, started=None):
        event = {"step": step, "status": status, "detail": detail,
                 "seconds": round(time.time() - started, 1) if started else 0}
        self.trace.append(event)
        if self.on_step:
            self.on_step(event)

    def run(self, question):
        self.question, started = question, time.time()
        blocked = check_question(question)
        if blocked:
            self.log("guardrails", "blocked", blocked)
            return self.finish(blocked, started, blocked=True)
        score = self.llm.injection_score(question)
        if score is not None and score >= 0.8:
            self.log("guardrails", "blocked", f"Prompt Guard rated this {score:.3f} likely to be a jailbreak.")
            return self.finish("This question looks like a jailbreak or prompt-injection attempt, so it was blocked.",
                               started, blocked=True)
        self.log("guardrails", "ok", "Question passed" + (f" (Prompt Guard score {score:.3f})" if score is not None else ""))

        t = time.time()
        self.plan = make_plan(self.llm, question, list(self.papers), self.history)
        self.log("plan", "ok" if self.plan["source"] == "llm" else "warning", self.plan["reasoning"], t)
        steps = self.plan["steps"]
        if not self.plan["in_scope"]:
            best = max((c["score"] for c in self.index.search(question, 3)), default=0)
            if best < 0.45:
                self.log("scope_check", "ok", f"Not about the papers (best match {best:.2f}), so no tools run.")
                return self.finish(OUT_OF_SCOPE, started)
            self.log("scope_check", "warning", f"The papers match this well ({best:.2f}), so it is answered anyway.")
            steps = ["rag_retrieval", "answer_question"]

        self.steps = with_prerequisites(steps)
        added = [s for s in self.steps if s not in steps]
        if added:
            self.log("dependencies", "info", "Added automatically: " + ", ".join(added))
        for step in self.steps:
            t = time.time()
            try:
                status, detail = getattr(self, step)()
            except Exception as e:
                status, detail = "error", f"{type(e).__name__}: {e}"
            self.log(step, status, detail, t)
        return self.finish(self.final_answer(), started)

    def finish(self, answer, started, blocked=False):
        return {"question": self.question, "answer": answer, "blocked": blocked, "plan": self.plan,
                "steps": self.steps, "trace": self.trace, "results": self.results, "infos": self.infos,
                "calls": self.llm.calls, "tokens": self.llm.tokens, "seconds": round(time.time() - started, 1)}

    def targets(self):
        return self.plan["papers"] or list(self.papers)

    def each_paper(self, kind, work):
        todo = [p for p in self.targets() if kind not in self.cache.get(p, {})]

        def attempt(name):
            try:
                return work(name)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=3) as pool:
            for name, out in zip(todo, pool.map(attempt, todo)):
                if out:
                    self.cache.setdefault(name, {})[kind] = out
        done = [p for p in self.targets() if kind in self.cache.get(p, {})]
        reused = len(self.targets()) - len(todo)
        return ("ok" if len(done) == len(self.targets()) else "warning",
                f"{len(done)}/{len(self.targets())} papers" + (f" ({reused} reused from earlier)" if reused else ""))


    def extract_metadata(self):
        return self.each_paper("metadata", lambda name: tools.metadata(self.llm, self.papers[name]))

    def analyze_papers(self):
        status, detail = self.each_paper("analysis", lambda name: tools.analyze(self.llm, self.index, name))
        for name in self.targets():
            if "analysis" in self.cache.get(name, {}):
                self.infos[name] = tools.paper_info(name, self.cache[name].get("metadata"), self.cache[name]["analysis"])
        return status, detail

    def search(self, query, k=6):
        hits = self.index.search(query, k, self.plan["papers"] or None, min_score=MIN_SCORE)
        if len(self.targets()) > 1:
            for paper in self.targets():
                hits += self.index.search(query, 2, [paper], min_score=MIN_SCORE)
        return hits

    def merge(self, *lists, k=None):
        best = {}
        for c in (c for found in lists for c in found):
            if c["id"] not in best or c["score"] > best[c["id"]]["score"]:
                best[c["id"]] = c
        return sorted(best.values(), key=lambda c: -c["score"])[:k or 6 + len(self.targets())]

    def rag_retrieval(self):
        query = self.question
        if self.history and len(query.split()) <= 8:
            query = self.history[-1]["question"] + " " + query
        chunks = self.merge(self.search(query))
        rewrites = []
        if max((c["score"] for c in chunks), default=0) < 0.35:
            rewrites = tools.rewrite_queries(self.llm, self.question, self.targets())
            chunks = self.merge(chunks, *[self.search(q) for q in rewrites])
        self.results["rag_retrieval"] = {"chunks": chunks, "rewrites": rewrites}
        if not chunks:
            return "warning", "No passage was relevant enough to use as evidence."
        return "ok", (f"{len(chunks)} passages (best match {chunks[0]['score']:.2f})" +
                      (f"; the first search was weak, so it was rephrased {len(rewrites)} ways" if rewrites else ""))

    def answer_question(self, feedback=""):
        chunks = self.results["rag_retrieval"]["chunks"]
        infos = list(self.infos.values())
        ans = tools.answer(self.llm, self.question, chunks, infos, self.history, feedback)
        if not feedback and chunks and ans["answerable"] in ("no", "partial"):
            queries = tools.rewrite_queries(self.llm, self.question, self.targets(), ans["text"])
            more = self.merge(chunks, *[self.search(q, k=10) for q in queries], k=len(chunks) + 6)
            if {c["id"] for c in more} - {c["id"] for c in chunks}:
                second = tools.answer(self.llm, self.question, more, infos, self.history)
                if RANK.get(second["answerable"], 0) > RANK.get(ans["answerable"], 0):
                    self.results["rag_retrieval"]["chunks"] = more
                    ans = dict(second, first_try=ans["answerable"])
        self.results["answer_question"] = ans
        note = f"; the first draft was '{ans['first_try']}', so it searched again" if "first_try" in ans else ""
        return "ok", f"Answer from {len(ans['sources'])} passages ({ans['answerable']}){note}"

    def verify_answer(self):
        ans = self.results.get("answer_question")
        if not ans or not ans["sources"]:
            return "skipped", "Nothing to check (no evidence-based answer)."
        check = tools.verify(self.llm, self.question, ans["text"], list(ans["sources"].values()))
        if check["verdict"] != "supported" and check["problems"]:
            self.answer_question(feedback="\n".join(check["problems"]))
            fixed = self.results["answer_question"]
            check = dict(tools.verify(self.llm, self.question, fixed["text"], list(fixed["sources"].values())),
                         first=check)
        self.results["verify_answer"] = check
        detail = f"Verdict: {check['verdict'].replace('_', ' ')}"
        if "first" in check:
            detail = f"First draft was {check['first']['verdict'].replace('_', ' ')}, so it was revised. " + detail
        return ("ok" if check["verdict"] == "supported" else "warning"), detail

    def compare_papers(self):
        out = self.results["compare_papers"] = tools.compare(self.llm, list(self.infos.values()), self.question)
        return ("ok", f"Compared {len(self.infos)} papers") if out["status"] == "ok" else ("warning", out["message"])

    def identify_research_gaps(self):
        differences = self.results.get("compare_papers", {}).get("differences", "")
        out = self.results["identify_research_gaps"] = tools.gaps(self.llm, list(self.infos.values()), differences)
        if out["status"] != "ok":
            return "warning", out["message"]
        return "ok", f"{len(out['gaps'])} potential directions from {len(out['limitations'])} stated limitations"

    def search_related_work(self):
        gap = self.results.get("identify_research_gaps", {})
        if gap.get("gaps"):
            with ThreadPoolExecutor(max_workers=4) as pool:
                found = list(pool.map(lambda g: tools.search_literature(g["query"]), gap["gaps"][:4]))
            for g, papers in zip(gap["gaps"], found):
                g["prior_work"] = papers
            return "ok", f"Prior-work check: {sum(map(len, found))} existing publications for {len(found)} gaps"
        titles = [self.cache.get(p, {}).get("metadata", {}).get("title") for p in self.targets()]
        found = [p for q in ([t for t in titles if t][:3] or [self.question]) for p in tools.search_literature(q)]
        self.results["search_related_work"] = found
        return ("ok" if found else "warning"), f"{len(found)} related publications on arXiv/OpenAlex"

    def generate_report(self):
        self.results["generate_report"] = tools.report(self.question, list(self.infos.values()), self.results, self.plan)
        return "ok", "Report ready in the Report tab"

    def final_answer(self):
        r = self.results
        if "answer_question" in r:
            return r["answer_question"]["text"]
        parts = []
        comp = r.get("compare_papers")
        if comp:
            parts.append(comp["message"] if comp["status"] != "ok" else
                         f"**Comparison** _(AI-generated)_\n\n- **Similarities:** {comp['similarities']}\n"
                         f"- **Differences:** {comp['differences']}\n- **Takeaway:** {comp['takeaway']}")
        gap = r.get("identify_research_gaps")
        if gap:
            parts.append(gap["message"] if gap["status"] != "ok" else
                         "**Potential research directions** _(suggestions, not confirmed novel ideas)_\n\n" +
                         "\n".join(f"{n}. {g['direction']} _(based on: {g['based_on']})_"
                                   for n, g in enumerate(gap["gaps"], 1)))
        if "search_related_work" in r:
            parts.append("**Related publications**\n\n" + ("\n".join(
                f"- {tools.link(p['title'], p['url'])} ({p['year']}, {p['source']})" for p in r["search_related_work"])
                or "None found."))
        if not parts:
            parts += [f"**{i['title'] or i['paper']}**\n- Problem: {i['problem']}\n- Method: {i['method']}\n"
                      f"- Dataset: {i['dataset']}\n- Results: {i['results']}" for i in self.infos.values()]
        if "generate_report" in r:
            parts.append("_The full report is in the Report tab._")
        return "\n\n".join(parts) or "Done - see the Trace tab for details."


def run_agent(question, papers, index, llm, history=(), cache=None, on_step=None):
    return Agent(llm, papers, index, history, cache, on_step).run(question)
