"""ResearchAgent: an AI agent that reads research papers, compares them and suggests research gaps.
Start it with ./run.sh (Windows: run.bat) or `streamlit run app.py`."""
import html
import io
import os
import re
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import storage
import tools
from agent import LLM, LLMError, run_agent
from rag import Index, chunk, embed, project_2d, read_pdf
from safety import check_pdf, has_injection

load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = {"rag_lewis_2020.pdf": "2005.11401", "self_rag_2023.pdf": "2310.11511", "corrective_rag_2024.pdf": "2401.15884"}
EXAMPLES = ["Compare these papers and identify potential research gaps.",
            "Which datasets and evaluation metrics does each paper use?",
            "What limitations do the authors report?", "Find recent papers related to this work."]
STATUS = {"ok": ("✓", "green"), "warning": ("!", "orange"), "error": ("✕", "red"), "blocked": ("⛔", "red"),
          "skipped": ("–", ""), "info": ("i", "blue")}
VERDICT = {"supported": ("✓ Verified against the cited passages", "green"),
           "partially_supported": ("! Partly verified - check the sources", "orange"),
           "unsupported": ("✕ Could not verify every claim", "red")}

st.set_page_config(page_title="ResearchAgent", page_icon="🔬", layout="wide")
with open(os.path.join(ROOT, "style.css")) as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

s = st.session_state
for key, value in {"user": None, "guest": False, "papers": {}, "shas": {}, "chunks": {}, "vectors": {}, "cache": {},
                   "chat": [], "seen": set(), "notice": "", "pending": None, "api_key": ""}.items():
    if key not in s:
        s[key] = value
if "index" not in s:
    s.index = Index()

# ---------------- small HTML helpers (everything shown is escaped first) ----------------

def esc(text):
    return html.escape(str(text if text is not None else ""))


def md_safe(text):
    return re.sub(r"([\\`*_\[\]<>#|])", r"\\\1", str(text))  # show text literally inside Markdown


def show(markup):
    st.markdown(markup, unsafe_allow_html=True)


def pills(items):
    show('<div class="pills">' + "".join(f'<span class="pill {c}">{esc(t)}</span>' for t, c in items) + "</div>")


def rows(items):
    """iOS-style grouped list. items: (icon, colour, title, subtitle, text on the right)"""
    show('<div class="list">' + "".join(
        f'<div class="row"><span class="icon {c}">{esc(i)}</span><div class="body"><b>{esc(t)}</b>'
        f'<div class="sub">{esc(sub)}</div></div><span class="trail">{esc(right)}</span></div>'
        for i, c, t, sub, right in items) + "</div>")


def section(title):
    show(f'<div class="section">{esc(title)}</div>')


def empty(icon, title, text):
    show(f'<div class="empty"><div class="big">{esc(icon)}</div><h3>{esc(title)}</h3><p>{esc(text)}</p></div>')


def hero(subtitle, extra=""):
    show(f'<div class="eyebrow">Research paper agent</div><div class="title">ResearchAgent</div>'
         f'<div class="subtitle">{esc(subtitle)} {extra}</div>')

# ---------------- papers and accounts ----------------

def add_paper(data, filename):
    """Checks, reads, splits and indexes one PDF; signed-in users also get it saved."""
    name = re.sub(r"[^\w.\- ()]", "_", os.path.basename(filename))[:80] or "paper.pdf"
    ok, message, sha = check_pdf(data)
    if not ok:
        s.notice = f"{name}: {message}"
        return
    if sha in s.shas.values():
        s.notice = f"{name} is already loaded."
        return
    if len(s.papers) >= 5:
        s.notice = "You can analyse at most 5 papers at a time."
        return
    while name in s.papers:
        name = "copy_" + name
    paper = read_pdf(io.BytesIO(data), name)
    s.papers[name], s.shas[name] = paper, sha
    if paper["status"] != "ok":
        s.notice = f"{name}: {paper['message']}"
        return
    s.chunks[name] = chunk(paper)
    s.vectors[name] = s.index.add(s.chunks[name])
    if any(has_injection(c["text"]) for c in s.chunks[name]):
        s.notice = f"{name} contains hidden instructions. They are removed before the AI reads the text."
    if s.user:
        storage.save_paper(s.user, sha, paper, s.chunks[name], s.vectors[name])


def load_samples():
    folder = os.path.join(ROOT, "data", "sample_papers")
    os.makedirs(folder, exist_ok=True)
    try:
        for name, arxiv_id in SAMPLES.items():
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                data = tools.fetch(f"https://arxiv.org/pdf/{arxiv_id}")
                with open(path, "wb") as f:
                    f.write(data)
            with open(path, "rb") as f:
                add_paper(f.read(), name)
    except Exception as e:
        s.notice = f"Could not download the sample papers ({e})."


def remove_papers():
    if s.user:
        for sha in s.shas.values():
            storage.delete_paper(s.user, sha)
    s.papers, s.shas, s.chunks, s.vectors, s.cache, s.index = {}, {}, {}, {}, {}, Index()


def clear_chat():
    s.chat = []
    if s.user:
        storage.clear_turns(s.user)


def open_account(user):
    """Signs in and restores the user's papers (with their saved vectors), analyses and chat history."""
    s.user, s.guest = user, False
    s.papers, s.shas, s.chunks, s.vectors, s.cache, s.index = {}, {}, {}, {}, {}, Index()
    for saved in storage.load_papers(user):
        name = saved["paper"]["name"]
        s.papers[name], s.shas[name], s.chunks[name] = saved["paper"], saved["sha"], saved["chunks"]
        s.cache[name] = saved["cache"]
        s.vectors[name] = s.index.add(saved["chunks"], saved["vectors"])
    s.chat = storage.load_turns(user)


def sign_out():
    for key in list(s.keys()):
        del s[key]


def queue(question):
    s.pending = question


def pick_example():
    queue(s.pick)
    s.pick = None


def login_screen():
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        hero("Sign in to keep your papers, analyses and chat history between visits.")
        with st.container(key="card_login"):
            sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])
            with sign_in_tab, st.form("sign_in"):
                name = st.text_input("Username", key="in_name")
                password = st.text_input("Password", type="password", key="in_password")
                if st.form_submit_button("Sign in", type="primary", width="stretch"):
                    user, message = storage.sign_in(name, password)
                    if user:
                        open_account(user)
                        st.rerun()
                    st.error(message)
            with sign_up_tab, st.form("sign_up"):
                name = st.text_input("Username", key="up_name", help="3-32 letters, numbers, dots, dashes or underscores")
                password = st.text_input("Password", type="password", key="up_password",
                                         help="At least 8 characters with a letter and a number")
                again = st.text_input("Repeat password", type="password", key="up_again")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    user, message = storage.sign_up(name, password, again)
                    if user:
                        open_account(user)
                        st.rerun()
                    st.error(message)
        if st.button("Continue as guest (nothing is saved)", width="stretch"):
            s.guest = True
            st.rerun()
        st.caption("Passwords are stored only as salted PBKDF2 hashes. Five wrong attempts lock an account for 5 minutes.")


if not s.user and not s.guest:
    login_screen()
    st.stop()

# ---------------- sidebar ----------------

with st.sidebar:
    section("Account")
    if s.user:
        rows([(s.user["name"][0].upper(), "blue", s.user["name"], "Your papers and chats are saved", "")])
    else:
        rows([("?", "", "Guest", "Nothing is saved - sign in to keep your work", "")])
    st.button("Sign out" if s.user else "Sign in or create an account", on_click=sign_out, width="stretch")

    section("Groq API key")
    if os.getenv("GROQ_API_KEY"):
        rows([("✓", "green", "Connected", "Key loaded from the .env file", "")])
    else:
        st.text_input("Groq API key", type="password", key="api_key", placeholder="gsk_...", label_visibility="collapsed",
                      help="Free key from console.groq.com/keys. It is only kept for this browser session.")

    section("Papers")
    for upload in st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed") or []:
        if upload.file_id not in s.seen:
            s.seen.add(upload.file_id)
            with st.spinner(f"Reading {upload.name}..."):
                add_paper(upload.getvalue(), upload.name)
    if not s.papers and st.button("Load 3 sample papers", type="primary", width="stretch"):
        with st.spinner("Downloading the sample papers from arXiv..."):
            load_samples()
        st.rerun()
    if s.notice:
        st.warning(s.notice)
        s.notice = ""
    if s.papers:
        rows([("✓", "green", name, f"{len(p['pages'])} pages · {len(s.chunks.get(name, []))} passages", "")
              if p["status"] == "ok" else ("!", "orange", name, p["message"], "") for name, p in s.papers.items()])
    st.button("Clear chat history", on_click=clear_chat, width="stretch", disabled=not s.chat)
    st.button("Remove all papers", on_click=remove_papers, width="stretch", disabled=not s.papers)
    if s.user:
        with st.expander("Delete my account"):
            password = st.text_input("Confirm with your password", type="password", key="delete_password")
            if st.button("Delete account and saved data"):
                if storage.delete_account(s.user, password):
                    sign_out()
                    st.rerun()
                st.error("Wrong password.")

# ---------------- running the agent ----------------

ok_papers = {name: p for name, p in s.papers.items() if p["status"] == "ok"}
infos = {name: tools.paper_info(name, c.get("metadata"), c["analysis"])
         for name, c in s.cache.items() if c.get("analysis") and name in ok_papers}
latest = {}
for turn in s.chat:
    latest.update(turn["results"])
has_key = bool(s.api_key or os.getenv("GROQ_API_KEY"))


def ask(question):
    """Runs the agent with a live, glowing progress panel, then saves the result."""
    try:
        llm = LLM(s.api_key or None)
    except LLMError as e:
        st.error(str(e))
        return None
    history = [{"question": t["question"], "answer": t["answer"]} for t in s.chat if not t["blocked"]]
    typing, panel = st.empty(), st.empty()
    typing.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
    with panel.container(key="thinking"), st.status("Thinking...", expanded=True) as status:
        def on_step(event):
            status.markdown(f"**{STATUS.get(event['status'], ('•', ''))[0]} {event['step']}** - {md_safe(event['detail'])}")
        result = run_agent(question, ok_papers, s.index, llm, history, s.cache, on_step)
    typing.empty()
    panel.empty()
    s.chat.append(result)
    latest.update(result["results"])
    if s.user:
        storage.save_turn(s.user, result)
        for name, cached in s.cache.items():
            if name in s.shas:
                storage.save_cache(s.user, s.shas[name], cached)
    return result


def typewriter(text):
    words = re.findall(r"\S+\s*", text)
    for word in words:
        yield word
        time.sleep(min(0.035, 1.6 / len(words)))


def show_turn(turn, number, animate=False, with_question=True):
    if with_question:
        show(f'<div class="me"><div>{esc(turn["question"])}</div></div>')
    with st.container(key=f"answer_{number}"):
        show('<div class="ai">✦ ResearchAgent</div>')
        if turn["blocked"]:
            pills([("Blocked by the guardrails", "red")])
            st.markdown(md_safe(turn["answer"]))
            return
        chips = [(f"{len(turn['steps'])} tools", "purple"), (f"{turn['calls']} LLM calls", ""), (f"{turn['seconds']}s", "")]
        if turn["plan"] and turn["plan"]["papers"]:
            chips.append(("only " + ", ".join(turn["plan"]["papers"]), ""))
        verdict = turn["results"].get("verify_answer", {}).get("verdict")
        if verdict in VERDICT:
            chips.append(VERDICT[verdict])
        pills(chips)
        if animate:
            st.write_stream(typewriter(turn["answer"]))
        else:
            st.markdown(turn["answer"])
        sources = turn["results"].get("answer_question", {}).get("sources", {})
        if sources:
            with st.expander(f"Sources ({len(sources)} passages)"):
                for sid, c in sources.items():
                    st.markdown(f"**[{sid}]** {md_safe(c['paper'])}, page {c['page']} ({c['section']}), "
                                f"match {c['score']:.2f}")
                    show(f'<div class="quote">{esc(c["text"][:600])}</div>')


def flow_graph(turn):
    """Graphviz picture of the steps that ran: dashed = added automatically, border colour = result."""
    colours = {"ok": "#34C759", "warning": "#FF9500", "error": "#FF3B30", "skipped": "#8E8E93"}
    status = {e["step"]: e["status"] for e in turn["trace"]}
    dot = ['digraph { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor=white, '
           'fontname=Helvetica, fontsize=11]; q [label="Question", shape=oval, fillcolor="#007AFF", fontcolor=white]; '
           'p [label="Planner", fillcolor="#5856D6", fontcolor=white]; q -> p;']
    previous = "p"
    for n, step in enumerate(turn["steps"]):
        dashed = ', style="rounded,filled,dashed"' if step not in turn["plan"]["steps"] else ""
        dot.append(f's{n} [label="{step}", color="{colours.get(status.get(step), "#34C759")}", penwidth=2{dashed}]; '
                   f'{previous} -> s{n};')
        previous = f"s{n}"
    dot.append(f'a [label="Answer", shape=oval, fillcolor="#34C759", fontcolor=white]; {previous} -> a; }}')
    return "\n".join(dot)


STAR = "M0,-1L0.29,-0.4L0.95,-0.31L0.47,0.15L0.59,0.81L0,0.5L-0.59,0.81L-0.47,0.15L-0.95,-0.31L-0.29,-0.4Z"


def galaxy(question, colour_by):
    """Every passage as a dot placed by meaning; the question is a star linked to the passages it retrieves."""
    names = [n for n in s.chunks if s.chunks[n]]
    passages = [c for n in names for c in s.chunks[n]]
    xy, place, kept = project_2d(np.vstack([s.vectors[n] for n in names]))
    dots = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "id": [c["id"] for c in passages],
                         "paper": [c["paper"] for c in passages], "section": [c["section"] for c in passages],
                         "page": [c["page"] for c in passages], "text": [c["text"][:160] for c in passages]})
    colour = alt.Color(f"{colour_by}:N", scale=alt.Scale(range=["#007AFF", "#FF9500", "#AF52DE", "#34C759", "#FF2D55"]),
                       legend=alt.Legend(orient="bottom", title=None))
    tooltip = ["paper", "section", "page", "text"]
    layers = [alt.Chart(dots).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None), color=colour, tooltip=tooltip)]
    if question:
        qx, qy = place(embed([question])[0])
        hits = dots[dots["id"].isin([c["id"] for c in s.index.search(question, 6)])].assign(qx=qx, qy=qy)
        layers += [alt.Chart(hits).mark_rule(color="#FF9500", strokeDash=[4, 3]).encode(x="qx", y="qy", x2="x", y2="y"),
                   alt.Chart(hits).mark_circle(size=260, stroke="#FF9500", strokeWidth=2.5).encode(
                       x="x", y="y", color=colour, tooltip=tooltip),
                   alt.Chart(pd.DataFrame({"x": [qx], "y": [qy], "question": [question]})).mark_point(
                       shape=STAR, size=900, filled=True, color="#FFCC00", stroke="#1C1C1E").encode(
                       x="x", y="y", tooltip="question")]
    chart = alt.layer(*layers).properties(height=520).configure_view(strokeWidth=0).interactive()
    return chart, kept, len(dots)

# ---------------- page ----------------

ready = ('<span class="pill green"><span class="live"></span>LLM ready</span>' if has_key
         else '<span class="pill orange">Add a Groq API key in the sidebar</span>')
hero("Plans its own steps · answers from your papers with page citations · checks its own answers", ready)
widgets = st.empty()
chat_tab, trace_tab, analysis_tab, galaxy_tab, report_tab = st.tabs(["Chat", "Trace", "Analysis", "Galaxy", "Report"])

with chat_tab:
    if not ok_papers:
        empty("🔬", "Add some research papers", "Upload 2-5 PDFs in the sidebar or load the 3 sample papers. Then ask a "
              "question: the agent plans its steps, searches the papers, cites pages and checks its own answer.")
    else:
        for n, turn in enumerate(s.chat):
            show_turn(turn, n)
        if s.pending and has_key:
            question, s.pending = s.pending, None
            show(f'<div class="me"><div>{esc(question)}</div></div>')
            result = ask(question)
            if result:
                show_turn(result, len(s.chat) - 1, animate=True, with_question=False)
        if not has_key:
            st.info("Paste a free Groq API key in the sidebar to start asking questions.")
        else:
            if not s.chat:
                section("Try asking")
                st.pills("Try asking", EXAMPLES, key="pick", label_visibility="collapsed", on_change=pick_example)
            st.chat_input("Ask about your papers...", key="box", max_chars=1000, on_submit=lambda: queue(s.box))

with trace_tab:
    if not s.chat:
        empty("🧭", "No questions yet", "The plan for each question, and every step the agent took, will appear here.")
    else:
        turn = s.chat[-1]
        section("Question")
        rows([("?", "blue", turn["question"], f"{len(turn['steps'])} tools · {turn['calls']} LLM calls · "
                                              f"{turn['tokens']} tokens · {turn['seconds']}s", "")])
        if turn["plan"]:
            section("Plan")
            rows([("◎", "purple", "Planner (" + ("LLM" if turn["plan"]["source"] == "llm" else "keyword rules") + ")",
                   turn["plan"]["reasoning"], ""),
                  ("→", "indigo", "Chosen tools", " → ".join(turn["plan"]["steps"]) or "none", "")])
            st.graphviz_chart(flow_graph(turn), width="stretch")
            st.caption("Solid boxes were chosen by the planner, dashed ones were added automatically. "
                       "Border colour: green = ok, orange = warning, red = error.")
        section("Every step")
        rows([(*STATUS.get(e["status"], ("•", "")), e["step"], e["detail"], f"{e['seconds']}s" if e["seconds"] else "")
              for e in turn["trace"]])

with analysis_tab:
    if not infos and not latest.get("search_related_work"):
        empty("📊", "No analysis yet", "Ask “Compare these papers and identify potential research gaps” to fill this page.")
    for name, info in infos.items():
        with st.expander(info["title"] or name):
            pills([(name, ""), (info["authors"] or "authors not found", ""), (info["year"] or "year not found", "")])
            rows([("✓" if info[key] != tools.NOT_FOUND else "–", "green" if info[key] != tools.NOT_FOUND else "", label,
                   info[key], ("p. " + ", ".join(map(str, info["pages"][key]))) if info["pages"].get(key) else "")
                  for key, label in tools.FIELDS.items()])
    comparison = latest.get("compare_papers")
    if comparison and comparison["status"] == "ok":
        section("Comparison (taken from the papers)")
        st.dataframe(pd.DataFrame(comparison["table"]).T, width="stretch")
        rows([("=", "blue", "Similarities (AI-generated)", comparison["similarities"], ""),
              ("≠", "purple", "Differences (AI-generated)", comparison["differences"], ""),
              ("★", "orange", "Takeaway (AI-generated)", comparison["takeaway"], "")])
    gap = latest.get("identify_research_gaps")
    if gap and gap["status"] == "ok":
        section("Potential research gaps (suggestions, not confirmed novel ideas)")
        for n, g in enumerate(gap["gaps"], 1):
            with st.container(key=f"card_gap_{n}"):
                st.markdown(f"**{n}. {md_safe(g['direction'])}**")
                st.caption("Based on: " + md_safe(g["based_on"]))
                if g.get("prior_work"):
                    st.markdown("Possibly related existing work:  \n" + "  \n".join(
                        f"• {tools.link(p['title'], p['url'])} ({p['year']})" for p in g["prior_work"]))
    if latest.get("search_related_work"):
        section("Related publications (arXiv / OpenAlex)")
        st.markdown("\n".join(f"- {tools.link(p['title'], p['url'])} ({p['year']}, {p['source']})"
                              for p in latest["search_related_work"]))

with galaxy_tab:
    if sum(len(c) for c in s.chunks.values()) < 3:
        empty("✦", "Your galaxy is empty", "Add papers to see every passage placed on a map by meaning.")
    else:
        st.caption("Every dot is a passage. Passages with similar meaning sit close together (their 384-number "
                   "embeddings squeezed into 2-D with PCA). Type a question to see where it lands (★) and which "
                   "passages the agent would retrieve (orange). Scroll to zoom, hover to read.")
        left, right = st.columns([3, 1])
        question = left.text_input("Drop a question into the galaxy", value=s.chat[-1]["question"] if s.chat else "",
                                   placeholder="e.g. Which datasets were used?", key=f"galaxy_{len(s.chat)}")
        colour_by = right.segmented_control("Colour by", ["paper", "section"], default="paper") or "paper"
        chart, kept, count = galaxy(question.strip(), colour_by)
        with st.container(key="card_galaxy"):
            st.altair_chart(chart, width="stretch")
        st.caption(f"{count} passages · the flat map keeps {kept:.0%} of the information; "
                   f"the agent itself searches using all 384 dimensions.")

with report_tab:
    report = latest.get("generate_report")
    if not report:
        empty("📄", "No report yet", "Ask for one, e.g. “Compare these papers and identify potential research gaps”.")
        if ok_papers and has_key:
            st.button("Write a full report", type="primary", on_click=queue,
                      args=("Write a full report comparing these papers, including potential research gaps.",))
    else:
        st.download_button("Download report (.md)", report, "research_report.md", type="primary")
        with st.container(key="card_report"):
            st.markdown(report)

last = s.chat[-1] if s.chat else None
widgets.markdown('<div class="widgets">' + "".join(
    f'<div class="widget"><div class="label">{label}</div><div class="value">{esc(value)}</div></div>'
    for label, value in [("Papers", len(ok_papers)), ("Passages indexed", len(s.index.chunks)),
                         ("Questions", len(s.chat)), ("Last answer", f"{last['seconds']}s" if last else "—")]) + "</div>",
    unsafe_allow_html=True)
