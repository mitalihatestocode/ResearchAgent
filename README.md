# ResearchAgent

**An agentic AI system for research paper analysis, comparison and research gap identification.**

Upload a few research papers and ask questions in plain English. For every question the agent **plans** which tools it needs, **searches** the papers (hybrid embedding + keyword search), **answers with page citations**, **fact-checks its own answer**, compares papers, suggests **potential research gaps** and checks them against **arXiv/OpenAlex**, all behind safety guardrails.

![Chat](docs/screenshots/03-chat-answer.png)

## Start it (one step)

```bash
./run.sh          # macOS / Linux
run.bat           # Windows
```

The first run creates a virtual environment and installs the packages (a few minutes, needs Python 3.11 or newer). Step-by-step manual instructions and troubleshooting are in **[HOW_TO_RUN.md](HOW_TO_RUN.md)**. Then the app opens at http://localhost:8501:

1. Create an account (or continue as a guest).
2. Paste a free Groq API key in the sidebar ([console.groq.com/keys](https://console.groq.com/keys)), or put `GROQ_API_KEY=...` in a `.env` file.
3. Click **Load 3 sample papers** (downloaded from arXiv) or upload your own PDFs, and ask a question.

## What it can do

| | |
|---|---|
| **Plans its own steps** | an LLM planner picks from 9 tools; missing prerequisites are added automatically |
| **Answers from the papers** | hybrid search (FAISS embeddings + BM25), page citations, "not reported" instead of guessing |
| **Checks itself** | a fact-checker reviews each answer and it is fixed once if needed; incomplete answers trigger a deeper search |
| **Compares and finds gaps** | structured summary of each paper, comparison table, potential research gaps, prior-work check on arXiv |
| **Stays safe** | blocks prompt injection (typed or hidden inside PDFs) and misuse, Llama Prompt Guard 2, rejects risky files |
| **Remembers you** | accounts keep your papers, analyses and chat history, so nothing is uploaded or analysed twice |
| **Shows its work** | Trace tab with the plan and every step; Paper Galaxy map of the papers; downloadable report |

| Agent at work | Paper Galaxy | Trace |
|---|---|---|
| ![Thinking](docs/screenshots/10-agent-thinking.png) | ![Galaxy](docs/screenshots/11-paper-galaxy.png) | ![Trace](docs/screenshots/04-trace.png) |

## Code

| File | What it does |
|---|---|
| `app.py` | Streamlit app: sign-in, sidebar, chat, trace, analysis, galaxy, report |
| `agent.py` | LLM client, planner, and the agent that runs the tools and checks its answers |
| `tools.py` | the tools and their prompts: metadata, analysis, answer, fact-check, compare, gaps, literature search, report |
| `rag.py` | PDF reading, chunking, embeddings, hybrid search, galaxy projection |
| `safety.py` | guardrails for questions, documents, files and output |
| `storage.py` | accounts and saved papers / chats (SQLite, `data/app.db`) |
| `style.css` | iOS-style look and animations |
| `tests/` | `test_project.py` (30 offline tests), `evaluate.py` + `benchmark.json` (quality benchmark) |

```bash
python -m pytest -q          # 30 tests, about 1 second
python tests/evaluate.py     # benchmark with the real LLM
```

## Documents

- [Short report](docs/REPORT.md) (all 12 sections) · [Test report](docs/TEST_REPORT.md) · [Evaluation results](docs/evaluation_results.md) · [Viva guide](docs/VIVA_GUIDE.md)
- [Architecture diagram](docs/architecture.png) · PDF and Word versions in `docs/submission/`
