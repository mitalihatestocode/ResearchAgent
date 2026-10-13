# ResearchAgent

An Agentic AI System for Research Paper Analysis, Comparison and Research Gap Identification.

**Current status: Phase 4 — core pipeline complete.** PDF extraction, RAG
(chunking/embeddings/FAISS), the LLM agent (Planner + Metadata + Paper
Analysis), and now Comparison, Research Gap Analysis, and Final Report
generation. Remaining work: filling in `tests/test_cases.md` with real
results, and a pass of UI/demo polish.

## Phase 4 Setup

No new dependencies beyond Phase 3, except `pandas` (added to
`requirements.txt`, used only for displaying the comparison table nicely).

```bash
pip install -r requirements.txt
```

### Test the full pipeline directly (recommended first)
```bash
python tests/test_agent.py paper1.pdf paper2.pdf paper3.pdf --query "Compare these papers and identify potential research gaps"
```
This is the exact kind of question from the FA brief's demonstration
flow. You should see: the plan, metadata, structured analysis per
paper, a comparison table with an AI-generated similarities/differences
narrative, potential research gaps tied to author-reported limitations,
and a full final report printed at the end.

### Run the Streamlit app
```bash
streamlit run app.py
```
Try the full demo flow:
1. Upload 3 papers on a similar topic
2. Ask: **"Compare these papers and identify potential research gaps."**
3. Watch the agent's plan appear (should include `compare_papers` and
   `identify_research_gaps`, not just basic analysis — this is the
   planner responding to what you actually asked)
4. Expand each paper's structured analysis
5. Check the comparison table + similarities/differences
6. Check the research gaps section — each one should be traceable back
   to a specific author-reported limitation
7. Expand and download the final report

Then try a **narrow** question like *"What dataset did paper 1 use?"*
and compare the plan — it should NOT include comparison/gap steps this
time. That contrast is your clearest planning demo for the viva.

## Phase 3 Setup

### 1. Install new dependencies
```bash
pip install -r requirements.txt
```
This adds `groq` (LLM client) and `pydantic` (structured schema).

### 2. Get a free Groq API key
1. Go to https://console.groq.com/keys
2. Sign up / log in (free)
3. Click "Create API Key", copy it

### 3. Set up your .env file
In the project root:
```bash
cp .env.example .env
```
Open `.env` and replace `your_groq_api_key_here` with the key you copied.
**Never commit this file** — it's already in `.gitignore`.

### 4. Test the agent directly (recommended first)
```bash
python tests/test_agent.py path/to/paper1.pdf path/to/paper2.pdf --query "Compare the datasets used"
```
This runs the full pipeline end to end and prints: the agent's plan,
metadata, and structured per-paper analysis. If something's going to
fail (bad key, rate limit, JSON parsing issue), you'll see it clearly
here before touching the UI.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```
Upload papers as before. Once the knowledge base is built, a new
**"Ask the Research Agent"** box appears — type a question and click
**Run Agent**. You'll see:
- The agent's plan (which steps it chose, and why)
- Metadata per paper
- Structured analysis per paper (dataset, methodology, results, etc.),
  each labeled clearly, with "Not explicitly reported in the provided
  paper." shown honestly wherever the LLM couldn't find something

Try asking different kinds of questions (e.g. "What dataset was used in
paper 1?" vs "Analyze all these papers") and compare the plans the agent
produces — this is the core evidence of genuine planning for your demo.

## Phase 2 Setup (RAG core)

If you already set up Phase 1, just install the new packages:

```bash
pip install -r requirements.txt
```

This adds `sentence-transformers`, `faiss-cpu`, and `numpy`.

**Heads up:** the first time you run the app (or the test script below),
`sentence-transformers` will download the embedding model (~80MB) from
the internet. This happens once and is cached afterward — just make
sure you have an internet connection for that first run.

### Test the full RAG pipeline directly (recommended first)
```bash
python tests/test_retriever.py path/to/paper1.pdf path/to/paper2.pdf --query "What dataset was used?"
```
This runs extract → chunk → embed → index → retrieve and prints the top
matching chunks with their similarity scores and source page numbers.

### Run the Streamlit app
```bash
streamlit run app.py
```
Upload your papers as before. Once processed, you'll now see a
**"Knowledge Base (RAG)"** section appear below — it builds automatically,
and includes a text box to test retrieval live: type a question and see
which chunks (with paper + page number) get retrieved. This box is
temporary — in Phase 3 it gets replaced by the full agent.

## Phase 1 Setup

### 1. Open the project in VS Code
Open the `ResearchAgent/` folder in VS Code (File → Open Folder).

### 2. Create a virtual environment
Open a terminal inside VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv .venv
```

Activate it:
- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

You should see `(.venv)` appear at the start of your terminal prompt.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Test the PDF extractor directly (recommended first)
Before touching Streamlit, confirm the extraction tool itself works:

```bash
python tests/test_pdf_extractor.py path/to/any/paper.pdf
```

You should see the status, page count, and a text preview printed.
Try this with:
- A normal text-based PDF (should say `status: ok`)
- A scanned/image-only PDF if you have one (should say `status: empty`, not crash)

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Upload 2-3 PDF research papers from the sidebar and confirm:
- Each shows ✅ with a text preview, OR
- Shows ⚠️/❌ with a clear explanation if something's wrong

## What's left
- Filling in real Pass/Fail results in `tests/test_cases.md` (TC02-TC06, TC08 especially)
- A pass of UI polish for the live demo (nice-to-have, not required)

## Project Structure
See `docs/architecture.md` for the full architecture explanation.
