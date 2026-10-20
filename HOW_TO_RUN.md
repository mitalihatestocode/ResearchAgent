# How to run ResearchAgent

## What you need

- **Python 3.11 or newer.** Check with `python3 --version` (Windows: `py --version`).
  If it is older, install Python 3.12 from https://www.python.org/downloads/ (macOS can also use `brew install python@3.12`).
- **Internet** for the first install (it downloads the Python packages and a small embedding model, about 1-2 GB in total).
- **A free Groq API key** from https://console.groq.com/keys (sign in, click *Create API Key*, copy the key that starts with `gsk_`).

## Quick start (one command)

macOS / Linux, inside the `ResearchAgent` folder:
```bash
./run.sh
```
Windows (double-click or run in Command Prompt):
```bat
run.bat
```
The first run takes a few minutes; later runs start in seconds. The app opens at **http://localhost:8501**.

## Manual start (step by step)

### macOS / Linux

```bash
cd ResearchAgent                     # 1. go into the project folder
python3.12 -m venv .venv             # 2. create a virtual environment (use python3.11 if that is what you have)
source .venv/bin/activate            # 3. switch it on - your prompt now starts with (.venv)
pip install -r requirements.txt      # 4. install the packages (only needed the first time)
cp .env.example .env                 # 5. optional (skip if .env already exists): put your key after GROQ_API_KEY=
streamlit run app.py                 # 6. start the app
```

### Windows (Command Prompt)

```bat
cd ResearchAgent
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```
Skip the `copy` line if you already have a `.env` file (or paste the key in the app instead).
In PowerShell, if activating is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Next time

Only steps 1, 3 and 6 are needed:
```bash
cd ResearchAgent
source .venv/bin/activate            # Windows: .venv\Scripts\activate
streamlit run app.py
```
Stop the app with **Ctrl + C** in the terminal.

## Using the app

1. **Create an account** (your papers and chats are saved) or click **Continue as guest** (nothing is saved).
2. If you did not put the key in `.env`, **paste your Groq key** in the sidebar.
3. Click **Load 3 sample papers** (downloaded from arXiv), or upload your own PDFs (2-5 papers, up to 25 MB each).
4. Ask a question, or click one of the suggestions, for example:
   - *Compare these papers and identify potential research gaps.*
   - *How does the retrieval evaluator in CRAG decide what to do?*
   - *What limitations do the authors report?*
5. Look at the **Trace** (the agent's plan and steps), **Analysis**, **Galaxy** and **Report** tabs.

The first full analysis of 3 papers takes about 15-60 seconds (the free Groq tier limits speed); later questions are faster because the analyses are reused.

## Running the tests

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pytest -q                  # 30 tests, about 1 second, no API key needed
python tests/evaluate.py             # quality benchmark (needs the sample papers and GROQ_API_KEY in .env)
```

## If something goes wrong

| Problem | Fix |
|---|---|
| `ERROR: No matching distribution found for numpy==...` | Your Python is older than 3.11. Install Python 3.12 and recreate `.venv` with it. |
| "No Groq API key" | Paste the key in the sidebar, or put `GROQ_API_KEY=gsk_...` in `.env` and restart. |
| "Groq rejected the API key" | The key is wrong or deleted; create a new one at console.groq.com/keys. |
| Answers are slow or a step shows a rate-limit warning | The free tier allows about 8,000 tokens a minute; wait a minute and ask again. |
| "Port 8501 is already in use" | Another copy is running: stop it, or start with `streamlit run app.py --server.port 8502`. |
| A PDF shows "No text found" | It is a scanned image without a text layer; use a text-based PDF. |
| "Could not download the sample papers" | Check the internet connection, or download the PDFs from arXiv and upload them yourself. |
| Forgot a password / want a clean start | Stop the app and delete `data/app.db` (this removes all accounts and saved papers). |
