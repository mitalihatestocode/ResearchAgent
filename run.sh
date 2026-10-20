#!/bin/sh
# One-step start for macOS/Linux: creates the virtual environment on first run, installs packages, opens the app.
cd "$(dirname "$0")"
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3)
[ -d .venv ] || "$PYTHON" -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
echo "ResearchAgent is starting at http://localhost:8501"
streamlit run app.py
