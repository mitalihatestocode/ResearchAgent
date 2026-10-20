@echo off
rem One-step start for Windows: creates the virtual environment on first run, installs packages, opens the app.
cd /d "%~dp0"
if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
echo ResearchAgent is starting at http://localhost:8501
streamlit run app.py
