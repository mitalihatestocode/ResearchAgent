# ResearchAgent — Test Cases

Format: Test Case ID | Input | Agent Action | Expected Result | Actual Result | Pass/Fail

More test cases will be added as later phases are built. Phase 1 only
covers PDF extraction, so only TC01 and TC07 are relevant so far.

## TC01 — Upload one valid research paper
- **Input:** One well-formed PDF research paper
- **Agent action:** PDF Extraction Tool reads and extracts text
- **Expected result:** Status "ok", extracted text preview shown, page count matches PDF
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC07 — Missing information in paper (scanned PDF, no extractable text)
- **Input:** A scanned PDF with no text layer
- **Agent action:** PDF Extraction Tool attempts extraction
- **Expected result:** Status "empty", clear message explaining no extractable text was found, no crash
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC02 — Compare two papers
- **Input:** 2 uploaded papers, query "Compare these two papers"
- **Agent action:** Plan includes compare_papers (+ prerequisites); Comparison Tool builds table + narrative
- **Expected result:** Comparison table with both papers' columns; similarities/differences narrative labeled "AI-generated analysis"
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC03 — Compare three papers
- **Input:** 3 uploaded papers, query "Compare the methodologies used in these three papers"
- **Agent action:** Same as TC02 but with 3 papers
- **Expected result:** Table has 3 columns; narrative addresses all 3
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC04 — Dataset comparison
- **Input:** 2+ papers, query "Which datasets were used by each paper?"
- **Agent action:** Planner should favor a narrower plan (rag_retrieval and/or analyze_papers) over a full comparison, since only one parameter was asked about
- **Expected result:** Answer focuses on datasets specifically; plan is leaner than a full "compare everything" request
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC05 — Methodology comparison
- **Input:** 2+ papers, query "Compare the machine learning models and evaluation metrics"
- **Agent action:** Plan includes compare_papers; comparison table highlights models/metrics rows
- **Expected result:** Table and narrative are relevant to models/metrics specifically
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC06 — Research gap identification
- **Input:** 2+ papers with real reported limitations, query "Find potential research gaps based on these papers"
- **Agent action:** Plan includes identify_research_gaps (+ prerequisites); Research Gap Tool grounds suggestions in author-reported limitations
- **Expected result:** Each suggested gap is traceable to a specific limitation; suggestions are labeled "potential," never "confirmed" or "novel"
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC07 — Missing information in paper (scanned PDF, no extractable text)
- **Input:** A scanned PDF with no text layer
- **Agent action:** PDF Extraction Tool attempts extraction
- **Expected result:** Status "empty", clear message explaining no extractable text was found, no crash
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC08 — Irrelevant/out-of-scope user question
- **Input:** A question unrelated to the uploaded papers (e.g. "What's the weather today?")
- **Agent action:** Planner still produces a plan (likely rag_retrieval); retrieval finds nothing relevant
- **Expected result:** System responds honestly that no relevant content was found rather than fabricating an answer
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC09 — Agent planning varies by query type
- **Input:** Two different questions: "What dataset was used in paper 1?" vs "Analyze these papers"
- **Agent action:** Planner produces a plan for each
- **Expected result:** The two plans contain different steps (narrow question → fewer/targeted steps; broad question → metadata + full analysis)
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC10 — Missing/incomplete field in structured analysis
- **Input:** A paper where, e.g., evaluation metrics are never explicitly stated
- **Agent action:** Paper Analysis Tool extracts structured fields via LLM
- **Expected result:** The `evaluation_metrics` field shows "Not explicitly reported in the provided paper." rather than a guessed value
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

## TC11 — Missing GROQ_API_KEY
- **Input:** Run the app without setting up `.env`
- **Agent action:** App checks for the key before allowing agent queries
- **Expected result:** Clear warning message with setup instructions shown in the UI; app does not crash
- **Actual result:** _(fill in after testing)_
- **Pass/Fail:** _(fill in after testing)_

---
_All 11 test cases are now defined (8 required by the FA brief + 3 extra).
Run through each using the Streamlit app or tests/test_agent.py and fill
in Actual Result / Pass-Fail before submission._
