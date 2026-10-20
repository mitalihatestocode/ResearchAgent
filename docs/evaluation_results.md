# Evaluation results

Benchmark: `tests/benchmark.json`: 12 questions about the 3 sample papers with answers checked against the papers' text, plus 3 questions the agent should decline (not in the papers, or off-topic). Run with `python tests/evaluate.py`. The language model is not fully deterministic, so the answer part was run twice.

## Search quality (top 6 passages, no LLM)

| Search | All papers: hit@6 | Paper named in the question: hit@6 |
|---|---|---|
| dense | 0.67 | 0.83 |
| keyword | 0.58 | 0.92 |
| hybrid | 0.75 | 0.83 |

## Answers (real LLM, two runs)

| Measure | Run 1 | Run 2 |
|---|---|---|
| Fully correct answers | 10/12 | 8/12 |
| Average share of expected facts found | 90% | 77% |
| Answers whose citations all exist | 12/12 | 12/12 |
| Answers the fact-checker marked as supported | 12/12 | 12/12 |
| Correctly declined (not in the papers / off topic) | 3/3 | 3/3 |
| Average time per question | 20.7s | 17.8s |

| ID | Question | Facts found (run 1 / run 2) | Answer in run 1 (start) |
|---|---|---|---|
| Q01 | Which datasets does the Corrective RAG (CRAG) paper evaluate on? | 100% / 100% | The Corrective RAG (CRAG) paper evaluates on four datasets: PopQA, Biography, PubHealth, and Arc‑Challenge [S3][S4]. |
| Q02 | Which pretrained model is used to initialize the retrieval evaluator in CRAG? | 100% / 100% | The retrieval evaluator in CRAG is initialized with T5‑large (Raffel et al., 2020) [S2]. |
| Q03 | What are the knowledge retrieval actions that CRAG can trigger? | 100% / 100% | - CRAG can trigger three knowledge retrieval actions: **Correct**, **Incorrect**, and **Ambiguous**. The actions are defined based on confidence thresholds for |
| Q04 | How does CRAG extend retrieval beyond a static corpus when retrieved documents are poor? | 100% / 100% | - **corrective_rag_2024.pdf** uses large‑scale web searches as a strategic extension when the initially retrieved documents are irrelevant or of low quality, ge |
| Q05 | Which model is used as the generator in the original RAG paper by Lewis et al.? | 100% / 100% | The original RAG paper uses BART‑large as the generator component [S9]. |
| Q06 | What document collection does the Lewis et al. RAG model retrieve from, and how many documents does it contain? | 100% / 50% | The Lewis et al. RAG model retrieves from a dense vector index built over a Wikipedia dump (December 2018). The index contains 21 million 100‑word document chun |
| Q07 | What are the two variants of the RAG model proposed by Lewis et al.? | 100% / 100% | - **rag_lewis_2020.pdf** proposes two variants of the Retrieval-Augmented Generation (RAG) model: **RAG‑Token** and **RAG‑Sequence** [S1]. |
| Q08 | Which open-domain question answering datasets are used to evaluate the Lewis et al. RAG paper? | 100% / 100% | The Lewis et al. RAG paper evaluates on the following open‑domain QA datasets: Natural Questions (NQ), TriviaQA (TQA), WebQuestions (WQ), and CuratedTrec (CT) [ |
| Q09 | What types of reflection tokens does Self-RAG use? | 75% / 75% | - **self_rag_2023.pdf** defines four reflection token types: ISREL (relevance of retrieved passage), ISSUP (support level of passage for the output), ISUSE (use |
| Q10 | Which base language models are used as the Self-RAG generator? | 100% / 0% | - **self_rag_2023.pdf** uses Llama2 7B and Llama2 13B as the base generator language models [S12]. |
| Q11 | How many instruction-output pairs are used to train Self-RAG? | 100% / 100% | The authors use 150,000 instruction–output pairs to train SELF‑RAG [S5]. |
| Q12 | Which retriever model does Self-RAG use by default? | 0% / 0% | Not explicitly reported in the paper. |
| N01 | What was the total training cost of Self-RAG in US dollars? | 100% / 100% | Not explicitly reported in the paper. |
| N02 | How much carbon dioxide was emitted while training CRAG? | 100% / 100% | Not explicitly reported in the paper. |
| N03 | What is the capital of France? | 100% / 100% | That question doesn't seem to be about the uploaded papers, so I haven't answered it. Try asking about their datasets, methods, results, limitations or research |

Notes: every citation pointed to a real passage and every answer was judged supported by the fact-checker in both runs; all 3 unanswerable/off-topic questions were declined instead of answered. Q12 (and Q10 in run 2) are facts packed into one dense paragraph on page 7 of the Self-RAG paper that search did not reach; the agent then said "not explicitly reported" rather than guessing. Q09 named 3 of the 4 reflection-token types.
