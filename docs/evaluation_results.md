# Evaluation Results

_Run 2026-10-14 21:48 · benchmark `ResearchAgent RAG benchmark v1` · 12 answerable + 3 unanswerable questions · papers: the 3 sample arXiv papers_

## Retrieval quality (top-6, no LLM)

| Search mode | Hit@k | MRR | Context recall |
|---|---|---|---|
| dense (unscoped) | 0.67 | 0.47 | 0.62 |
| keyword (unscoped) | 0.67 | 0.37 | 0.60 |
| hybrid (unscoped) | 0.75 | 0.42 | 0.62 |
| dense (scoped) | 0.83 | 0.52 | 0.79 |
| keyword (scoped) | 0.92 | 0.51 | 0.77 |
| hybrid (scoped) | 0.83 | 0.54 | 0.79 |

_Hit@k: a chunk from the right paper containing an expected fact is in the top k. MRR: mean reciprocal rank of the first such chunk. Context recall: share of expected facts present in the retrieved chunks._

## End-to-end answer quality (real LLM)

_Before the answerability-driven second search was added, the same benchmark scored 67% answer accuracy (8/12) and 77% fact recall. After: see below._

| Metric | Value |
|---|---|
| Answer accuracy (all expected facts present) | 75% |
| Mean fact recall | 81% |
| Citation validity (code check) | 100% |
| Faithfulness (verifier: supported) | 100% |
| Correct refusals on unanswerable / out-of-scope | 100% |
| Mean latency | 19.0s |
| Mean LLM calls / tokens per question | 3.9 / 6958 |

### Per question

| ID | Question | Exact | Fact recall | Citations ok | Faithful | Plan |
|---|---|---|---|---|---|---|
| Q01 | Which datasets does the Corrective RAG (CRAG) paper evaluate on? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q02 | Which pretrained model is used to initialize the retrieval evaluator in CRAG? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q03 | What are the knowledge retrieval actions that CRAG can trigger? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q04 | How does CRAG extend retrieval beyond a static corpus when retrieved documents are poor? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q05 | Which model is used as the generator in the original RAG paper by Lewis et al.? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q06 | What document collection does the Lewis et al. RAG model retrieve from, and how many documents does it contain? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q07 | What are the two variants of the RAG model proposed by Lewis et al.? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q08 | Which open-domain question answering datasets are used to evaluate the Lewis et al. RAG paper? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q09 | What types of reflection tokens does Self-RAG use? | ❌ | 75% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q10 | Which base language models are used as the Self-RAG generator? | ❌ | 0% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q11 | How many instruction-output pairs are used to train Self-RAG? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| Q12 | Which retriever model does Self-RAG use by default? | ❌ | 0% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| N01 | What was the total training cost of Self-RAG in US dollars? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| N02 | How much carbon dioxide was emitted while training CRAG? | ✅ | 100% | ✅ | ✅ | rag_retrieval → answer_question → verify_answer |
| N03 | What is the capital of France? | ✅ | 100% | – | – | — |
