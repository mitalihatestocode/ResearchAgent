"""
BM25 Keyword Index
------------------
A small, dependency-free implementation of Okapi BM25 - the classic
keyword-ranking function used by search engines.

WHY ADD KEYWORD SEARCH WHEN WE ALREADY HAVE EMBEDDINGS? (viva point)
Dense embeddings are good at meaning ("dataset" ~ "corpus") but weak at
exact tokens: model names (BERT, ResNet-50), dataset names (MIMIC-III,
SQuAD), metric names (F1, BLEU) and numbers. Research questions are full
of these. BM25 is strong exactly where embeddings are weak, so combining
both ("hybrid retrieval") gives noticeably better evidence.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

STOPWORDS = set("""
a an the and or of to in on for with by from at as is are was were be been being this that these those
it its into than then there their they them we our you your he she his her i me my what which who whom
how why when where do does did done can could should would may might will shall not no yes so such
paper papers study studies used use using also about over under between each both all any more most
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()

    def add(self, texts: list[str]):
        for text in texts:
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(tokens))
            self.df.update(freqs.keys())

    def __len__(self):
        return len(self.doc_freqs)

    def scores(self, query: str) -> list[float]:
        """BM25 score of the query against every indexed document."""
        n = len(self.doc_freqs)
        if n == 0:
            return []
        avg_len = (sum(self.doc_lens) / n) or 1.0
        query_terms = set(tokenize(query))
        results = []
        for freqs, doc_len in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len))
            results.append(score)
        return results
