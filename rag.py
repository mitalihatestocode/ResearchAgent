import math
import re
import threading
from collections import Counter

import faiss
import numpy as np
import pdfplumber

MAX_PAGES = 80
CHUNK_SIZE, OVERLAP = 900, 150
MIN_SCORE = 0.28

SECTIONS = {
    "abstract": "Abstract", "introduction": "Introduction", "background": "Background",
    "related work": "Related Work", "method": "Method", "methods": "Method", "methodology": "Method",
    "approach": "Method", "proposed method": "Method", "data": "Data", "dataset": "Data", "datasets": "Data",
    "experiments": "Experiments", "experimental setup": "Experiments", "evaluation": "Evaluation",
    "results": "Results", "discussion": "Discussion", "limitations": "Limitations", "conclusion": "Conclusion",
    "conclusions": "Conclusion", "future work": "Future Work", "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements", "references": "References", "bibliography": "References",
}
HEADING = re.compile(r"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?([A-Za-z][A-Za-z &]{2,40}?)\s*:?\s*$")


def page_text(page):
    words = page.extract_words(x_tolerance=1.5)
    mid = page.width / 2
    crossing = [w for w in words if w["x0"] < mid - 2 and w["x1"] > mid + 2]
    left = sum(w["x1"] <= mid for w in words)
    right = len(words) - left - len(crossing)
    if len(words) < 80 or len(crossing) > 0.03 * len(words) or min(left, right) < 0.2 * len(words):
        return page.extract_text(x_tolerance=1.5) or ""
    top = max((w["bottom"] for w in crossing if w["top"] < page.height * 0.4), default=0)
    boxes = [(0, 0, page.width, top), (0, top, mid, page.height), (mid, top, page.width, page.height)]
    return "\n".join(page.crop(b).extract_text(x_tolerance=1.5) or "" for b in boxes if b[3] > b[1])


def clean(text):
    for glyph, letters in {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}.items():
        text = text.replace(glyph, letters)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text).strip()


def read_pdf(source, name):
    try:
        with pdfplumber.open(source) as pdf:
            total = len(pdf.pages)
            pages = [clean(page_text(p)) for p in pdf.pages[:MAX_PAGES]]
    except Exception as e:
        return {"name": name, "status": "error", "message": f"Could not read this PDF ({e}).", "pages": []}
    if not any(pages):
        return {"name": name, "status": "empty", "message": "No text found - probably a scanned PDF (needs OCR).",
                "pages": []}
    note = f" (first {MAX_PAGES} of {total})" if total > MAX_PAGES else ""
    return {"name": name, "status": "ok", "message": f"{len(pages)} pages{note}", "pages": pages}


def section_of(line):
    match = HEADING.match(line) if len(line) <= 60 else None
    return SECTIONS.get(match.group(1).strip().lower()) if match else None


def sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", " ".join(text.split()))


def pack(sents):
    chunks, current = [], []
    for s in sents:
        while len(s) > CHUNK_SIZE:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(s[:CHUNK_SIZE])
            s = s[CHUNK_SIZE - OVERLAP:]
        if current and len(" ".join(current + [s])) > CHUNK_SIZE:
            chunks.append(" ".join(current))
            keep = []
            for prev in reversed(current):
                if len(" ".join([prev] + keep)) > OVERLAP:
                    break
                keep.insert(0, prev)
            current = keep
        current.append(s)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) >= 40]


def chunk(paper):
    chunks, section = [], "Front Matter"
    for page_no, text in enumerate(paper["pages"], 1):
        blocks, lines = [], []
        for line in text.splitlines():
            heading = section_of(line)
            if heading:
                blocks.append((section, " ".join(lines)))
                lines, section = [], heading
            else:
                lines.append(line)
        blocks.append((section, " ".join(lines)))
        n = 0
        for sec, block in blocks:
            if sec == "References":
                continue
            for piece in pack(sentences(block)):
                chunks.append({"id": f"{paper['name']}_p{page_no}_c{n}", "paper": paper["name"],
                               "page": page_no, "section": sec, "text": piece})
                n += 1
    return chunks


_model, _lock = None, threading.Lock()


def embed(texts):
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(_model.encode(list(texts), normalize_embeddings=True), dtype="float32")


STOPWORDS = set("""a an the and or of to in on for with by from at as is are was were be been this that these those it
its into than then there their they we our you your what which who how why when where do does did can could should would
may might will not no so such paper papers study used use using also about each all any more most""".split())


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


class Index:

    def __init__(self):
        self.vectors = faiss.IndexFlatIP(384)
        self.chunks, self.term_counts, self.doc_freq = [], [], Counter()

    def add(self, chunks):
        if not chunks:
            return np.zeros((0, 384), dtype="float32")
        vectors = embed(c["text"] for c in chunks)
        self.vectors.add(vectors)
        self.chunks += chunks
        for c in chunks:
            counts = Counter(tokens(c["text"]))
            self.term_counts.append(counts)
            self.doc_freq.update(counts.keys())
        return vectors

    def bm25(self, query, k1=1.5, b=0.75):
        n = len(self.chunks)
        avg_len = sum(sum(c.values()) for c in self.term_counts) / n or 1
        scores = []
        for counts in self.term_counts:
            length, score = sum(counts.values()), 0.0
            for term in set(tokens(query)):
                if counts[term]:
                    idf = math.log(1 + (n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                    score += idf * counts[term] * (k1 + 1) / (counts[term] + k1 * (1 - b + b * length / avg_len))
            scores.append(score)
        return scores

    def search(self, query, k=6, papers=None, mode="hybrid", min_score=None):
        n = len(self.chunks)
        if not n:
            return []
        sims, ids = self.vectors.search(embed([query]), n)
        cosine = dict(zip(ids[0].tolist(), sims[0].tolist()))
        keyword = self.bm25(query)
        pool = [i for i in range(n) if not papers or self.chunks[i]["paper"] in papers]
        dense_rank = {i: r for r, i in enumerate(sorted(pool, key=lambda i: -cosine[i]), 1)}
        word_rank = {i: r for r, i in enumerate(sorted(pool, key=lambda i: -keyword[i]), 1)}

        def rank_score(i):
            if mode == "dense":
                return cosine[i]
            if mode == "keyword":
                return keyword[i]
            return 1 / (60 + dense_rank[i]) + (1 / (60 + word_rank[i]) if keyword[i] > 0 else 0)

        best = sorted(pool, key=rank_score, reverse=True)[:k]
        hits = [dict(self.chunks[i], score=round(cosine[i], 3), bm25=round(keyword[i], 2)) for i in best]
        return [h for h in hits if min_score is None or h["score"] >= min_score]


def project_2d(vectors):
    mean = vectors.mean(axis=0)
    _, strengths, directions = np.linalg.svd(vectors - mean, full_matrices=False)
    axes = directions[:2].T
    kept = float((strengths[:2] ** 2).sum() / (strengths ** 2).sum())
    return (vectors - mean) @ axes, (lambda v: (np.asarray(v) - mean) @ axes), kept
