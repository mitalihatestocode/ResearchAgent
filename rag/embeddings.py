"""
Embeddings
----------
Wraps a local embedding model (sentence-transformers) that turns text
into a vector (list of numbers) capturing its meaning. Similar-meaning
text produces similar vectors - this is what lets us do "search by
meaning" instead of just keyword matching.

WHY LOCAL, NOT AN API? (viva point)
- No API key needed, no cost, no rate limits
- Works offline once the model is downloaded (one-time, ~80MB)
- Keeps retrieval quality independent of which LLM API we later choose
  for generation (Groq/Gemini) - the two concerns are decoupled

Model used: all-MiniLM-L6-v2
- Small (fast to run on a laptop CPU, no GPU needed)
- Well-established, good quality-for-size tradeoff
- 384-dimensional vectors
"""

import threading

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # loaded lazily so importing this file doesn't trigger a slow model load
_lock = threading.Lock()  # the agent analyses papers in parallel threads


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading is slow-ish)."""
    global _model
    with _lock:
        if _model is None:
            _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]):
    """
    Turn a list of text strings into a list of embedding vectors.

    Args:
        texts: list of strings (e.g. chunk texts)

    Returns:
        A numpy array of shape (len(texts), 384)
    """
    model = get_model()
    # normalize_embeddings=True makes cosine similarity == dot product,
    # which is what our FAISS index (IndexFlatIP) expects.
    with _lock:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str):
    """Embed a single query string the same way chunks were embedded."""
    return embed_texts([query])[0]
