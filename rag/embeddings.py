"""
Local, offline embeddings for the econ.TH RAG system.

Wraps the ``sentence-transformers`` model ``all-MiniLM-L6-v2`` (384-dimensional).
The model is loaded once (module-level singleton) and runs fully locally — no API
keys and no external network calls at inference time. The model is expected to be
already cached on disk.
"""
from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the SentenceTransformer model exactly once."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of texts into a list of 384-dim float vectors (local model)."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Convenience wrapper around embed_texts."""
    return embed_texts([text])[0]
