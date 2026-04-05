"""
embedder.py
───────────
Thin wrapper around sentence-transformers MiniLM-L6-v2.

Initialised once at module level — model load costs ~0.5s.
Reused across all embed calls for the session.
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loaded once, shared across all requests
_embed_model = SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of float vectors (one per text).
    """
    if not texts:
        return []
    embeddings = _embed_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
