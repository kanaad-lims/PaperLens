"""
store.py
────────
ChromaDB wrapper for PaperLens.

Design decisions:
  - Embedded (in-process) ChromaDB — no server needed, perfect for
    a 3-paper corpus of ~30-50 chunks.
  - Each upload session gets its own collection keyed by session_id.
    This isolates users from each other and makes cleanup trivial.
  - Embeddings are computed externally (embedder.py) and passed in
    directly — ChromaDB accepts pre-computed embeddings.
  - Persistent storage in ./chroma_db/ so the vector store survives
    server restarts within a session.
"""

import chromadb
import os

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
_client = chromadb.PersistentClient(path=_PERSIST_DIR)


def get_or_create_collection(session_id: str):
    """Get or create a ChromaDB collection for this session."""
    # Collection names must be alphanumeric + hyphens, 3-63 chars
    collection_name = f"session-{session_id}"
    return _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(session_id: str, chunks: list, embeddings: list):
    """
    Insert chunks into the session collection.

    Args:
        session_id  : UUID string for this upload batch
        chunks      : list of {text, metadata} dicts from chunker.py
        embeddings  : list of float vectors, one per chunk
    """
    collection = get_or_create_collection(session_id)

    ids = [f"{session_id}-chunk-{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_collection(session_id: str, query_embedding: list, top_k: int = 5) -> dict:
    """
    Retrieve the top-k most relevant chunks for a query embedding.

    Returns ChromaDB query result dict with keys:
      documents, metadatas, distances
    """
    collection = get_or_create_collection(session_id)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return results


def delete_session(session_id: str):
    """Drop a session's collection — call on session end / cleanup."""
    collection_name = f"session-{session_id}"
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass
