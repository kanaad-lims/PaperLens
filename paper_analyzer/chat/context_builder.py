"""
context_builder.py
───────────────────
Assembles the prompt for the conversational Q&A call.

Combines:
  1. Retrieved chunks from ChromaDB (the RAG context)
  2. Conversation history (last N turns for memory)
  3. The current user question

History is capped at MAX_HISTORY_TURNS to avoid overflowing the context
window and adding unnecessary latency.
"""

MAX_HISTORY_TURNS = 6   # = 6 user + 6 assistant messages = 12 total

SYSTEM_PROMPT = """You are PaperLens, an expert research assistant.
You answer questions about research papers that the user has uploaded.
Base your answers strictly on the provided context from the papers.
If the answer is not found in the context, say so clearly — do not hallucinate.
When relevant, mention which paper or section the information comes from.
Be concise, precise, and cite sources."""


def build_messages(
    question: str,
    retrieved_chunks: list[str],
    retrieved_metadatas: list[dict],
    history: list[dict],
) -> tuple[list[dict], list[str]]:
    """
    Build the messages list for the Groq chat call.

    Args:
        question            : current user question
        retrieved_chunks    : text strings from ChromaDB query
        retrieved_metadatas : metadata dicts from ChromaDB query
        history             : list of {role, content} dicts

    Returns:
        (messages, source_titles) tuple
          - messages      : list of {role, content} for Groq
          - source_titles : deduplicated list of paper titles used as context
    """

    # Build context block from retrieved chunks
    context_parts = []
    source_titles = []

    for chunk, meta in zip(retrieved_chunks, retrieved_metadatas):
        paper_title = meta.get("paper_title", "Unknown paper")
        section = meta.get("section", "")
        if paper_title not in source_titles:
            source_titles.append(paper_title)
        context_parts.append(f"[Source: {paper_title} | Section: {section}]\n{chunk}")

    context_block = "\n\n---\n\n".join(context_parts)

    # Cap history to last MAX_HISTORY_TURNS turns
    capped_history = history[-(MAX_HISTORY_TURNS * 2):]

    # Assemble messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject context as a system-level addendum
    if context_block:
        messages.append({
            "role": "system",
            "content": f"RELEVANT CONTEXT FROM UPLOADED PAPERS:\n\n{context_block}"
        })

    # Add conversation history
    for turn in capped_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    return messages, source_titles
