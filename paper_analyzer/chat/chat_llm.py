"""
chat_llm.py
───────────
Handles the conversational Q&A flow:
  1. Embed the user's question
  2. Retrieve top-k relevant chunks from ChromaDB
  3. Build the prompt (context + history + question)
  4. Call Groq and return the answer + source titles
"""

import os
from groq import AsyncGroq
from dotenv import load_dotenv

from paper_analyzer.vectorstore.embedder import embed_query
from paper_analyzer.vectorstore.store import query_collection
from paper_analyzer.chat.context_builder import build_messages

load_dotenv()

_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5          # chunks retrieved per question
MAX_TOKENS = 1024  # answer length cap


async def answer_question(
    session_id: str,
    question: str,
    history: list,
) -> tuple[str, list[str]]:
    """
    Answer a user question using RAG over the session's uploaded papers.

    Args:
        session_id : identifies which ChromaDB collection to query
        question   : the user's question string
        history    : list of {role, content} dicts (conversation so far)

    Returns:
        (answer_text, source_titles)
    """

    # 1. Embed the question
    query_embedding = embed_query(question)

    # 2. Retrieve relevant chunks
    results = query_collection(session_id, query_embedding, top_k=TOP_K)

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    retrieved_metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not retrieved_chunks:
        return (
            "I couldn't find relevant information in the uploaded papers to answer your question.",
            []
        )

    # 3. Build prompt
    history_dicts = [{"role": m.role, "content": m.content} for m in history]
    messages, source_titles = build_messages(
        question=question,
        retrieved_chunks=retrieved_chunks,
        retrieved_metadatas=retrieved_metadatas,
        history=history_dicts,
    )

    # 4. Call Groq
    try:
        response = await _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        answer = response.choices[0].message.content.strip()
        return answer, source_titles

    except Exception as e:
        print(f"[chat_llm] Groq call failed: {e}")
        return "Sorry, I ran into an error generating a response. Please try again.", []
