"""
test_chat.py
────────────
Tests for the conversational Q&A layer.
Mocks ChromaDB and Groq — no real services needed.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from paper_analyzer.chat.context_builder import build_messages, MAX_HISTORY_TURNS


# ── context_builder tests ─────────────────────────────────────────────────────

def test_build_messages_basic():
    messages, sources = build_messages(
        question="What method is used?",
        retrieved_chunks=["[Methods]\nWe use gradient descent."],
        retrieved_metadatas=[{"paper_title": "Deep Learning Paper", "section": "Methods"}],
        history=[],
    )
    assert any(m["role"] == "user" for m in messages)
    assert "Deep Learning Paper" in sources


def test_build_messages_deduplicates_sources():
    messages, sources = build_messages(
        question="Summarise the methods.",
        retrieved_chunks=["chunk 1", "chunk 2"],
        retrieved_metadatas=[
            {"paper_title": "Paper A", "section": "Methods"},
            {"paper_title": "Paper A", "section": "Results"},
        ],
        history=[],
    )
    assert sources.count("Paper A") == 1


def test_build_messages_caps_history():
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(30)
    ]
    messages, _ = build_messages(
        question="Final question",
        retrieved_chunks=["some context"],
        retrieved_metadatas=[{"paper_title": "P", "section": "S"}],
        history=long_history,
    )
    # Count non-system messages (history + user question)
    non_system = [m for m in messages if m["role"] != "system"]
    # Should be at most MAX_HISTORY_TURNS * 2 history + 1 current question
    assert len(non_system) <= MAX_HISTORY_TURNS * 2 + 1


def test_build_messages_no_context():
    messages, sources = build_messages(
        question="What is this?",
        retrieved_chunks=[],
        retrieved_metadatas=[],
        history=[],
    )
    assert sources == []
    assert any(m["role"] == "user" for m in messages)


# ── chat_llm tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_answer_question_success():
    mock_chroma_result = {
        "documents": [["[Methods]\nWe use attention mechanism."]],
        "metadatas": [[{"paper_title": "Transformer Paper", "section": "Methods"}]],
        "distances": [[0.1]],
    }
    mock_groq_response = MagicMock()
    mock_groq_response.choices[0].message.content = "The paper uses attention mechanisms."

    with patch("paper_analyzer.chat.chat_llm.embed_query", return_value=[0.1] * 384), \
         patch("paper_analyzer.chat.chat_llm.query_collection", return_value=mock_chroma_result), \
         patch(
             "paper_analyzer.chat.chat_llm._client.chat.completions.create",
             new_callable=AsyncMock,
             return_value=mock_groq_response,
         ):
        from paper_analyzer.chat.chat_llm import answer_question
        answer, sources = await answer_question("session-123", "What method?", [])

    assert "attention" in answer.lower()
    assert "Transformer Paper" in sources


@pytest.mark.asyncio
async def test_answer_question_no_chunks():
    mock_chroma_result = {"documents": [], "metadatas": [], "distances": []}

    with patch("paper_analyzer.chat.chat_llm.embed_query", return_value=[0.1] * 384), \
         patch("paper_analyzer.chat.chat_llm.query_collection", return_value=mock_chroma_result):
        from paper_analyzer.chat.chat_llm import answer_question
        answer, sources = await answer_question("session-empty", "What method?", [])

    assert "couldn't find" in answer.lower()
    assert sources == []
