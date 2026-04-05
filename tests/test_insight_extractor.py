"""
test_insight_extractor.py
──────────────────────────
Tests for the Groq-based insight extractor.
Mocks the Groq API call to avoid hitting the real API in tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from paper_analyzer.extraction.insight_extractor import extract_insights, _build_insight
from paper_analyzer.extraction.schemas import PaperInsightData, INFO_NOT_FOUND


MOCK_PARSED = {
    "raw_markdown": "# Attention Is All You Need\n\nAuthors: Vaswani et al.\n\n## Abstract\nWe propose the Transformer...",
    "sections": [
        {"heading": "Abstract", "content": "We propose the Transformer architecture."},
        {"heading": "Methods", "content": "We use multi-head self-attention."},
    ],
    "tables": [],
    "formula_hints": ["\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V"],
    "page_count": 10,
}

MOCK_JSON_RESPONSE = """{
  "title": "Attention Is All You Need",
  "authors": "Ashish Vaswani, Noam Shazeer",
  "abstract": "We propose a new network architecture, the Transformer.",
  "problem_statement": "RNNs are slow due to sequential computation.",
  "methods": "Multi-head self-attention mechanism.",
  "math_formulas": [
    {"formula": "\\\\text{Attention}(Q,K,V) = \\\\text{softmax}(QK^T/\\\\sqrt{d_k})V", "meaning": "Scaled dot-product attention"}
  ],
  "datasets": "WMT 2014 English-German, WMT 2014 English-French",
  "results": "28.4 BLEU on EN-DE, 41.0 BLEU on EN-FR",
  "limitations": "Quadratic memory complexity with sequence length.",
  "future_work": "Apply to images, audio, and video.",
  "keywords": ["transformer", "attention", "NLP", "sequence modeling"],
  "year": "2017"
}"""


def test_build_insight_full():
    import json
    data = json.loads(MOCK_JSON_RESPONSE)
    insight = _build_insight(data)
    assert insight.title == "Attention Is All You Need"
    assert insight.year == "2017"
    assert len(insight.math_formulas) == 1
    assert insight.math_formulas[0].meaning == "Scaled dot-product attention"
    assert "transformer" in insight.keywords


def test_build_insight_missing_fields():
    insight = _build_insight({})
    assert insight.title == INFO_NOT_FOUND
    assert insight.abstract == INFO_NOT_FOUND
    assert insight.math_formulas == []
    assert insight.keywords == []


def test_build_insight_partial():
    insight = _build_insight({"title": "Some Paper", "year": "2023"})
    assert insight.title == "Some Paper"
    assert insight.year == "2023"
    assert insight.methods == INFO_NOT_FOUND


@pytest.mark.asyncio
async def test_extract_insights_success():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = MOCK_JSON_RESPONSE

    with patch(
        "paper_analyzer.extraction.insight_extractor._client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await extract_insights(MOCK_PARSED, "attention.pdf")

    assert isinstance(result, PaperInsightData)
    assert result.title == "Attention Is All You Need"
    assert result.year == "2017"


@pytest.mark.asyncio
async def test_extract_insights_json_error():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "NOT VALID JSON {{{"

    with patch(
        "paper_analyzer.extraction.insight_extractor._client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await extract_insights(MOCK_PARSED, "broken.pdf")

    # Should fall back gracefully
    assert isinstance(result, PaperInsightData)
    assert result.title == "broken.pdf"


@pytest.mark.asyncio
async def test_extract_insights_api_error():
    with patch(
        "paper_analyzer.extraction.insight_extractor._client.chat.completions.create",
        new_callable=AsyncMock,
        side_effect=Exception("API unavailable"),
    ):
        result = await extract_insights(MOCK_PARSED, "error.pdf")

    assert isinstance(result, PaperInsightData)
    assert result.title == "error.pdf"
