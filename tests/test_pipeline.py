"""
test_pipeline.py
────────────────
Integration-level tests for the full pipeline.
Mocks both Docling and Groq so no real files or API calls are needed.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from paper_analyzer.extraction.schemas import PaperInsightData


MOCK_PARSED = {
    "raw_markdown": "# Test Paper\n## Abstract\nThis is a test.",
    "sections": [{"heading": "Abstract", "content": "This is a test."}],
    "tables": [],
    "formula_hints": [],
    "page_count": 2,
}

MOCK_INSIGHT = PaperInsightData(
    title="Test Paper",
    authors="Jane Doe",
    abstract="This is a test.",
    problem_statement="Testing the pipeline.",
    methods="Unit tests.",
    math_formulas=[],
    datasets="Info not found",
    results="All tests pass.",
    limitations="Info not found",
    future_work="Info not found",
    keywords=["testing", "pipeline"],
    year="2024",
)


@pytest.mark.asyncio
async def test_pipeline_returns_session_and_insights():
    with patch("paper_analyzer.pipeline.parse_pdf", return_value=MOCK_PARSED), \
         patch("paper_analyzer.pipeline.extract_insights", new_callable=AsyncMock, return_value=MOCK_INSIGHT), \
         patch("paper_analyzer.pipeline.embed_texts", return_value=[[0.1] * 384]), \
         patch("paper_analyzer.pipeline.upsert_chunks"):

        from paper_analyzer.pipeline import run_pipeline
        session_id, insights = await run_pipeline(
            pdf_paths=["/tmp/fake.pdf"],
            filenames=["fake.pdf"],
        )

    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID length
    assert len(insights) == 1
    assert insights[0].title == "Test Paper"


@pytest.mark.asyncio
async def test_pipeline_handles_single_paper_failure():
    with patch("paper_analyzer.pipeline.parse_pdf", side_effect=Exception("parse failed")), \
         patch("paper_analyzer.pipeline.extract_insights", new_callable=AsyncMock, return_value=MOCK_INSIGHT), \
         patch("paper_analyzer.pipeline.embed_texts", return_value=[]), \
         patch("paper_analyzer.pipeline.upsert_chunks"):

        from paper_analyzer.pipeline import run_pipeline
        session_id, insights = await run_pipeline(
            pdf_paths=["/tmp/bad.pdf"],
            filenames=["bad.pdf"],
        )

    # Failed paper should be filtered out, not crash the whole batch
    assert isinstance(session_id, str)
    assert len(insights) == 0


@pytest.mark.asyncio
async def test_pipeline_batch_of_three():
    with patch("paper_analyzer.pipeline.parse_pdf", return_value=MOCK_PARSED), \
         patch("paper_analyzer.pipeline.extract_insights", new_callable=AsyncMock, return_value=MOCK_INSIGHT), \
         patch("paper_analyzer.pipeline.embed_texts", return_value=[[0.1] * 384]), \
         patch("paper_analyzer.pipeline.upsert_chunks"):

        from paper_analyzer.pipeline import run_pipeline
        session_id, insights = await run_pipeline(
            pdf_paths=["/tmp/a.pdf", "/tmp/b.pdf", "/tmp/c.pdf"],
            filenames=["a.pdf", "b.pdf", "c.pdf"],
        )

    assert len(insights) == 3
