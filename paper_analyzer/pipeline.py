"""
pipeline.py
───────────
Orchestrates the full upload pipeline for a batch of PDFs.

Flow per paper (all 3 run concurrently via asyncio.gather):
  1. Docling parses the PDF → structured markdown + tables + formula hints
  2. Two tasks fire in parallel:
       a. Groq extracts structured insights (title, methods, formulas, etc.)
       b. Chunker splits sections → embedder encodes → ChromaDB stores chunks
  3. Results are collected and returned as a list of PaperInsight objects

Session isolation:
  Each batch gets a UUID session_id. ChromaDB uses it as the collection key.
  The frontend sends this session_id with every /chat request.
"""

import asyncio
import uuid

from paper_analyzer.ingestion.pdf_parser import parse_pdf
from paper_analyzer.ingestion.chunker import build_chunks
from paper_analyzer.extraction.insight_extractor import extract_insights
from paper_analyzer.vectorstore.embedder import embed_texts
from paper_analyzer.vectorstore.store import upsert_chunks
from app.schemas import PaperInsight, FormulaItem


async def _process_single_paper(
    pdf_path: str,
    filename: str,
    session_id: str,
) -> PaperInsight:
    """
    Full pipeline for one PDF:
      parse → (extract insights ∥ embed + store) → return PaperInsight
    """

    # Step 1: Parse (blocking IO — run in thread pool to not block event loop)
    loop = asyncio.get_event_loop()
    parsed = await loop.run_in_executor(None, parse_pdf, pdf_path)

    # Step 2a + 2b run concurrently
    insight_task = asyncio.create_task(extract_insights(parsed, filename))
    store_task   = asyncio.create_task(_embed_and_store(parsed, filename, session_id))

    insight_data, _ = await asyncio.gather(insight_task, store_task)

    # Map extraction schema → HTTP response schema
    return PaperInsight(
        filename=filename,
        title=insight_data.title,
        authors=insight_data.authors,
        abstract=insight_data.abstract,
        problem_statement=insight_data.problem_statement,
        methods=insight_data.methods,
        math_formulas=[
            FormulaItem(formula=f.formula, meaning=f.meaning)
            for f in insight_data.math_formulas
        ],
        datasets=insight_data.datasets,
        results=insight_data.results,
        limitations=insight_data.limitations,
        future_work=insight_data.future_work,
        keywords=insight_data.keywords,
        year=insight_data.year,
    )


async def _embed_and_store(parsed: dict, filename: str, session_id: str):
    """
    Chunk the parsed doc, embed all chunks, store in ChromaDB.
    Uses a placeholder title until extraction finishes — updated in metadata.
    """
    # Use filename as placeholder title for chunk metadata
    placeholder_title = filename.replace(".pdf", "")
    chunks = build_chunks(parsed, paper_title=placeholder_title, filename=filename)

    if not chunks:
        return

    texts = [c["text"] for c in chunks]

    # Embedding is CPU-bound — run in thread pool
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, embed_texts, texts)

    upsert_chunks(session_id, chunks, embeddings)


async def run_pipeline(
    pdf_paths: list[str],
    filenames: list[str],
) -> tuple[str, list[PaperInsight]]:
    """
    Entry point called by the /analyze route.

    Args:
        pdf_paths : list of absolute paths to uploaded PDFs (in temp dir)
        filenames : original filenames matching pdf_paths order

    Returns:
        (session_id, list[PaperInsight])
    """
    session_id = str(uuid.uuid4())

    # All papers processed concurrently
    tasks = [
        _process_single_paper(path, name, session_id)
        for path, name in zip(pdf_paths, filenames)
    ]

    insights = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out any papers that failed — log the error but don't crash
    clean_insights = []
    for i, result in enumerate(insights):
        if isinstance(result, Exception):
            print(f"[pipeline] Paper '{filenames[i]}' failed: {result}")
        else:
            clean_insights.append(result)

    return session_id, clean_insights
