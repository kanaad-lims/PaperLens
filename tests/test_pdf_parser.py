"""
test_pdf_parser.py
──────────────────
Tests for the Docling-based PDF parser.
Requires a sample PDF at tests/fixtures/sample.pdf
"""

import os
import pytest
from paper_analyzer.ingestion.pdf_parser import parse_pdf, _looks_like_formula, _split_into_sections


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_PDF = os.path.join(FIXTURES_DIR, "sample.pdf")


def test_looks_like_formula_with_math_chars():
    assert _looks_like_formula("∑ x_i = 0") is True


def test_looks_like_formula_with_latex():
    assert _looks_like_formula("\\nabla J(\\theta)") is True


def test_looks_like_formula_plain_text():
    assert _looks_like_formula("This is a normal sentence without math.") is False


def test_looks_like_formula_too_long():
    long_text = "x " * 150
    assert _looks_like_formula(long_text) is False


def test_split_into_sections_basic():
    markdown = """# Abstract
This paper presents a novel approach.

## Introduction
We study the problem of optimization.

## Methods
We use gradient descent.
"""
    sections = _split_into_sections(markdown)
    headings = [s["heading"] for s in sections]
    assert "Abstract" in headings
    assert "Introduction" in headings
    assert "Methods" in headings


def test_split_into_sections_content():
    markdown = "# Abstract\nThis is the abstract.\n## Methods\nThis is the method."
    sections = _split_into_sections(markdown)
    abstract = next(s for s in sections if s["heading"] == "Abstract")
    assert "abstract" in abstract["content"].lower()


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="No sample PDF fixture found")
def test_parse_pdf_returns_structure():
    result = parse_pdf(SAMPLE_PDF)
    assert "raw_markdown" in result
    assert "sections" in result
    assert "tables" in result
    assert "formula_hints" in result
    assert "page_count" in result
    assert isinstance(result["sections"], list)
    assert result["page_count"] > 0


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="No sample PDF fixture found")
def test_parse_pdf_has_content():
    result = parse_pdf(SAMPLE_PDF)
    assert len(result["raw_markdown"]) > 100
    assert len(result["sections"]) > 0
