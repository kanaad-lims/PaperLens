"""
schemas.py  (extraction layer)
──────────────────────────────
Pydantic model for a fully extracted paper insight.
Every optional field defaults to "Info not found" so the frontend always
has a consistent structure to render, regardless of what the LLM returns.
"""

from pydantic import BaseModel, Field
from typing import List


INFO_NOT_FOUND = "Info not found"


class FormulaItem(BaseModel):
    formula: str = INFO_NOT_FOUND
    meaning: str = INFO_NOT_FOUND


class PaperInsightData(BaseModel):
    title: str = INFO_NOT_FOUND
    authors: str = INFO_NOT_FOUND
    abstract: str = INFO_NOT_FOUND
    problem_statement: str = INFO_NOT_FOUND
    methods: str = INFO_NOT_FOUND
    math_formulas: List[FormulaItem] = Field(default_factory=list)
    datasets: str = INFO_NOT_FOUND
    results: str = INFO_NOT_FOUND
    limitations: str = INFO_NOT_FOUND
    future_work: str = INFO_NOT_FOUND
    keywords: List[str] = Field(default_factory=list)
    year: str = INFO_NOT_FOUND
