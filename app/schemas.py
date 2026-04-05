from pydantic import BaseModel
from typing import List, Optional


# ── Analysis ──────────────────────────────────────────────────────────────────

class FormulaItem(BaseModel):
    formula: str
    meaning: str


class PaperInsight(BaseModel):
    filename: str
    title: str
    authors: str
    abstract: str
    problem_statement: str
    methods: str
    math_formulas: List[FormulaItem]
    datasets: str
    results: str
    limitations: str
    future_work: str
    keywords: List[str]
    year: str


class AnalyzeResponse(BaseModel):
    session_id: str
    papers: List[PaperInsight]


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]  # paper titles that were retrieved as context
