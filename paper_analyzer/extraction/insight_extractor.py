"""
insight_extractor.py
─────────────────────
Sends parsed paper content to Groq and extracts structured insights.

Design decisions:
  - Uses the full raw_markdown (not just sections) so the LLM has maximum
    context for extraction.
  - Formula hints from the PDF text layer are appended separately so the
    LLM can reconstruct proper LaTeX even from partial Unicode fragments.
  - Returns a validated PaperInsightData object. If parsing fails, all
    fields default to "Info not found" — never crashes the pipeline.
  - Async — runs concurrently across 3 papers via asyncio.gather().
"""

import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

from paper_analyzer.extraction.schemas import PaperInsightData, FormulaItem, INFO_NOT_FOUND

load_dotenv()

# Async Groq client — one instance, reused across all calls
_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 2048

# Cap markdown length sent to LLM to avoid context overflow (~12k chars ~ 3k tokens)
MAX_MARKDOWN_CHARS = 12000


_SYSTEM_PROMPT = """You are an expert research paper analyst.
You extract structured information from academic papers and return it as valid JSON.
Be precise and thorough. If a field cannot be found in the provided text, set its value to "Info not found".
For math_formulas: extract all significant equations. Represent formulas in LaTeX notation where possible.
Even if the formula text is fragmented or uses Unicode math symbols, reconstruct the LaTeX equivalent.
Output ONLY valid JSON — no markdown fences, no explanation, no preamble."""


_USER_PROMPT_TEMPLATE = """Extract structured insights from the following research paper.

--- PAPER CONTENT START ---
{markdown}
--- PAPER CONTENT END ---

Additional formula fragments detected from PDF layer (may be partial):
{formula_hints}

Return a JSON object with EXACTLY these fields:
{{
  "title": "Full paper title",
  "authors": "Comma-separated author names",
  "abstract": "Full abstract text",
  "problem_statement": "What specific problem or gap does this paper address?",
  "methods": "What methods, models, architectures, or algorithms are used?",
  "math_formulas": [
    {{"formula": "LaTeX string e.g. E = mc^2", "meaning": "Plain English explanation of what this formula represents"}}
  ],
  "datasets": "What datasets were used for experiments? If none, state Info not found.",
  "results": "Key quantitative results and findings",
  "limitations": "Limitations stated by the authors or apparent from the work",
  "future_work": "Future directions mentioned",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "year": "Publication year as a string, e.g. 2024"
}}"""


async def extract_insights(parsed: dict, filename: str) -> PaperInsightData:
    """
    Extract structured insights from a single parsed paper.

    Args:
        parsed   : output of pdf_parser.parse_pdf()
        filename : original PDF filename (used as fallback identifier)

    Returns:
        PaperInsightData with all fields populated (or defaulting to INFO_NOT_FOUND)
    """
    markdown = parsed["raw_markdown"][:MAX_MARKDOWN_CHARS]
    formula_hints = "\n".join(parsed.get("formula_hints", [])) or "None detected"

    prompt = _USER_PROMPT_TEMPLATE.format(
        markdown=markdown,
        formula_hints=formula_hints,
    )

    try:
        response = await _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,      # low temp for factual extraction
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences if model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return _build_insight(data)

    except json.JSONDecodeError as e:
        print(f"[extractor] JSON parse error for '{filename}': {e}")
        return PaperInsightData(title=filename)

    except Exception as e:
        print(f"[extractor] Groq call failed for '{filename}': {e}")
        return PaperInsightData(title=filename)


def _build_insight(data: dict) -> PaperInsightData:
    """Safely map raw JSON dict → validated PaperInsightData."""

    # Parse math_formulas list
    raw_formulas = data.get("math_formulas", [])
    formulas = []
    if isinstance(raw_formulas, list):
        for item in raw_formulas:
            if isinstance(item, dict):
                formulas.append(FormulaItem(
                    formula=item.get("formula", INFO_NOT_FOUND),
                    meaning=item.get("meaning", INFO_NOT_FOUND),
                ))

    # Parse keywords list
    raw_keywords = data.get("keywords", [])
    keywords = raw_keywords if isinstance(raw_keywords, list) else []

    return PaperInsightData(
        title=data.get("title", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        authors=data.get("authors", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        abstract=data.get("abstract", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        problem_statement=data.get("problem_statement", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        methods=data.get("methods", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        math_formulas=formulas,
        datasets=data.get("datasets", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        results=data.get("results", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        limitations=data.get("limitations", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        future_work=data.get("future_work", INFO_NOT_FOUND) or INFO_NOT_FOUND,
        keywords=keywords,
        year=str(data.get("year", INFO_NOT_FOUND)) or INFO_NOT_FOUND,
    )
