"""
pdf_parser.py
─────────────
Uses Docling to extract structured content from a research paper PDF.

Docling gives us:
  - Proper reading order (handles two-column layouts)
  - Table structure preserved as markdown
  - Inline formula text from the PDF text layer
  - Section-level grouping via heading detection

OCR is disabled — we assume digitally typeset PDFs (standard for arXiv papers).
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import ConversionStatus
import re

# ── Initialise once at module level — Docling loads ML models at init time.
# Re-creating this per request would cost ~2s on every call.
_pipeline_options = PdfPipelineOptions(
    do_ocr=False,                       # disabled — papers are digitally typeset
    do_table_structure=True,            # core requirement
    do_formula_enrichment=False,        # skip — needs extra model, Groq handles interpretation
    do_picture_classification=False,    # not needed for RAG
)
_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
_pipeline_options.table_structure_options.do_cell_matching = True

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

# Section headings commonly found in research papers
_SECTION_PATTERNS = re.compile(
    r"^(abstract|introduction|related work|background|methodology|methods?|"
    r"approach|experiments?|results?|evaluation|discussion|conclusion|"
    r"future work|references|appendix|acknowledgements?)",
    re.IGNORECASE
)


def parse_pdf(pdf_path: str) -> dict:
    """
    Parse a single PDF and return a structured dict:
    {
        "raw_markdown": str,          # full document as markdown
        "sections": [
            {
                "heading": str,
                "content": str        # text + inline tables for this section
            }
        ],
        "tables": [str],              # all tables as markdown strings
        "formula_hints": [str],       # raw formula text fragments from PDF layer
        "page_count": int
    }
    """

    result = _converter.convert(pdf_path)

    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        raise RuntimeError(f"Docling failed to convert '{pdf_path}': {result.status}")

    doc = result.document

    # ── Full markdown export ───────────────────────────────────────────────────
    raw_markdown = doc.export_to_markdown()

    # ── Tables ────────────────────────────────────────────────────────────────
    tables = []
    for table in doc.tables:
        try:
            md = table.export_to_markdown()
            if md.strip():
                tables.append(md.strip())
        except Exception:
            pass

    # ── Formula hints (text-layer fragments) ──────────────────────────────────
    formula_hints = []
    for item in doc.texts:
        text = item.text.strip()
        # Heuristic: lines with math-like characters and short length
        if text and _looks_like_formula(text):
            formula_hints.append(text)

    # ── Section chunking ──────────────────────────────────────────────────────
    sections = _split_into_sections(raw_markdown)

    return {
        "raw_markdown": raw_markdown,
        "sections": sections,
        "tables": tables,
        "formula_hints": formula_hints[:30],   # cap at 30 to keep prompt size sane
        "page_count": len(doc.pages),
    }


def _looks_like_formula(text: str) -> bool:
    """Heuristic to detect formula fragments from the PDF text layer."""
    if len(text) > 200:
        return False
    math_chars = set(r"∑∫∂∇αβγδεζηθλμνξπρστφχψω×÷±≤≥≠≈∞√∈∉⊂⊃∪∩")
    has_math_chars = any(c in math_chars for c in text)
    has_latex_hint = any(p in text for p in ["\\", "^{", "_{", "frac", "sum", "int"])
    return has_math_chars or has_latex_hint


def _split_into_sections(markdown: str) -> list:
    """
    Split the full markdown into sections based on heading lines.
    Returns a list of {heading, content} dicts.
    """
    lines = markdown.split("\n")
    sections = []
    current_heading = "Preamble"
    current_lines = []

    for line in lines:
        # Markdown headings: # Introduction, ## Methods, etc.
        heading_match = re.match(r"^#{1,3}\s+(.+)$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            # Save previous section
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines).strip()
                })
            current_heading = heading_text
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_lines).strip()
        })

    # Filter out empty sections
    sections = [s for s in sections if s["content"]]
    return sections
