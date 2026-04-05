"""
chunker.py
──────────
Converts the structured output of pdf_parser into a flat list of chunks
ready for embedding and storage in ChromaDB.

Each chunk carries:
  - text     : the content to embed
  - metadata : {paper_title, filename, section, chunk_index}

Strategy:
  - One chunk per section (Docling already gives us clean sections)
  - If a section is very long (> MAX_CHARS), split it further into
    overlapping windows so no single chunk overwhelms the LLM context
  - Tables are appended to the section they belong to (already inline
    in Docling's markdown output)
"""

MAX_CHARS = 1500      # max characters per chunk
OVERLAP = 150         # character overlap between split chunks


def build_chunks(parsed: dict, paper_title: str, filename: str) -> list:
    """
    Args:
        parsed      : output of pdf_parser.parse_pdf()
        paper_title : title string (from insight extraction or filename)
        filename    : original PDF filename

    Returns:
        List of dicts: [{text, metadata}]
    """
    chunks = []
    chunk_index = 0

    for section in parsed["sections"]:
        heading = section["heading"]
        content = section["content"]

        if not content.strip():
            continue

        # If short enough, keep as single chunk
        if len(content) <= MAX_CHARS:
            chunks.append({
                "text": f"[{heading}]\n{content}",
                "metadata": {
                    "paper_title": paper_title,
                    "filename": filename,
                    "section": heading,
                    "chunk_index": chunk_index,
                }
            })
            chunk_index += 1
        else:
            # Sliding window split
            sub_chunks = _sliding_window(content, MAX_CHARS, OVERLAP)
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "text": f"[{heading} — part {i+1}]\n{sub}",
                    "metadata": {
                        "paper_title": paper_title,
                        "filename": filename,
                        "section": heading,
                        "chunk_index": chunk_index,
                    }
                })
                chunk_index += 1

    return chunks


def _sliding_window(text: str, window: int, overlap: int) -> list:
    """Split text into overlapping windows of `window` characters."""
    parts = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start += window - overlap
    return parts
