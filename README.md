# PaperLens

**AI-powered research paper analysis and Q&A engine.**

Upload up to 3 research papers (PDF) and get structured insights extracted by an LLM, plus a conversational interface to ask questions across your uploaded corpus.

---

## Features

- **Batch upload** — up to 3 PDFs per session
- **Structured extraction** — title, authors, abstract, problem statement, methods, math formulas (LaTeX), datasets, results, limitations, future work, keywords
- **Math rendering** — formulas rendered via KaTeX in the browser
- **RAG Q&A** — ask questions across all uploaded papers, grounded in retrieved context
- **Conversation memory** — last 6 turns kept in context per session
- **Session isolation** — each upload batch gets its own vector store collection

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| PDF extraction | Docling (OCR disabled) |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (embedded, persistent) |
| Frontend | Vanilla HTML/CSS/JS + KaTeX |

---

## Setup

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd PaperLens
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Docling downloads its ML models (~500MB) on first run. This is a one-time cost.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 4. Run

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

> Tests mock all external services (Docling, Groq, ChromaDB). No API key needed for tests.
> 
> To test PDF parsing, place a sample PDF at `tests/fixtures/sample.pdf`.

---

## Project Structure

```
PaperLens/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── schemas.py               # HTTP-level Pydantic models
│   └── routes/
│       ├── analysis_routes.py   # POST /analyze
│       └── chat_routes.py       # POST /chat
│
├── paper_analyzer/
│   ├── ingestion/
│   │   ├── pdf_parser.py        # Docling extraction
│   │   └── chunker.py           # Section-aware chunking
│   ├── extraction/
│   │   ├── insight_extractor.py # Groq LLM extraction
│   │   └── schemas.py           # PaperInsight model
│   ├── vectorstore/
│   │   ├── embedder.py          # MiniLM embeddings
│   │   └── store.py             # ChromaDB operations
│   └── chat/
│       ├── context_builder.py   # RAG prompt assembly
│       └── chat_llm.py          # Groq chat call
│   └── pipeline.py              # asyncio.gather orchestration
│
├── frontend/
│   └── index.html               # Full UI
│
└── tests/
    ├── test_pdf_parser.py
    ├── test_insight_extractor.py
    ├── test_pipeline.py
    └── test_chat.py
```

---

## API Endpoints

### `POST /analyze`
Upload PDFs and get structured insights.

**Request:** `multipart/form-data` with `files` (1–3 PDF files)

**Response:**
```json
{
  "session_id": "uuid",
  "papers": [
    {
      "filename": "paper.pdf",
      "title": "...",
      "authors": "...",
      "abstract": "...",
      "problem_statement": "...",
      "methods": "...",
      "math_formulas": [{"formula": "...", "meaning": "..."}],
      "datasets": "...",
      "results": "...",
      "limitations": "...",
      "future_work": "...",
      "keywords": ["..."],
      "year": "2024"
    }
  ]
}
```

### `POST /chat`
Ask a question about the uploaded papers.

**Request:**
```json
{
  "session_id": "uuid",
  "question": "What methods are common across the papers?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": ["Paper Title A", "Paper Title B"]
}
```

---

## Design Notes

- **Parallel processing:** All 3 papers are processed concurrently via `asyncio.gather()`. Total latency ≈ slowest single paper, not the sum.
- **Dual parallelism per paper:** Groq extraction and ChromaDB population run simultaneously for each paper.
- **3-paper limit:** Enforced at the route level with a clear error message.
- **"Info not found":** Every extraction field defaults to this string so the UI always has consistent data to render.
- **Docling OCR disabled:** Assumes digitally typeset PDFs (standard for arXiv). Enable `do_ocr=True` in `pdf_parser.py` for scanned documents (slower).
