from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import tempfile, os, shutil

from app.schemas import AnalyzeResponse
from paper_analyzer.pipeline import run_pipeline

router = APIRouter()

MAX_PAPERS = 3


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_papers(files: List[UploadFile] = File(...)):

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > MAX_PAPERS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_PAPERS} papers per batch. You uploaded {len(files)}."
        )

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{f.filename}' is not a PDF. Only PDF files are accepted."
            )

    # Save uploads to a temp dir — Docling needs file paths, not byte streams
    tmp_dir = tempfile.mkdtemp()
    try:
        pdf_paths = []
        original_names = []
        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(upload.file, out)
            pdf_paths.append(dest)
            original_names.append(upload.filename)

        session_id, insights = await run_pipeline(pdf_paths, original_names)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return AnalyzeResponse(session_id=session_id, papers=insights)
