from fastapi import APIRouter, HTTPException
from app.schemas import ChatRequest, ChatResponse
from paper_analyzer.chat.chat_llm import answer_question

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, sources = await answer_question(
        session_id=request.session_id,
        question=request.question,
        history=request.history,
    )

    return ChatResponse(answer=answer, sources=sources)
