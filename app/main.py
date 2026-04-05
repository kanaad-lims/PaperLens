from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis_routes import router as analysis_router
from app.routes.chat_routes import router as chat_router

app = FastAPI(title="PaperLens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.include_router(analysis_router)
app.include_router(chat_router)


@app.get("/")
def home():
    return FileResponse("frontend/index.html")
