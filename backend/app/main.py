from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.db import Base, engine
from app.core.migrations import run_startup_migrations
from app.models import models  # noqa: F401 ensures models are registered
from app.nlp.nltk_setup import ensure_nltk_data
from app.api import auth, documents, chat, search, analysis, summary, comparison, graph, evaluation, workspaces, settings as settings_api

Base.metadata.create_all(bind=engine)
run_startup_migrations()
ensure_nltk_data()

app = FastAPI(title="DocuMind AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(search.router)
app.include_router(analysis.router)
app.include_router(summary.router)
app.include_router(comparison.router)
app.include_router(graph.router)
app.include_router(evaluation.router)
app.include_router(workspaces.router)
app.include_router(settings_api.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
