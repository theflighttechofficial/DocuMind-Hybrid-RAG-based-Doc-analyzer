from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.core.pipeline import get_bm25
from app.models.models import Chunk, User
from app.models.schemas import SearchRequest, SearchResultOut
from app.retrieval.hybrid import retrieve
from app.qa.generator import evidence_item

router = APIRouter(prefix="/api/documents", tags=["search"])

_VALID_METHODS = {"semantic", "bm25", "hybrid"}


@router.post("/{document_id}/search", response_model=list[SearchResultOut])
def search_document(
    document_id: str, payload: SearchRequest,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    doc = get_owned_document(document_id, db, current_user)
    if doc.processing_status != "ready":
        raise HTTPException(409, f"Document is not ready yet (status: {doc.processing_status}).")
    if payload.method not in _VALID_METHODS:
        raise HTTPException(400, f"method must be one of {sorted(_VALID_METHODS)}")

    chunks = db.query(Chunk).filter_by(document_id=document_id).order_by(Chunk.chunk_index).all()
    chunk_texts = [c.index_text for c in chunks]
    bm25 = get_bm25(document_id, chunk_texts)

    # Rerank only in hybrid mode — bm25/semantic are meant to show each method's
    # own raw ranking honestly, so the Search page can compare them side by side.
    results = retrieve(
        document_id, payload.query, chunks, bm25, method=payload.method, top_k=payload.top_k,
        rerank=(payload.method == "hybrid"),
    )

    return [SearchResultOut(**evidence_item(r["chunk"], r["score"])) for r in results]
