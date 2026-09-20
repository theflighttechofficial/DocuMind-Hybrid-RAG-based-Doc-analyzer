from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.models.models import DocumentPage, User
from app.models.schemas import SummaryRequest
from app.nlp.summarization import summarize_document

router = APIRouter(prefix="/api/documents", tags=["summary"])

_VALID_LENGTHS = {"brief", "standard", "detailed"}


@router.post("/{document_id}/summarize")
def summarize(
    document_id: str, payload: SummaryRequest,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    doc = get_owned_document(document_id, db, current_user)
    if doc.processing_status != "ready":
        raise HTTPException(409, f"Document is not ready yet (status: {doc.processing_status}).")
    if payload.length not in _VALID_LENGTHS:
        raise HTTPException(400, f"length must be one of {sorted(_VALID_LENGTHS)}")

    # cache per length so repeated requests don't re-call the LLM
    cache = doc.summary_json or {}
    if payload.length in cache:
        return cache[payload.length]

    pages = db.query(DocumentPage).filter_by(document_id=document_id).order_by(DocumentPage.page_number).all()
    full_text = "\n\n".join(p.text for p in pages)

    result = summarize_document(full_text, length=payload.length)

    if result.get("available"):
        cache[payload.length] = result
        doc.summary_json = cache
        db.commit()

    return result
