from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.models.models import Document, User

router = APIRouter(prefix="/api/documents", tags=["analysis"])


def _get_ready_doc(document_id: str, db: Session, current_user: User) -> Document:
    doc = get_owned_document(document_id, db, current_user)
    if doc.processing_status != "ready":
        raise HTTPException(409, f"Document is not ready yet (status: {doc.processing_status}).")
    return doc


@router.get("/{document_id}/entities")
def get_entities(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = _get_ready_doc(document_id, db, current_user)
    return doc.entities_json or {}


@router.get("/{document_id}/keywords")
def get_keywords(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = _get_ready_doc(document_id, db, current_user)
    return doc.keywords_json or {}


@router.get("/{document_id}/pos")
def get_pos(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = _get_ready_doc(document_id, db, current_user)
    return doc.pos_json or {}


@router.get("/{document_id}/statistics")
def get_statistics(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = _get_ready_doc(document_id, db, current_user)
    return doc.stats_json or {}


@router.get("/{document_id}/toc")
def get_toc(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = _get_ready_doc(document_id, db, current_user)
    return doc.toc_json or []
