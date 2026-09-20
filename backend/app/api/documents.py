import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.config import get_settings
from app.core.deps import get_current_user, get_owned_document
from app.core.pipeline import process_document, invalidate_bm25
from app.models.models import Document, DocumentPage, User
from app.models.schemas import DocumentOut, DocumentDetailOut, PageOut
from app.retrieval import vector_store

router = APIRouter(prefix="/api/documents", tags=["documents"])
settings = get_settings()

_ALLOWED_TYPES = {"pdf", "docx", "txt"}
os.makedirs(settings.upload_dir, exist_ok=True)


@router.post("", response_model=DocumentOut)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "").lower()
    if ext not in _ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported file type. Please upload PDF, DOCX, or TXT.")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}.{ext}"
    stored_path = os.path.join(settings.upload_dir, stored_name)

    with open(stored_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    file_size = os.path.getsize(stored_path)

    document = Document(
        id=doc_id,
        user_id=current_user.id,
        name=file.filename,
        file_type=ext,
        file_size=file_size,
        storage_path=stored_path,
        processing_status="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    def _run():
        from app.core.db import SessionLocal
        session = SessionLocal()
        try:
            doc = session.get(Document, doc_id)
            process_document(session, doc)
        finally:
            session.close()

    background_tasks.add_task(_run)
    return document


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return (
        db.query(Document)
        .filter_by(user_id=current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )


@router.get("/{document_id}", response_model=DocumentDetailOut)
def get_document(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_owned_document(document_id, db, current_user)


@router.get("/{document_id}/pages", response_model=list[PageOut])
def get_pages(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    get_owned_document(document_id, db, current_user)
    return db.query(DocumentPage).filter_by(document_id=document_id).order_by(DocumentPage.page_number).all()


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = get_owned_document(document_id, db, current_user)
    if doc.storage_path and os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)
    vector_store.delete_index(document_id)
    invalidate_bm25(document_id)
    db.delete(doc)
    db.commit()
    return {"status": "deleted"}
