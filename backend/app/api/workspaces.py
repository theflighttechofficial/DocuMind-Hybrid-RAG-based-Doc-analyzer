from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.core.pipeline import get_bm25
from app.core.config import get_settings
from app.models.models import Workspace, WorkspaceDocument, Chunk, User
from app.models.schemas import (
    WorkspaceCreate, WorkspaceOut, WorkspaceDocumentsRequest,
    WorkspaceAskRequest, WorkspaceAskResponse,
)
from app.retrieval.hybrid import retrieve_multi
from app.qa.generator import answer_question

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])
settings = get_settings()


def _get_workspace(db: Session, workspace_id: str, current_user: User) -> Workspace:
    workspace = db.get(Workspace, workspace_id)
    if not workspace or workspace.user_id != current_user.id:
        raise HTTPException(404, "Workspace not found.")
    return workspace


@router.post("", response_model=WorkspaceOut)
def create_workspace(
    payload: WorkspaceCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    workspace = Workspace(name=payload.name, user_id=current_user.id)
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    return WorkspaceOut(id=workspace.id, name=workspace.name, created_at=workspace.created_at, documents=[])


@router.get("", response_model=list[WorkspaceOut])
def list_workspaces(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspaces = (
        db.query(Workspace)
        .filter_by(user_id=current_user.id)
        .order_by(Workspace.created_at.desc())
        .all()
    )
    return [_to_workspace_out(w) for w in workspaces]


@router.get("/{workspace_id}", response_model=WorkspaceOut)
def get_workspace(workspace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = _get_workspace(db, workspace_id, current_user)
    return _to_workspace_out(workspace)


@router.delete("/{workspace_id}")
def delete_workspace(workspace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = _get_workspace(db, workspace_id, current_user)
    db.delete(workspace)
    db.commit()
    return {"status": "deleted"}


@router.post("/{workspace_id}/documents", response_model=WorkspaceOut)
def add_documents(
    workspace_id: str, payload: WorkspaceDocumentsRequest,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    workspace = _get_workspace(db, workspace_id, current_user)
    existing_ids = {link.document_id for link in workspace.document_links}

    for document_id in payload.document_ids:
        if document_id in existing_ids:
            continue
        get_owned_document(document_id, db, current_user)  # 404s if missing/not owned
        db.add(WorkspaceDocument(workspace_id=workspace_id, document_id=document_id))
        existing_ids.add(document_id)

    db.commit()
    db.refresh(workspace)
    return _to_workspace_out(workspace)


@router.delete("/{workspace_id}/documents/{document_id}", response_model=WorkspaceOut)
def remove_document(
    workspace_id: str, document_id: str,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    workspace = _get_workspace(db, workspace_id, current_user)
    link = (
        db.query(WorkspaceDocument)
        .filter_by(workspace_id=workspace_id, document_id=document_id)
        .first()
    )
    if not link:
        raise HTTPException(404, "Document is not in this workspace.")
    db.delete(link)
    db.commit()
    db.refresh(workspace)
    return _to_workspace_out(workspace)


@router.post("/{workspace_id}/ask", response_model=WorkspaceAskResponse)
def ask_workspace(
    workspace_id: str, payload: WorkspaceAskRequest,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    workspace = _get_workspace(db, workspace_id, current_user)
    documents = [link.document for link in workspace.document_links]
    ready_documents = [d for d in documents if d.processing_status == "ready"]

    if not ready_documents:
        raise HTTPException(409, "No ready documents in this workspace yet.")

    documents_chunks = {}
    bm25_by_document = {}
    for document in ready_documents:
        chunks = db.query(Chunk).filter_by(document_id=document.id).order_by(Chunk.chunk_index).all()
        if not chunks:
            continue
        documents_chunks[document.id] = chunks
        bm25_by_document[document.id] = get_bm25(document.id, [c.index_text for c in chunks])

    evidence = retrieve_multi(
        documents_chunks, payload.question, bm25_by_document,
        method="hybrid", top_k=settings.top_k_retrieval, rerank=True,
    )
    result = answer_question(payload.question, evidence, show_document=True)

    return WorkspaceAskResponse(
        answer=result["answer"], grounded=result["grounded"], evidence=result["evidence"],
    )


def _to_workspace_out(workspace: Workspace) -> WorkspaceOut:
    documents = [link.document for link in workspace.document_links]
    return WorkspaceOut(
        id=workspace.id, name=workspace.name, created_at=workspace.created_at, documents=documents,
    )
