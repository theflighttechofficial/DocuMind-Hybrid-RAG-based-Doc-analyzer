"""Shared auth dependencies: current-user extraction and per-user document
ownership checks, reused across every router that touches documents/workspaces.
"""
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import decode_access_token
from app.models.models import Document, User


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        user_id = decode_access_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired token.")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(401, "User not found.")
    return user


def get_owned_document(document_id: str, db: Session, user: User) -> Document:
    """Fetches a document and checks it belongs to `user`. Returns 404 (not 403)
    on a mismatch so an unauthorized caller can't learn a document exists.
    """
    doc = db.get(Document, document_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404, "Document not found.")
    return doc
