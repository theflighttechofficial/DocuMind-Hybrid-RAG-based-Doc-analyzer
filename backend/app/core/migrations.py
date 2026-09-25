"""Lightweight startup migrations for SQLite.

There's no Alembic in this project — `Base.metadata.create_all()` only creates
tables that don't exist yet, it never adds columns to tables that already
exist on disk. Phase 5 (auth) adds `user_id` to two pre-existing tables
(documents, workspaces), so this module patches those in with raw ALTER TABLE
statements, then backfills any pre-existing rows onto a single "legacy" user
so nothing already uploaded becomes orphaned or inaccessible.
"""
import logging

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.core.db import engine, SessionLocal
from app.core.security import hash_password
from app.models.models import User

logger = logging.getLogger(__name__)

LEGACY_USER_EMAIL = "legacy@documind.local"
LEGACY_USER_PASSWORD = "documind123"


def _add_column_if_missing(table: str, column: str, coltype: str):
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return  # table doesn't exist yet — create_all will create it with the column already present
    existing_columns = {col["name"] for col in inspector.get_columns(table)}
    if column in existing_columns:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
    logger.info("Migration: added column %s.%s", table, column)


def _get_or_create_legacy_user(db: Session) -> User:
    user = db.query(User).filter_by(email=LEGACY_USER_EMAIL).first()
    if user:
        return user
    user = User(email=LEGACY_USER_EMAIL, hashed_password=hash_password(LEGACY_USER_PASSWORD))
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.warning(
        "Migration: created legacy user %s (password: %s) and will attach any "
        "pre-existing documents/workspaces to it — log in with these credentials "
        "to see data uploaded before auth was added.",
        LEGACY_USER_EMAIL, LEGACY_USER_PASSWORD,
    )
    return user


def run_startup_migrations():
    _add_column_if_missing("documents", "user_id", "VARCHAR")
    _add_column_if_missing("workspaces", "user_id", "VARCHAR")
    _add_column_if_missing("documents", "title", "VARCHAR")
    _add_column_if_missing("documents", "structure_json", "JSON")

    db = SessionLocal()
    try:
        orphan_count = db.execute(
            text("SELECT COUNT(*) FROM documents WHERE user_id IS NULL")
        ).scalar() or 0
        orphan_count += db.execute(
            text("SELECT COUNT(*) FROM workspaces WHERE user_id IS NULL")
        ).scalar() or 0

        if orphan_count == 0:
            return

        legacy_user = _get_or_create_legacy_user(db)
        db.execute(
            text("UPDATE documents SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": legacy_user.id},
        )
        db.execute(
            text("UPDATE workspaces SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": legacy_user.id},
        )
        db.commit()
    finally:
        db.close()
