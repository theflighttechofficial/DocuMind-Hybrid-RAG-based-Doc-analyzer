"""ORM models for DocuMind AI."""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, ARRAY, JSON
)
from sqlalchemy.orm import relationship

from app.core.db import Base


def gen_id() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    sentence_count = Column(Integer, default=0)
    paragraph_count = Column(Integer, default=0)
    processing_status = Column(String, default="uploaded")
    # uploaded -> extracting -> analyzing -> embedding -> indexing -> ready -> failed
    error_message = Column(Text, nullable=True)
    storage_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # cached analysis (JSON blobs, computed once)
    entities_json = Column(JSON, nullable=True)
    keywords_json = Column(JSON, nullable=True)
    pos_json = Column(JSON, nullable=True)
    stats_json = Column(JSON, nullable=True)
    summary_json = Column(JSON, nullable=True)
    toc_json = Column(JSON, nullable=True)

    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="document", cascade="all, delete-orphan")
    workspace_links = relationship("WorkspaceDocument", back_populates="document", cascade="all, delete-orphan")


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id = Column(String, primary_key=True, default=gen_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    document = relationship("Document", back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=gen_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    section = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    vector_row = Column(Integer, nullable=True)  # row index into the FAISS index

    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=gen_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=gen_id)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    evidence_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document_links = relationship(
        "WorkspaceDocument", back_populates="workspace", cascade="all, delete-orphan"
    )


class WorkspaceDocument(Base):
    __tablename__ = "workspace_documents"

    id = Column(String, primary_key=True, default=gen_id)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    workspace = relationship("Workspace", back_populates="document_links")
    document = relationship("Document", back_populates="workspace_links")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=gen_id)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    retrieved_context = Column(Text, nullable=True)
    generated_answer = Column(Text, nullable=True)
    method = Column(String, nullable=True)  # bm25 | semantic | hybrid
    metrics_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
