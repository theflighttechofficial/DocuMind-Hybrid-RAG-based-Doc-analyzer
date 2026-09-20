import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.db import get_db, SessionLocal
from app.core.deps import get_current_user, get_owned_document
from app.core.pipeline import get_bm25
from app.models.models import Document, Chunk, Conversation, Message, User
from app.models.schemas import AskRequest, AskResponse
from app.retrieval.hybrid import retrieve
from app.qa.generator import answer_question, stream_answer_question
from app.core.config import get_settings

router = APIRouter(prefix="/api/documents", tags=["chat"])
settings = get_settings()


def _require_ready(doc: Document):
    if doc.processing_status != "ready":
        raise HTTPException(409, f"Document is not ready yet (status: {doc.processing_status}).")


def _build_retrieval_query(question: str, history: list[Message]) -> str:
    """The generation prompt gets full conversation history, but retrieval was
    only ever searching on the raw current question — so a pronoun-style
    follow-up like "what weighting did it use?" had nothing to anchor "it" to
    and would pull irrelevant chunks, even though the LLM prompt had the answer
    in reach. Folding the last couple of user turns into the search query fixes
    recall for follow-ups without an extra LLM round trip.
    """
    if not history:
        return question
    recent_user_turns = [m.content for m in history[-6:] if m.role == "user"]
    if not recent_user_turns:
        return question
    return " ".join(recent_user_turns[-2:] + [question])


@router.post("/{document_id}/ask", response_model=AskResponse)
def ask_document(
    document_id: str, payload: AskRequest,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    doc = get_owned_document(document_id, db, current_user)
    _require_ready(doc)

    if payload.conversation_id:
        conversation = db.get(Conversation, payload.conversation_id)
        if not conversation or conversation.document_id != document_id:
            raise HTTPException(404, "Conversation not found.")
    else:
        conversation = Conversation(document_id=document_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    history = (
        db.query(Message)
        .filter_by(conversation_id=conversation.id)
        .order_by(Message.created_at)
        .all()
    )

    chunks = db.query(Chunk).filter_by(document_id=document_id).order_by(Chunk.chunk_index).all()
    chunk_texts = [c.text for c in chunks]
    bm25 = get_bm25(document_id, chunk_texts)

    retrieval_query = _build_retrieval_query(payload.question, history)
    evidence = retrieve(document_id, retrieval_query, chunks, bm25, method="hybrid", top_k=settings.top_k_retrieval, rerank=True)
    result = answer_question(payload.question, evidence, history_messages=history)

    db.add(Message(conversation_id=conversation.id, role="user", content=payload.question))
    db.add(Message(
        conversation_id=conversation.id, role="assistant",
        content=result["answer"], evidence_json=result["evidence"],
    ))
    db.commit()

    return AskResponse(
        conversation_id=conversation.id,
        answer=result["answer"],
        grounded=result["grounded"],
        evidence=result["evidence"],
    )


@router.post("/{document_id}/ask/stream")
def ask_document_stream(
    document_id: str, payload: AskRequest,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Server-Sent Events variant of /ask. Emits `delta` events with each answer
    chunk as it streams from Gemini, then a final `done` event carrying the same
    shape as AskResponse, once the conversation/message rows have been persisted.
    """
    doc = get_owned_document(document_id, db, current_user)
    _require_ready(doc)

    if payload.conversation_id:
        conversation = db.get(Conversation, payload.conversation_id)
        if not conversation or conversation.document_id != document_id:
            raise HTTPException(404, "Conversation not found.")
    else:
        conversation = Conversation(document_id=document_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    conversation_id = conversation.id
    history = (
        db.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at)
        .all()
    )

    chunks = db.query(Chunk).filter_by(document_id=document_id).order_by(Chunk.chunk_index).all()
    chunk_texts = [c.text for c in chunks]
    bm25 = get_bm25(document_id, chunk_texts)

    retrieval_query = _build_retrieval_query(payload.question, history)
    evidence = retrieve(document_id, retrieval_query, chunks, bm25, method="hybrid", top_k=settings.top_k_retrieval, rerank=True)

    def event_stream():
        yield f"data: {json.dumps({'type': 'conversation', 'conversation_id': conversation_id})}\n\n"

        result = None
        try:
            for kind, data in stream_answer_question(payload.question, evidence, history_messages=history):
                if kind == "delta":
                    yield f"data: {json.dumps({'type': 'delta', 'text': data})}\n\n"
                else:
                    result = data
        except Exception:
            logging.getLogger(__name__).exception("Streaming ask failed mid-response")

        if result is None:
            # the generator raised before ever yielding a final result — surface
            # a real error to the client instead of crashing the connection silently
            result = {
                "answer": "The answer service is temporarily unavailable. Please try again shortly.",
                "grounded": False, "evidence": [],
            }
            yield f"data: {json.dumps({'type': 'delta', 'text': result['answer']})}\n\n"

        # persist with a fresh session — the request-scoped one may be closed by
        # the time this generator finishes draining in some ASGI setups
        session = SessionLocal()
        try:
            session.add(Message(conversation_id=conversation_id, role="user", content=payload.question))
            session.add(Message(
                conversation_id=conversation_id, role="assistant",
                content=result["answer"], evidence_json=result["evidence"],
            ))
            session.commit()
        finally:
            session.close()

        yield f"data: {json.dumps({'type': 'done', **result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{document_id}/conversations/{conversation_id}/messages")
def get_messages(
    document_id: str, conversation_id: str,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    get_owned_document(document_id, db, current_user)
    conversation = db.get(Conversation, conversation_id)
    if not conversation or conversation.document_id != document_id:
        raise HTTPException(404, "Conversation not found.")
    messages = (
        db.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()
    )
    return [
        {"role": m.role, "content": m.content, "evidence": m.evidence_json, "created_at": m.created_at}
        for m in messages
    ]
