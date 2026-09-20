"""Compares two documents by matching their chunks semantically (not raw
string diff), then asking the LLM to characterize each matched/unmatched
pair as added, removed, modified, or unchanged.
"""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.models.models import Chunk, User
from app.models.schemas import CompareRequest
from app.retrieval.embeddings import embed_texts
from app.qa.llm_client import generate

router = APIRouter(prefix="/api/compare", tags=["comparison"])

_SYSTEM = """You compare two related passages from different versions of a document.
Classify the relationship as exactly one of: UNCHANGED, MODIFIED, ADDED, REMOVED.
Then give a one-sentence explanation of what changed, if anything.
Respond in the format:
CLASSIFICATION: <label>
EXPLANATION: <sentence>
"""

_SIMILARITY_MATCH_THRESHOLD = 0.55


@router.post("")
def compare_documents(
    payload: CompareRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    doc_a = get_owned_document(payload.document_id_a, db, current_user)
    doc_b = get_owned_document(payload.document_id_b, db, current_user)
    if doc_a.processing_status != "ready" or doc_b.processing_status != "ready":
        raise HTTPException(409, "Both documents must be fully processed before comparing.")

    chunks_a = db.query(Chunk).filter_by(document_id=doc_a.id).order_by(Chunk.chunk_index).all()
    chunks_b = db.query(Chunk).filter_by(document_id=doc_b.id).order_by(Chunk.chunk_index).all()

    if not chunks_a or not chunks_b:
        raise HTTPException(422, "One or both documents have no content to compare.")

    vecs_a = embed_texts([c.text for c in chunks_a])
    vecs_b = embed_texts([c.text for c in chunks_b])

    # cosine similarity matrix (vectors are already normalized)
    sim_matrix = vecs_a @ vecs_b.T

    matched_b = set()
    results = []

    for i, chunk_a in enumerate(chunks_a):
        best_j = int(np.argmax(sim_matrix[i])) if sim_matrix.shape[1] else -1
        best_score = float(sim_matrix[i, best_j]) if best_j >= 0 else 0.0

        if best_score >= _SIMILARITY_MATCH_THRESHOLD:
            matched_b.add(best_j)
            chunk_b = chunks_b[best_j]

            if best_score > 0.97:
                classification, explanation = "UNCHANGED", "The content is effectively identical."
            else:
                try:
                    llm_out = generate(
                        f"Version A (page {chunk_a.page_number}):\n{chunk_a.text}\n\n"
                        f"Version B (page {chunk_b.page_number}):\n{chunk_b.text}",
                        system_instruction=_SYSTEM,
                        temperature=0.1,
                    )
                    classification, explanation = _parse_classification(llm_out)
                except RuntimeError:
                    classification, explanation = "MODIFIED", "Answer service unavailable for detailed comparison."

            results.append({
                "section": chunk_a.section or chunk_b.section,
                "classification": classification,
                "explanation": explanation,
                "similarity": round(best_score, 3),
                "version_a": {"page": chunk_a.page_number, "text": chunk_a.text[:500]},
                "version_b": {"page": chunk_b.page_number, "text": chunk_b.text[:500]},
            })
        else:
            results.append({
                "section": chunk_a.section,
                "classification": "REMOVED",
                "explanation": "This content does not appear in Document B.",
                "similarity": round(best_score, 3),
                "version_a": {"page": chunk_a.page_number, "text": chunk_a.text[:500]},
                "version_b": None,
            })

    for j, chunk_b in enumerate(chunks_b):
        if j not in matched_b:
            results.append({
                "section": chunk_b.section,
                "classification": "ADDED",
                "explanation": "This content is new in Document B.",
                "similarity": 0.0,
                "version_a": None,
                "version_b": {"page": chunk_b.page_number, "text": chunk_b.text[:500]},
            })

    summary = {
        "unchanged": sum(1 for r in results if r["classification"] == "UNCHANGED"),
        "modified": sum(1 for r in results if r["classification"] == "MODIFIED"),
        "added": sum(1 for r in results if r["classification"] == "ADDED"),
        "removed": sum(1 for r in results if r["classification"] == "REMOVED"),
    }

    return {"document_a": doc_a.name, "document_b": doc_b.name, "summary": summary, "results": results}


def _parse_classification(text: str) -> tuple[str, str]:
    classification, explanation = "MODIFIED", text.strip()
    for line in text.splitlines():
        if line.upper().startswith("CLASSIFICATION:"):
            classification = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()
    if classification not in {"UNCHANGED", "MODIFIED", "ADDED", "REMOVED"}:
        classification = "MODIFIED"
    return classification, explanation
