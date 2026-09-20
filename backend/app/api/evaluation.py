from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.evaluation.evaluator import run_evaluation
from app.models.models import User

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvalItem(BaseModel):
    question: str
    answer: str | None = None
    page: int | None = None
    is_summary_task: bool = False


class EvalRequest(BaseModel):
    document_id: str
    eval_set: list[EvalItem]
    compare_rerank: bool = False


@router.post("/run")
def run_eval(payload: EvalRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    get_owned_document(payload.document_id, db, current_user)
    try:
        result = run_evaluation(
            db, payload.document_id, [item.model_dump() for item in payload.eval_set],
            compare_rerank=payload.compare_rerank,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return result
