"""Shows raw reranker logits next to the displayed relevance for one question
on one of the eval user's documents, plus the live /ask answer and cited cards.

    python scripts/score_probe.py <label> "<question>" [document-name-prefix]
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.core.pipeline import get_bm25  # noqa: E402
from app.models.models import Chunk, Document, User  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402
from scripts.run_eval import EVAL_EMAIL, EVAL_PASSWORD  # noqa: E402

label, question = sys.argv[1], sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else "KESTREL"

db = SessionLocal()
user = db.query(User).filter_by(email=EVAL_EMAIL).one()
doc = db.query(Document).filter(Document.user_id == user.id, Document.name.startswith(prefix)).one()
chunks = db.query(Chunk).filter_by(document_id=doc.id).order_by(Chunk.chunk_index).all()
bm25 = get_bm25(doc.id, [c.index_text for c in chunks])
ranked = retrieve(doc.id, question, chunks, bm25, method="hybrid", top_k=get_settings().top_k_retrieval, rerank=True)
candidates = [{
    "rank": i, "page": r["chunk"].page_number, "words": len(r["chunk"].text.split()),
    "raw_logit": round(r["rerank_score"], 3), "sigmoid_of_logit": r.get("rerank_prob"), "displayed_relevance": r["score"],
    "text": r["chunk"].text[:90],
} for i, r in enumerate(ranked, start=1)]

B = "http://127.0.0.1:8000/api"
s = requests.Session()
s.headers["Authorization"] = "Bearer " + s.post(B + "/auth/login", json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD}).json()["access_token"]
ask = s.post(f"{B}/documents/{doc.id}/ask", json={"question": question}).json()
out = {
    "question": question,
    "chunks_in_document": len(chunks),
    "reranked_candidates": candidates,
    "answer": ask["answer"],
    "evidence": [{k: e.get(k) for k in ("citation", "page", "rerank_logit", "relevance", "text")} for e in ask["evidence"]],
}
path = Path(__file__).resolve().parents[1] / "eval_results" / "examples" / f"score_probe_{label}.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
