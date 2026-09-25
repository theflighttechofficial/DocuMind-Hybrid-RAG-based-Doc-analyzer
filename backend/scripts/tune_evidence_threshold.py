"""Tunes settings.evidence_min_relevance: the reranker-probability cut-off
used to pick evidence when an answer cites no passage.

Runs the same retrieval the Ask endpoint uses (hybrid + rerank, top_k from
settings) for every answerable single-document question in the eval set,
over the eval user's already-processed documents (run scripts/run_eval.py
once first). A passage is labelled relevant when it contains one of the
question's gold evidence quotes. Prints precision/recall/F1/F2 per
threshold and picks the F2-best value: the fallback only runs when the answer
cited nothing, so hiding the passage that holds the answer (recall) costs
more than showing one extra card (precision). No LLM calls.

    python scripts/tune_evidence_threshold.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.core.pipeline import get_bm25  # noqa: E402
from app.models.models import Chunk, Document, User  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402
from scripts.run_eval import EVAL_EMAIL, EVAL_SET, contains  # noqa: E402


def main():
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    db = SessionLocal()
    user = db.query(User).filter_by(email=EVAL_EMAIL).one()
    docs = {d.name: d for d in db.query(Document).filter_by(user_id=user.id, processing_status="ready")}
    top_k = get_settings().top_k_retrieval

    rows = []  # (score, relevant)
    for q in eval_set["questions"]:
        if len(q["documents"]) != 1 or not q["evidence"] or q["type"] == "follow-up":
            continue
        doc = docs[eval_set["documents"][q["documents"][0]]]
        chunks = db.query(Chunk).filter_by(document_id=doc.id).order_by(Chunk.chunk_index).all()
        bm25 = get_bm25(doc.id, [c.index_text for c in chunks])
        for item in retrieve(doc.id, q["question"], chunks, bm25, method="hybrid", top_k=top_k, rerank=True):
            rows.append((item["score"], any(contains(item["chunk"].text, e) for e in q["evidence"])))

    n_rel = sum(r for _, r in rows)
    print(f"{len(rows)} passages, {n_rel} relevant\n threshold  precision  recall     f1     f2   kept")
    best = (0, None)
    for t in [i / 100 for i in range(1, 100)]:
        kept = [r for s, r in rows if s >= t]
        tp = sum(kept)
        p = tp / len(kept) if kept else 0.0
        r = tp / n_rel if n_rel else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
        if f2 > best[0]:
            best = (f2, t)
        if t in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) or t == best[1]:
            print(f"  {t:8.2f}  {p:9.3f}  {r:6.3f}  {f1:6.3f}  {f2:6.3f}  {len(kept):5d}")
    print(f"\nbest F2 {best[0]:.3f} at threshold {best[1]}")


if __name__ == "__main__":
    main()
