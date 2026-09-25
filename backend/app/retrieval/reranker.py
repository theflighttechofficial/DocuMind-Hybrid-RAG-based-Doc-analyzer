"""Optional cross-encoder reranking stage. Re-scores retrieval candidates with
a query-passage cross-encoder (more accurate, more expensive than bi-encoder
similarity) before truncating to top_k.
"""
import math

from app.core.model_loader import load_model


def get_reranker():
    from sentence_transformers import CrossEncoder
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return load_model("reranker", [(name, lambda: CrossEncoder(name, device="cpu"))])


def rerank(query: str, candidates: list, top_k: int = 6) -> list:
    """candidates: list of {"chunk": Chunk, "score": float, "method": str}, already
    ranked by the upstream retrieval score but not yet truncated to top_k.

    Returns the list sorted by cross-encoder score and truncated to top_k, with:
      "rerank_score"     raw cross-encoder logit (unbounded, uncalibrated)
      "rerank_prob"      sigmoid(logit); an absolute but poorly calibrated value —
                         ms-marco logits for under-specified questions ("what company
                         is this?") are negative even for the passage that answers it
      "score"            displayed relevance: the logit min-max scaled over the whole
                         candidate pool (~20 passages, most of them irrelevant), so the
                         best passage for this question reads 1.0 and a clearly
                         unrelated one ~0. The list is ordered by this value.
      "retrieval_score"  the fused BM25/dense score from before reranking
    """
    if not candidates:
        return candidates

    model = get_reranker()
    # score the header-prefixed text: a bare markdown table row like
    # "| Maximum payload | 4.5 kg |" means more with its section title attached
    pairs = [(query, getattr(item["chunk"], "index_text", item["chunk"].text)) for item in candidates]
    logits = [float(x) for x in model.predict(pairs)]
    lo, hi = min(logits), max(logits)

    for item, logit in zip(candidates, logits):
        item["retrieval_score"] = item["score"]
        item["rerank_score"] = logit
        item["rerank_prob"] = round(1.0 / (1.0 + math.exp(-logit)), 4)
        item["score"] = round((logit - lo) / (hi - lo), 4) if hi > lo else 1.0

    return sorted(candidates, key=lambda item: -item["rerank_score"])[:top_k]
