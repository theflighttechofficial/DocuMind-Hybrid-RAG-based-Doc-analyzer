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

    Returns the list sorted by cross-encoder score and truncated to top_k. Each
    item's "score" becomes the cross-encoder probability (sigmoid of the logit,
    0-1) so the number shown as relevance is the one the list is ordered by;
    the fused retrieval score is kept as "retrieval_score" and the raw logit as
    "rerank_score".
    """
    if not candidates:
        return candidates

    model = get_reranker()
    # score the header-prefixed text: a bare markdown table row like
    # "| Maximum payload | 4.5 kg |" means more with its section title attached
    pairs = [(query, getattr(item["chunk"], "index_text", item["chunk"].text)) for item in candidates]
    scores = model.predict(pairs)

    for item, score in zip(candidates, scores):
        item["retrieval_score"] = item["score"]
        item["rerank_score"] = float(score)
        item["score"] = round(1.0 / (1.0 + math.exp(-float(score))), 4)

    return sorted(candidates, key=lambda item: -item["rerank_score"])[:top_k]
