"""Optional cross-encoder reranking stage. Re-scores retrieval candidates with
a query-passage cross-encoder (more accurate, more expensive than bi-encoder
similarity) before truncating to top_k.
"""
from app.core.model_loader import load_model


def get_reranker():
    from sentence_transformers import CrossEncoder
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return load_model("reranker", [(name, lambda: CrossEncoder(name, device="cpu"))])


def rerank(query: str, candidates: list, top_k: int = 6) -> list:
    """candidates: list of {"chunk": Chunk, "score": float, "method": str}, already
    ranked by the upstream retrieval score but not yet truncated to top_k.
    Returns the list re-sorted by cross-encoder score and truncated to top_k, with
    each item's original score preserved and a "rerank_score" added.
    """
    if not candidates:
        return candidates

    model = get_reranker()
    pairs = [(query, item["chunk"].text) for item in candidates]
    scores = model.predict(pairs)

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    return sorted(candidates, key=lambda item: -item["rerank_score"])[:top_k]
