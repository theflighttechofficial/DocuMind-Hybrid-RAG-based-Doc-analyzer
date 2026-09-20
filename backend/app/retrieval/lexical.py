"""BM25-based lexical retrieval over a document's chunks."""
import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunk_texts: list[str]) -> BM25Okapi | None:
    if not chunk_texts:
        return None
    tokenized = [_tokenize(t) for t in chunk_texts]
    return BM25Okapi(tokenized)


def search_bm25(bm25: BM25Okapi, query: str, top_k: int = 6):
    """Returns list of (chunk_position, score) sorted descending, normalized to 0-1."""
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    if len(scores) == 0:
        return []
    max_score = max(scores) or 1.0
    ranked = sorted(enumerate(scores), key=lambda kv: -kv[1])[:top_k]
    return [(idx, round(float(score / max_score), 4)) for idx, score in ranked if score > 0]
