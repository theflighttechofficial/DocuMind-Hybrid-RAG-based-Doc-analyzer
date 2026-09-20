"""Combines semantic (FAISS) and lexical (BM25) retrieval into three
comparable retrieval methods, as required by the evaluation dashboard.
"""
from app.retrieval import vector_store, lexical
from app.retrieval.embeddings import embed_query


def retrieve(document_id: str, query: str, chunks: list, bm25, method: str = "hybrid", top_k: int = 6, rerank: bool = False):
    """chunks: ordered list of Chunk ORM rows (index == vector_row == bm25 position).
    Returns list of dicts: {chunk, score, method}.

    rerank: when True, over-fetch candidates and re-score them with a cross-encoder
    before truncating to top_k (see app.retrieval.reranker).
    """
    if not chunks:
        return []

    semantic_scores = {}
    lexical_scores = {}

    if method in ("semantic", "hybrid"):
        query_vec = embed_query(query)
        scores, indices = vector_store.search(document_id, query_vec, top_k=max(top_k * 3, 15))
        if scores is not None:
            for score, idx in zip(scores, indices):
                if idx == -1 or idx >= len(chunks):
                    continue
                semantic_scores[int(idx)] = float(score)

    if method in ("bm25", "hybrid"):
        for idx, score in lexical.search_bm25(bm25, query, top_k=max(top_k * 3, 15)):
            lexical_scores[idx] = score

    if method == "semantic":
        combined = {idx: score for idx, score in semantic_scores.items()}
    elif method == "bm25":
        combined = {idx: score for idx, score in lexical_scores.items()}
    else:  # hybrid: weighted sum, favoring semantic slightly
        all_idx = set(semantic_scores) | set(lexical_scores)
        combined = {
            idx: 0.6 * semantic_scores.get(idx, 0.0) + 0.4 * lexical_scores.get(idx, 0.0)
            for idx in all_idx
        }

    candidate_k = max(top_k * 3, 20) if rerank else top_k
    ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:candidate_k]

    results = []
    for idx, score in ranked:
        if score <= 0:
            continue
        results.append({"chunk": chunks[idx], "score": round(score, 4), "method": method})

    if rerank and results:
        from app.retrieval.reranker import rerank as rerank_fn
        results = rerank_fn(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    return results


def retrieve_multi(
    documents_chunks: dict, query: str, bm25_by_document: dict, method: str = "hybrid", top_k: int = 6,
    rerank: bool = False,
):
    """Runs retrieval independently per document, then merges and re-ranks by score.

    documents_chunks: {document_id: [Chunk, ...]}
    bm25_by_document: {document_id: bm25_index}
    Returns a merged, score-sorted list of {chunk, score, method}, truncated to top_k,
    each item's chunk carrying its own document_id for downstream tagging.

    rerank: raw hybrid scores aren't comparable across documents (each document's
    BM25/FAISS scores are normalized against its own chunk set), so merging by raw
    score can favor one document's ranking over another's for the wrong reasons.
    A cross-encoder score, by contrast, is a direct query-passage relevance judgment
    with no per-document normalization — so when True, each document's candidates are
    over-fetched, merged, then reranked together in one global pass for a fair,
    cross-document-comparable final ordering.
    """
    # over-fetch per document so the merged top_k isn't dominated by one document's
    # ranking alone before scores are compared globally
    per_doc_top_k = max(top_k, 10) if rerank else top_k

    merged = []
    for document_id, chunks in documents_chunks.items():
        bm25 = bm25_by_document.get(document_id)
        if bm25 is None or not chunks:
            continue
        per_doc_results = retrieve(document_id, query, chunks, bm25, method=method, top_k=per_doc_top_k)
        merged.extend(per_doc_results)

    if rerank and merged:
        from app.retrieval.reranker import rerank as rerank_fn
        return rerank_fn(query, merged, top_k=top_k)

    merged.sort(key=lambda item: -item["score"])
    return merged[:top_k]
