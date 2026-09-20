"""Real metric computation for retrieval and QA evaluation. No fabricated numbers."""
import re


def precision_at_k(retrieved_pages: list[int], relevant_pages: set[int], k: int) -> float:
    top_k = retrieved_pages[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for p in top_k if p in relevant_pages)
    return round(hits / len(top_k), 4)


def recall_at_k(retrieved_pages: list[int], relevant_pages: set[int], k: int) -> float:
    if not relevant_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for p in top_k if p in relevant_pages)
    return round(hits / len(relevant_pages), 4)


def mean_reciprocal_rank(retrieved_pages: list[int], relevant_pages: set[int]) -> float:
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return round(1.0 / rank, 4)
    return 0.0


_STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or", "for", "on", "at"}


def _normalize_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOP]


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_tokens(prediction)
    truth_tokens = _normalize_tokens(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), truth_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def semantic_similarity(prediction: str, ground_truth: str) -> float:
    """Cosine similarity via the same local embedder used for retrieval."""
    from app.retrieval.embeddings import embed_texts
    import numpy as np

    vecs = embed_texts([prediction, ground_truth])
    if vecs.shape[0] < 2:
        return 0.0
    sim = float(np.dot(vecs[0], vecs[1]))
    return round(sim, 4)


def bertscore_f1(prediction: str, ground_truth: str) -> float:
    """Token-level BERTScore F1: greedy-matches each prediction token against
    its best-matching reference token (and vice versa) by cosine similarity,
    then combines into an F1. Unlike semantic_similarity (one pooled vector
    per whole answer), this gives partial credit when only part of a longer,
    more verbose answer actually matches the reference — a single wrong or
    missing clause doesn't sink the whole score the way pooled cosine can.

    Reuses the same all-MiniLM-L6-v2 weights already downloaded for retrieval
    (bert_score pulls from the same shared HuggingFace Hub cache by repo id,
    so this doesn't download a second model).
    """
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    from bert_score import score

    _, _, f1 = score(
        [prediction], [ground_truth],
        model_type="sentence-transformers/all-MiniLM-L6-v2", num_layers=6,
        verbose=False,
    )
    return round(float(f1.item()), 4)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence over tokens."""
    pred = _normalize_tokens(prediction)
    ref = _normalize_tokens(reference)
    if not pred or not ref:
        return 0.0

    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return round(2 * precision * recall / (precision + recall), 4)


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    pred = _normalize_tokens(prediction)
    ref = _normalize_tokens(reference)
    if len(pred) < n or len(ref) < n:
        return 0.0

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    pred_grams = ngrams(pred, n)
    ref_grams = ngrams(ref, n)
    if not pred_grams or not ref_grams:
        return 0.0

    from collections import Counter
    pred_counts, ref_counts = Counter(pred_grams), Counter(ref_grams)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_grams)
    recall = overlap / len(ref_grams)
    return round(2 * precision * recall / (precision + recall), 4)
