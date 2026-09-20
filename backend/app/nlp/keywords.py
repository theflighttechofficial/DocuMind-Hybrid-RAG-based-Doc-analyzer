"""Keyword extraction: TF-IDF (against a background corpus of the doc's own
sentences, so it isn't just raw frequency), a lightweight TextRank
implementation, and KeyBERT (BERT-embedding cosine similarity between
candidate n-grams and the whole document). All three are real computations
over the document text.
"""
import re
from collections import Counter
from functools import lru_cache

import networkx as nx
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from app.nlp.nltk_setup import ensure_nltk_data

ensure_nltk_data()

try:
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = set()


def extract_keywords_tfidf(full_text: str, top_n: int = 15) -> list:
    try:
        sentences = [s for s in sent_tokenize(full_text) if len(s.split()) > 3]
    except Exception:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.split()) > 3]

    if len(sentences) < 3:
        return _fallback_frequency(full_text, top_n)

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=2000, ngram_range=(1, 2), min_df=1
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return _fallback_frequency(full_text, top_n)

    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda kv: -kv[1])

    max_score = ranked[0][1] if ranked else 1.0
    return [
        {"term": term, "score": round(float(score / max_score), 3)}
        for term, score in ranked[:top_n]
    ]


def extract_keywords_textrank(full_text: str, top_n: int = 15, window: int = 4) -> list:
    """Graph-based keyword ranking (TextRank), computed via networkx PageRank."""
    try:
        tokens = [
            w.lower() for w in word_tokenize(full_text)
            if w.isalpha() and w.lower() not in _STOPWORDS and len(w) > 2
        ]
    except Exception:
        tokens = [
            w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", full_text)
            if w.lower() not in _STOPWORDS and len(w) > 2
        ]

    if len(tokens) < 5:
        return _fallback_frequency(full_text, top_n)

    graph = nx.Graph()
    graph.add_nodes_from(set(tokens))
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            if tokens[i] != tokens[j]:
                if graph.has_edge(tokens[i], tokens[j]):
                    graph[tokens[i]][tokens[j]]["weight"] += 1
                else:
                    graph.add_edge(tokens[i], tokens[j], weight=1)

    try:
        scores = nx.pagerank(graph, weight="weight")
    except Exception:
        return _fallback_frequency(full_text, top_n)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    max_score = ranked[0][1] if ranked else 1.0
    return [{"term": term, "score": round(score / max_score, 3)} for term, score in ranked]


@lru_cache
def _get_keybert():
    from keybert import KeyBERT
    from app.retrieval.embeddings import get_embedder

    # Reuse the same sentence-transformers instance already loaded for
    # retrieval embeddings — KeyBERT accepts a pre-built model instead of a
    # model name, so this doesn't load a second embedding model into memory.
    return KeyBERT(model=get_embedder())


def _noun_phrase_vocabulary(text: str) -> list[str]:
    """Restricts KeyBERT's own CountVectorizer to a vocabulary of spaCy noun
    chunks (already loaded for NER/POS), instead of letting it score every raw
    n-gram in the text — that default is what produces ungrammatical fragments
    like "answering combines" or "documents evaluate" (a verb glued to
    whatever noun sits next to it).
    """
    from app.nlp.spacy_model import parse

    doc = parse(text, max_chars=20_000)
    vocabulary = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        words = [w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", phrase)]
        while words and words[0] in _STOPWORDS:
            words = words[1:]
        # require 2+ words — single generic nouns ("retrieval", "document")
        # are technically valid noun chunks but far less informative than the
        # multi-word phrases KeyBERT is actually meant to surface
        if 2 <= len(words) <= 4:
            vocabulary.add(" ".join(words))
    return list(vocabulary)


def extract_keywords_keybert(full_text: str, top_n: int = 15) -> list:
    text = full_text[:20_000]  # KeyBERT embeds the whole doc; cap for latency
    if len(text.split()) < 10:
        return _fallback_frequency(full_text, top_n)

    vocabulary = _noun_phrase_vocabulary(text)
    if len(vocabulary) < 5:
        return _fallback_frequency(full_text, top_n)

    kw_model = _get_keybert()
    try:
        vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(2, 4))
        pairs = kw_model.extract_keywords(
            text,
            vectorizer=vectorizer,
            use_mmr=True,       # Maximal Marginal Relevance — diversifies results
            diversity=0.5,
            top_n=top_n,
        )
    except Exception:
        return _fallback_frequency(full_text, top_n)

    if not pairs:
        return _fallback_frequency(full_text, top_n)

    return [{"term": term, "score": round(float(score), 3)} for term, score in pairs]


def _fallback_frequency(full_text: str, top_n: int) -> list:
    words = [
        w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", full_text)
        if w.lower() not in _STOPWORDS
    ]
    counts = Counter(words).most_common(top_n)
    max_count = counts[0][1] if counts else 1
    return [{"term": w, "score": round(c / max_count, 3)} for w, c in counts]
