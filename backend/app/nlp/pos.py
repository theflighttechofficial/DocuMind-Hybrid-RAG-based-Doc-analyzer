"""POS tag distribution via spaCy's tagger (part of the same en_core_web_lg
pipeline used for NER) — replaces NLTK's averaged_perceptron_tagger. spaCy's
.pos_ already gives Universal Dependencies coarse categories directly, so this
maps cleanly onto the app's existing coarse category names.
"""
from collections import Counter

from app.nlp.spacy_model import parse

_COARSE_MAP = {
    "NOUN": "NOUN", "PROPN": "NOUN",
    "VERB": "VERB", "AUX": "VERB",
    "ADJ": "ADJECTIVE",
    "PRON": "PRONOUN",
    "ADV": "ADVERB",
    "DET": "DETERMINER",
    "ADP": "PREPOSITION",
}


def compute_pos_distribution(full_text: str, max_chars: int = 200_000) -> dict:
    return compute_pos_distribution_from_doc(parse(full_text, max_chars))


def compute_pos_distribution_from_doc(doc) -> dict:
    """Same as compute_pos_distribution, but takes an already-parsed spaCy Doc
    — lets callers that also need NER (pipeline.py) share one parse.
    """
    coarse_counts = Counter()
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        category = _COARSE_MAP.get(token.pos_, "OTHER")
        coarse_counts[category] += 1

    total = sum(coarse_counts.values()) or 1
    distribution = {
        cat: round(count / total * 100, 1)
        for cat, count in coarse_counts.most_common()
    }
    return distribution
