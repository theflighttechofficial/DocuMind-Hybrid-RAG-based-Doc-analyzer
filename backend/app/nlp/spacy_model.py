"""Shared spaCy pipeline, loaded once and reused by ner.py and pos.py so we
don't pay the ~1GB load cost twice. Loaded lazily (only on first actual NLP
call) rather than at import time, since not every request needs it.

Model choice: en_core_web_lg over en_core_web_trf. The transformer model is
more accurate but needs 2-4GB+ RAM to load its RoBERTa weights — this machine
had only ~2.1GB free at install time, which is enough for en_core_web_lg
(~1GB resident) but too tight to safely run trf alongside the app server.
"""
from app.core.model_loader import load_model


def get_nlp():
    import spacy
    return load_model("spacy", [("en_core_web_lg", lambda: spacy.load("en_core_web_lg"))])


def parse(text: str, max_chars: int = 200_000):
    """Runs the shared spaCy pipeline once. NER and POS both need a parsed Doc
    over the same document text — computing it once here (instead of each
    module calling nlp() separately) halves the per-document parse cost.
    """
    return get_nlp()(text[:max_chars])
