"""Named entity recognition via spaCy's en_core_web_lg statistical NER model
(word-vector + CNN pipeline, OntoNotes label set) — replaces the previous
NLTK chunker, which only distinguished PERSON/ORGANIZATION/GPE and needed a
regex bolted on for DATE. spaCy natively recognizes 18 entity types, including
MONEY, PERCENT, LAW, NORP, WORK_OF_ART, PRODUCT, EVENT, and more.
"""
from collections import defaultdict

from app.nlp.spacy_model import parse

# A few OntoNotes labels renamed to the friendlier names this app already used
# (PERSON/ORGANIZATION/LOCATION/DATE), so existing frontend color maps for
# those four still apply. Every other spaCy label passes through unchanged —
# that's the whole point of the upgrade (MONEY, PERCENT, LAW, NORP, ...).
_LABEL_RENAME = {
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
}


def extract_entities(full_text: str, max_chars: int = 200_000) -> dict:
    """Returns {entity_type: [{"text":..., "count":...}, ...]}."""
    return extract_entities_from_doc(parse(full_text, max_chars))


def extract_entities_from_doc(doc) -> dict:
    """Same as extract_entities, but takes an already-parsed spaCy Doc — lets
    callers that also need POS (pipeline.py) share one parse instead of two.
    """
    counts = defaultdict(lambda: defaultdict(int))

    for ent in doc.ents:
        entity_text = ent.text.strip()
        if len(entity_text) < 2:
            continue
        label = _LABEL_RENAME.get(ent.label_, ent.label_)
        counts[label][entity_text] += 1

    result = {}
    for label, entity_counts in counts.items():
        ranked = sorted(entity_counts.items(), key=lambda kv: -kv[1])
        result[label] = [{"text": t, "count": c} for t, c in ranked[:30]]

    return result
