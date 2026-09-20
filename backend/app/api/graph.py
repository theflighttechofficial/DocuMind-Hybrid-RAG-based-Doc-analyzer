"""Builds a co-occurrence graph of named entities: entities mentioned in the
same chunk are linked, weighted by co-occurrence count. This is a real,
computed graph from the document (not a fabricated/staged one).
"""
import re
from collections import defaultdict
from itertools import combinations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.deps import get_current_user, get_owned_document
from app.models.models import Chunk, User

router = APIRouter(prefix="/api/documents", tags=["graph"])


@router.get("/{document_id}/graph")
def get_knowledge_graph(
    document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    doc = get_owned_document(document_id, db, current_user)
    if doc.processing_status != "ready":
        raise HTTPException(409, f"Document is not ready yet (status: {doc.processing_status}).")

    entities = doc.entities_json or {}
    # flatten to a lookup of entity_text -> type, keep top entities only for a readable graph
    entity_lookup = {}
    for etype, items in entities.items():
        for item in items[:15]:
            entity_lookup[item["text"].lower()] = {"text": item["text"], "type": etype, "count": item["count"]}

    if not entity_lookup:
        return {"nodes": [], "edges": []}

    # Word-boundary regex per entity, not a plain substring check — otherwise a
    # short entity like "US" false-positives inside "using"/"bus"/"business",
    # inflating mention counts and fabricating co-occurrence edges that were
    # never actually in the text.
    entity_patterns = {
        key: re.compile(r"\b" + re.escape(key) + r"\b")
        for key in entity_lookup
    }

    chunks = db.query(Chunk).filter_by(document_id=document_id).all()

    edge_weights = defaultdict(int)
    node_pages = defaultdict(set)
    node_mentions = defaultdict(int)

    for chunk in chunks:
        text_lower = chunk.text.lower()
        present = []
        for key, pattern in entity_patterns.items():
            count = len(pattern.findall(text_lower))
            if count > 0:
                present.append(key)
                node_pages[key].add(chunk.page_number)
                node_mentions[key] += count
        for a, b in combinations(sorted(set(present)), 2):
            edge_weights[(a, b)] += 1

    nodes = [
        {
            "id": key,
            "label": entity_lookup[key]["text"],
            "type": entity_lookup[key]["type"],
            "mentions": node_mentions[key] or entity_lookup[key]["count"],
            "pages": sorted(node_pages[key])[:10],
        }
        for key in entity_lookup
        if node_mentions[key] > 0
    ]

    edges = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in edge_weights.items()
        if w > 0
    ]

    return {"nodes": nodes, "edges": edges}
