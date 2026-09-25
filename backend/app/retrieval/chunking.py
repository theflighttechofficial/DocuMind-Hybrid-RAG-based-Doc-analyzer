"""Structure-aware chunking over parsed layout blocks.

Two granularities are produced:

* child chunks (~150 words) are what gets embedded, indexed and shown as
  evidence. They never cross a section boundary, never split a sentence,
  and a table is always its own chunk, kept whole.
* parent units are whole sections (split at paragraph boundaries past
  ~700 words). Retrieval finds children; the LLM reads their parent so it
  sees the surrounding context.

Each child keeps its pages, section path, character offsets into the page
text, and the bounding boxes of the blocks it came from, for highlighting.
The text that gets embedded is prefixed with "Document > Section > ..." so
the embedding knows where a passage sits; the displayed text is not.
"""
from __future__ import annotations

import re

from app.nlp.extraction import Block

CHILD_TARGET_WORDS = 150
CHILD_MAX_WORDS = 220
PARENT_MAX_WORDS = 700

_ABBREV = re.compile(r"\b(?:e\.g|i\.e|et al|etc|vs|Fig|Eq|No|Dr|Mr|Mrs|Ms|Prof|approx)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps decimals ("0.83"), abbreviations and
    citations intact; falls back to NLTK's punkt when available."""
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    merged: list[str] = []
    for s in sents:
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return [s.strip() for s in merged if s.strip()]


def _words(text: str) -> int:
    return len(text.split())


def _span(block: Block, start: int | None = None, end: int | None = None) -> dict:
    return {
        "page": block.page,
        "bbox": block.bbox,
        "char_start": block.char_start if start is None else start,
        "char_end": block.char_end if end is None else end,
    }


def _embed_header(title: str | None, path: list[str]) -> str:
    parts = [p for p in [title, *path] if p]
    return " > ".join(parts)


def _block_pieces(block: Block) -> list[tuple[str, dict]]:
    """A paragraph small enough stays whole; a long one is split into
    sentence runs of at most CHILD_MAX_WORDS, with page offsets per run."""
    if block.kind == "table" or _words(block.text) <= CHILD_MAX_WORDS:
        return [(block.text, _span(block))]
    pieces, cur, cur_start, cursor = [], [], None, 0
    for sent in split_sentences(block.text):
        pos = block.text.find(sent, cursor)
        pos = cursor if pos < 0 else pos
        cursor = pos + len(sent)
        if cur and _words(" ".join(cur)) + _words(sent) > CHILD_TARGET_WORDS:
            text = " ".join(cur)
            pieces.append((text, _span(block, block.char_start + cur_start, block.char_start + prev_end)))
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = pos
        cur.append(sent)
        prev_end = cursor
    if cur:
        pieces.append((" ".join(cur), _span(block, block.char_start + cur_start, block.char_start + prev_end)))
    return pieces


def build_chunks(blocks: list[Block], title: str | None) -> tuple[list[dict], list[dict]]:
    """Returns (parents, children). Children reference parents by index."""
    # 1. sections: consecutive non-heading blocks sharing a section path
    sections: list[dict] = []
    for b in blocks:
        if b.kind == "heading":
            continue
        path = list(b.section_path or [])
        if sections and sections[-1]["path"] == path:
            sections[-1]["blocks"].append(b)
        else:
            sections.append({"path": path, "blocks": [b]})

    parents: list[dict] = []
    children: list[dict] = []

    for sec in sections:
        path = sec["path"]
        header = _embed_header(title, path)

        # 2. parents: the section, split at paragraph boundaries if very long
        groups, cur = [], []
        for b in sec["blocks"]:
            if cur and sum(_words(x.text) for x in cur) + _words(b.text) > PARENT_MAX_WORDS:
                groups.append(cur)
                cur = []
            cur.append(b)
        if cur:
            groups.append(cur)

        for group in groups:
            parent_idx = len(parents)
            parents.append({
                "section_path": path,
                "page_start": group[0].page,
                "page_end": group[-1].page,
                "text": "\n\n".join(b.text for b in group),
            })

            # 3. children: pack paragraph pieces up to the target size; tables
            #    alone. Never merge across the section (we're inside one).
            buf_text: list[str] = []
            buf_spans: list[dict] = []

            def flush():
                if not buf_text:
                    return
                text = "\n\n".join(buf_text)
                children.append(_child(text, list(buf_spans), path, header, parent_idx, "text"))
                buf_text.clear()
                buf_spans.clear()

            for b in group:
                if b.kind == "table":
                    flush()
                    children.append(_child(b.text, [_span(b)], path, header, parent_idx, "table"))
                    continue
                for text, span in _block_pieces(b):
                    if buf_text and _words(" ".join(buf_text)) + _words(text) > CHILD_TARGET_WORDS:
                        flush()
                    buf_text.append(text)
                    buf_spans.append(span)
            flush()

    for i, c in enumerate(children):
        c["chunk_index"] = i
    return parents, children


def _child(text, spans, path, header, parent_idx, kind) -> dict:
    return {
        "text": text,
        "embed_text": f"{header}\n{text}" if header else text,
        "kind": kind,
        "page": spans[0]["page"],
        "page_end": spans[-1]["page"],
        "section": path[-1] if path else None,
        "section_path": path,
        "char_start": spans[0]["char_start"],
        "char_end": spans[-1]["char_end"],
        "spans": spans,  # [{page, bbox, char_start, char_end}] per source block piece
        "parent_idx": parent_idx,
    }
