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
MIN_CHUNK_WORDS = 40  # shorter children (titles, lone headings) merge into the next

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

    children = _merge_short_children(children, parents)
    for i, c in enumerate(children):
        c["chunk_index"] = i
    return parents, children


def _merge_short_children(children: list[dict], parents: list[dict]) -> list[dict]:
    """A child under MIN_CHUNK_WORDS (a title line, a lone heading-like
    paragraph, a one-sentence lead-in to a table) is folded into the next
    child, so it's retrieved together with the content it introduces. A
    short last child folds into the previous one instead. Tables are never
    the short piece that moves, so a table chunk stays whole."""
    out: list[dict] = []
    carry: dict | None = None
    for i, c in enumerate(children):
        if carry is not None:
            c = _join(carry, c, parents, into_parent_of=c)
            carry = None
        nxt = children[i + 1] if i + 1 < len(children) else None
        prev = out[-1] if out else None
        if c["kind"] == "table" or _words(c["text"]) >= MIN_CHUNK_WORDS:
            out.append(c)
        elif nxt is not None and nxt["section_path"] == c["section_path"]:
            carry = c  # lead-in: attach to the content that follows in its section
        elif prev is not None and prev["section_path"] == c["section_path"]:
            out[-1] = _join(prev, c, parents, into_parent_of=prev)  # section's tail
        elif nxt is not None and _title_like(c["text"]):
            carry = c  # a lone title/heading line: attach to what it introduces
        else:
            out.append(c)  # a complete short section stays its own chunk
    return out


def _title_like(text: str) -> bool:
    return _words(text) <= 12 or not text.rstrip().endswith((".", "?", "!"))


def _join(first: dict, second: dict, parents: list[dict], into_parent_of: dict) -> dict:
    """Concatenates two children; the result keeps into_parent_of's section
    and parent. The moved text is also added to that parent, so the LLM (which
    reads parents) sees it too."""
    keep = into_parent_of
    moved = second if keep is first else first
    parent = parents[keep["parent_idx"]]
    if moved["text"] not in parent["text"]:
        parent["text"] = (moved["text"] + "\n\n" + parent["text"]) if moved is first else (parent["text"] + "\n\n" + moved["text"])
        parent["page_start"] = min(parent["page_start"], moved["page"])
        parent["page_end"] = max(parent["page_end"], moved["page_end"])
    text = first["text"] + "\n\n" + second["text"]
    header = keep["embed_text"][: len(keep["embed_text"]) - len(keep["text"])].rstrip("\n")
    return {
        **keep,
        "text": text,
        "embed_text": f"{header}\n{text}" if header else text,
        "kind": "table" if "table" in (first["kind"], second["kind"]) else "text",
        "page": first["page"],
        "page_end": second["page_end"],
        "char_start": first["char_start"],
        "char_end": second["char_end"],
        "spans": first["spans"] + second["spans"],
    }


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
