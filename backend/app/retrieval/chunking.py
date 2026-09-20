"""Splits page-scoped text into overlapping word chunks, preserving page and
a best-effort section label for citation purposes.
"""
import re
from typing import List, Tuple, Dict

_HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Za-z0-9 ,'&/-]{2,60}|"
    r"\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,'&/-]{2,60}|"
    r"(?:Chapter|Section|Part)\s+\d+.*)$"
)


def _detect_section(line: str) -> str | None:
    line = line.strip()
    if 3 <= len(line) <= 80 and _HEADING_RE.match(line) and not line.endswith((".", ",")):
        return line
    return None


def chunk_document(
    pages: List[Tuple[int, str]], chunk_size_words: int = 220, overlap_words: int = 40
) -> List[Dict]:
    """Returns a list of dicts: {page, section, chunk_index, text}."""
    chunks = []
    current_section = None
    chunk_index = 0

    for page_number, page_text in pages:
        for line in page_text.splitlines()[:5]:
            section = _detect_section(line)
            if section:
                current_section = section
                break

        words = page_text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({
                    "page": page_number,
                    "section": current_section,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                })
                chunk_index += 1
            if end == len(words):
                break
            start = end - overlap_words

    return chunks
