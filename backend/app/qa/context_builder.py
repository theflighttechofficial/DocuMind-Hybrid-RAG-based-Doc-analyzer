"""Builds the LLM context window from retrieved evidence, and formats
conversation history for follow-up-question resolution (coreference).
"""
from app.core.config import get_settings


def parent_window(parent_text: str, child_text: str, max_words: int) -> str:
    """The child plus as much of the parent around it as fits in max_words,
    split evenly before and after. Falls back to the child alone when it can't
    be located in the parent."""
    parent_words = parent_text.split()
    child_words = child_text.split()
    if len(parent_words) <= max_words:
        return parent_text
    if len(child_words) >= max_words:
        return " ".join(child_words[:max_words])
    probe = child_words[: min(8, len(child_words))]
    start = next((i for i in range(len(parent_words) - len(probe) + 1)
                  if parent_words[i:i + len(probe)] == probe), None)
    if start is None:
        return child_text
    room = max_words - len(child_words)
    before = min(room // 2, start)
    end = min(len(parent_words), start + len(child_words) + room - before)
    begin = max(0, end - max_words)
    text = " ".join(parent_words[begin:end])
    return ("… " if begin > 0 else "") + text + (" …" if end < len(parent_words) else "")


def build_context(evidence: list, show_document: bool = False) -> str:
    """evidence: list of {"chunk": Chunk, "score": float}, already sorted by
    relevance; passage i is labelled [i] so the model can cite it inline.

    Parent-child retrieval: each retrieved child chunk is widened with the
    surrounding text of its parent section, up to settings.llm_passage_max_words,
    so the LLM sees context around the hit without overflowing a small local
    model's context window.

    show_document: prefix each passage with its source document name — used for
    workspace (cross-document) QA so the LLM can attribute passages correctly.
    """
    max_words = get_settings().llm_passage_max_words
    blocks = []
    for i, item in enumerate(evidence, start=1):
        chunk = item["chunk"]
        path = " > ".join(chunk.section_path) if getattr(chunk, "section_path", None) else (chunk.section or "")
        section = f" · {path}" if path else ""
        doc_label = f" — {chunk.document.name}" if show_document and chunk.document else ""
        pages = f"Page {chunk.page_number}" + (
            f"-{chunk.page_end}" if getattr(chunk, "page_end", None) and chunk.page_end != chunk.page_number else ""
        )
        parent = getattr(chunk, "parent", None)
        body = parent_window(parent.text, chunk.text, max_words) if parent is not None else chunk.text
        blocks.append(f"[{i}]{doc_label} ({pages}{section})\n{body}")
    return "\n\n".join(blocks)


def build_history(messages: list, max_turns: int = 6) -> str:
    """messages: list of Message ORM rows, oldest first."""
    recent = messages[-max_turns:]
    lines = []
    for m in recent:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)
