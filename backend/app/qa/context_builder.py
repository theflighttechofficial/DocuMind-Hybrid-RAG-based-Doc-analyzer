"""Builds the LLM context window from retrieved evidence, and formats
conversation history for follow-up-question resolution (coreference).
"""


def build_context(evidence: list, show_document: bool = False) -> str:
    """evidence: list of {"chunk": Chunk, "score": float}.

    Parent-child retrieval: each retrieved child chunk is replaced by its
    parent section so the LLM sees the surrounding context. When several
    children share a parent, the parent is included once and later passages
    point back to it.

    show_document: prefix each passage with its source document name — used for
    workspace (cross-document) QA so the LLM can attribute passages correctly.
    """
    blocks = []
    seen_parent: dict[str, int] = {}
    for i, item in enumerate(evidence, start=1):
        chunk = item["chunk"]
        path = " > ".join(chunk.section_path) if getattr(chunk, "section_path", None) else (chunk.section or "")
        section = f" · {path}" if path else ""
        doc_label = f" — {chunk.document.name}" if show_document and chunk.document else ""
        pages = f"Page {chunk.page_number}" + (
            f"-{chunk.page_end}" if getattr(chunk, "page_end", None) and chunk.page_end != chunk.page_number else ""
        )
        parent = getattr(chunk, "parent", None)
        if parent is not None and parent.id in seen_parent:
            body = f"(Same section as Passage {seen_parent[parent.id]}.) Most relevant part:\n{chunk.text}"
        elif parent is not None and len(parent.text) > len(chunk.text):
            seen_parent[parent.id] = i
            body = parent.text
        else:
            if parent is not None:
                seen_parent[parent.id] = i
            body = chunk.text
        blocks.append(f"[Passage {i}{doc_label} — {pages}{section}]\n{body}")
    return "\n\n".join(blocks)


def build_history(messages: list, max_turns: int = 6) -> str:
    """messages: list of Message ORM rows, oldest first."""
    recent = messages[-max_turns:]
    lines = []
    for m in recent:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)
