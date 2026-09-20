"""Builds the LLM context window from retrieved evidence, and formats
conversation history for follow-up-question resolution (coreference).
"""


def build_context(evidence: list, show_document: bool = False) -> str:
    """evidence: list of {"chunk": Chunk, "score": float}.

    show_document: prefix each passage with its source document name — used for
    workspace (cross-document) QA so the LLM can attribute passages correctly.
    """
    blocks = []
    for i, item in enumerate(evidence, start=1):
        chunk = item["chunk"]
        section = f" · {chunk.section}" if chunk.section else ""
        doc_label = f" — {chunk.document.name}" if show_document and chunk.document else ""
        blocks.append(
            f"[Passage {i}{doc_label} — Page {chunk.page_number}{section}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_history(messages: list, max_turns: int = 6) -> str:
    """messages: list of Message ORM rows, oldest first."""
    recent = messages[-max_turns:]
    lines = []
    for m in recent:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)
