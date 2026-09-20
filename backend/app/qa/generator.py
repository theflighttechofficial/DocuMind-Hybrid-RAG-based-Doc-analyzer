"""Generates grounded answers. Enforces the 'not found' behavior when
retrieval doesn't surface enough evidence, and refuses to let the LLM
answer from outside knowledge.
"""
from app.qa.llm_client import generate, generate_stream
from app.qa.context_builder import build_context, build_history

NOT_FOUND_MESSAGE = (
    "I couldn't find sufficient information in the uploaded document to answer this question."
)

# Below this best-match relevance score, we don't even bother calling the LLM —
# the retrieval itself signals there's nothing relevant.
MIN_RELEVANCE = 0.18

SYSTEM_INSTRUCTION = """You are DocuMind, a document question-answering assistant.

STRICT RULES:
1. Answer ONLY using the passages provided in the context. Never use outside knowledge.
2. If the passages do not contain enough information to answer, respond with EXACTLY:
   "I couldn't find sufficient information in the uploaded document to answer this question."
3. Do not speculate, guess, or fill gaps with general knowledge, even if the question seems easy.
4. Keep answers concise and directly responsive to the question.
5. When useful, refer to passages by number (e.g., "as shown in Passage 2").
"""


def _build_prompt(question: str, evidence: list, history_messages: list | None, show_document: bool) -> str:
    context = build_context(evidence, show_document=show_document)
    history = build_history(history_messages) if history_messages else ""

    prompt_parts = []
    if history:
        prompt_parts.append(f"Conversation so far:\n{history}\n")
    prompt_parts.append(f"Document passages:\n{context}\n")
    prompt_parts.append(f"Question: {question}\n\nAnswer using only the passages above.")
    return "\n".join(prompt_parts)


def _build_evidence_out(evidence: list, show_document: bool) -> list:
    evidence_out = []
    for item in evidence:
        chunk = item["chunk"]
        evidence_item = {
            "page": chunk.page_number,
            "section": chunk.section,
            "text": chunk.text[:600],
            "relevance": item["score"],
        }
        if show_document:
            evidence_item["document_id"] = chunk.document_id
            evidence_item["document_name"] = chunk.document.name if chunk.document else None
        evidence_out.append(evidence_item)
    return evidence_out


def answer_question(
    question: str, evidence: list, history_messages: list | None = None, show_document: bool = False
) -> dict:
    """Returns {"answer": str, "grounded": bool, "evidence": [...]}.

    show_document: when True (workspace/cross-document QA), tag each evidence item
    with the document it came from, and let the LLM see document labels in context.
    """
    if not evidence or max((e["score"] for e in evidence), default=0) < MIN_RELEVANCE:
        return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []}

    prompt = _build_prompt(question, evidence, history_messages, show_document)

    try:
        answer_text = generate(prompt, system_instruction=SYSTEM_INSTRUCTION, temperature=0.1)
    except RuntimeError:
        return {
            "answer": "The answer service is temporarily unavailable. Please try again shortly.",
            "grounded": False,
            "evidence": [],
        }

    if not answer_text:
        return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []}

    is_not_found = NOT_FOUND_MESSAGE.lower() in answer_text.lower()
    evidence_out = [] if is_not_found else _build_evidence_out(evidence, show_document)

    return {"answer": answer_text, "grounded": not is_not_found, "evidence": evidence_out}


def stream_answer_question(
    question: str, evidence: list, history_messages: list | None = None, show_document: bool = False
):
    """Generator variant of answer_question for the streaming /ask endpoint.

    Yields ("delta", text_chunk) events as tokens arrive, then a final
    ("done", {"answer", "grounded", "evidence"}) event with the same shape
    answer_question() returns — so callers can persist/display it identically.
    """
    if not evidence or max((e["score"] for e in evidence), default=0) < MIN_RELEVANCE:
        yield ("delta", NOT_FOUND_MESSAGE)
        yield ("done", {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []})
        return

    prompt = _build_prompt(question, evidence, history_messages, show_document)

    full_text = ""
    try:
        for chunk_text in generate_stream(prompt, system_instruction=SYSTEM_INSTRUCTION, temperature=0.1):
            full_text += chunk_text
            yield ("delta", chunk_text)
    except RuntimeError:
        fallback = "The answer service is temporarily unavailable. Please try again shortly."
        yield ("delta", fallback)
        yield ("done", {"answer": fallback, "grounded": False, "evidence": []})
        return

    answer_text = full_text.strip()
    if not answer_text:
        yield ("done", {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []})
        return

    is_not_found = NOT_FOUND_MESSAGE.lower() in answer_text.lower()
    evidence_out = [] if is_not_found else _build_evidence_out(evidence, show_document)

    yield ("done", {"answer": answer_text, "grounded": not is_not_found, "evidence": evidence_out})
