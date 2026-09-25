"""Generates grounded answers. Enforces the 'not found' behavior when
retrieval doesn't surface enough evidence, and refuses to let the LLM
answer from outside knowledge.

Evidence handling:
* passages are sorted by relevance (highest first) and numbered [1], [2]...
  in that order, so a passage's citation number is also its rank;
* the model must cite inline ([1], [2][3]); the citations are parsed and only
  cited passages are returned as evidence, each tagged with its number;
* if the answer cites nothing, evidence falls back to passages whose
  relevance clears settings.evidence_min_relevance (tuned on the eval set by
  scripts/tune_evidence_threshold.py) instead of every top_k passage.
"""
import re

from app.core.config import get_settings
from app.qa.llm_client import generate, generate_stream
from app.qa.context_builder import build_context, build_history

NOT_FOUND_MESSAGE = (
    "I couldn't find sufficient information in the uploaded document to answer this question."
)
UNAVAILABLE_MESSAGE = "The answer service is temporarily unavailable. Please try again shortly."

# Below this best-match retrieval score, we don't even bother calling the LLM —
# the retrieval itself signals there's nothing relevant.
MIN_RELEVANCE = 0.18

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")

# Safety net for the prompt rule below: strip a leading "Based on the provided
# documents," style preamble if the model still produces one.
_PREAMBLE = re.compile(
    r"^\s*(?:based on|according to|as (?:stated|shown|described) in)\s+(?:the\s+)?"
    r"(?:provided\s+|given\s+|above\s+)?(?:documents?|passages?|context|text|information)"
    r"(?:\s+provided)?(?:\s*\[\d+\])?\s*,\s*",
    re.I,
)


def system_instruction(multi_document: bool) -> str:
    source = "the documents" if multi_document else "the document"
    return f"""You are DocuMind, a document question-answering assistant.

STRICT RULES:
1. Answer ONLY using the numbered passages provided. Never use outside knowledge.
2. If the passages do not contain enough information to answer, respond with EXACTLY:
   "{NOT_FOUND_MESSAGE}"
3. Do not speculate, guess, or fill gaps with general knowledge, even if the question seems easy.
4. Keep answers concise and directly responsive to the question.
5. Cite every claim inline with the number of the passage that supports it, in square
   brackets right after the claim, e.g. "The payload is 4.5 kg [2]." Use several
   numbers when several passages support a claim ("[1][3]"). Only cite passages that
   actually support the claim.
6. Start with the answer itself. Do not open with phrases like "Based on the provided
   documents", "According to the passages" or "As shown in Passage 2". When you need
   to refer to the source, call it "{source}", never "the passages" or "the provided documents".
"""


def _build_prompt(question: str, evidence: list, history_messages: list | None, show_document: bool) -> str:
    context = build_context(evidence, show_document=show_document)
    history = build_history(history_messages) if history_messages else ""

    prompt_parts = []
    if history:
        prompt_parts.append(f"Conversation so far:\n{history}\n")
    prompt_parts.append(f"Numbered passages:\n{context}\n")
    prompt_parts.append(f"Question: {question}\n\nAnswer using only the passages above, citing them inline as [n].")
    return "\n".join(prompt_parts)


def evidence_item(chunk, score: float, show_document: bool = False) -> dict:
    item = {
        "chunk_id": chunk.id,
        "page": chunk.page_number,
        "page_end": getattr(chunk, "page_end", None) or chunk.page_number,
        "section": chunk.section,
        "section_path": getattr(chunk, "section_path", None) or ([chunk.section] if chunk.section else []),
        "kind": getattr(chunk, "kind", None) or "text",
        "text": chunk.text,
        "relevance": score,
        "spans": getattr(chunk, "spans_json", None) or [],
    }
    if show_document:
        item["document_id"] = chunk.document_id
        item["document_name"] = chunk.document.name if chunk.document else None
    return item


def sort_by_relevance(evidence: list) -> list:
    return sorted(evidence, key=lambda item: -item["score"])


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Citation numbers in order of first appearance, limited to 1..n_passages.
    Handles [1], [1, 3], [1][2] and ranges like [2-4]."""
    seen: list[int] = []
    for group in _CITATION_GROUP.findall(answer):
        for part in re.split(r"\s*,\s*", group):
            bounds = re.split(r"\s*[–-]\s*", part)
            lo, hi = int(bounds[0]), int(bounds[-1])
            for n in range(lo, min(hi, lo + 20) + 1):
                if 1 <= n <= n_passages and n not in seen:
                    seen.append(n)
    return seen


def select_evidence(answer: str, evidence: list, show_document: bool) -> list:
    """evidence must already be sorted by relevance (the order passages were
    numbered in). Returns cited passages, highest relevance first, each with its
    "citation" number; if nothing was cited, passages above the tuned threshold."""
    cited = set(parse_citations(answer, len(evidence)))
    if cited:
        chosen = [(i, item) for i, item in enumerate(evidence, start=1) if i in cited]
    else:
        threshold = get_settings().evidence_min_relevance
        chosen = [(i, item) for i, item in enumerate(evidence, start=1) if item["score"] >= threshold]
    out = []
    for number, item in chosen:
        ev = evidence_item(item["chunk"], item["score"], show_document)
        if "rerank_score" in item:
            ev["rerank_logit"] = round(item["rerank_score"], 3)
        ev["citation"] = number
        ev["cited"] = number in cited
        out.append(ev)
    return out


def clean_answer(answer: str) -> str:
    cleaned = _PREAMBLE.sub("", answer, count=1)
    if cleaned != answer and cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.strip()


def _retrieval_gate(evidence: list) -> bool:
    """True when the best passage clears MIN_RELEVANCE on the fused retrieval
    score (the scale that threshold was set on, before reranking existed)."""
    return bool(evidence) and max(e.get("retrieval_score", e["score"]) for e in evidence) >= MIN_RELEVANCE


def _finish(answer_text: str, evidence: list, show_document: bool) -> dict:
    if not answer_text:
        return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []}
    if NOT_FOUND_MESSAGE.lower() in answer_text.lower():
        return {"answer": answer_text, "grounded": False, "evidence": []}
    answer_text = clean_answer(answer_text)
    return {"answer": answer_text, "grounded": True, "evidence": select_evidence(answer_text, evidence, show_document)}


def answer_question(
    question: str, evidence: list, history_messages: list | None = None, show_document: bool = False
) -> dict:
    """Returns {"answer": str, "grounded": bool, "evidence": [...]}.

    show_document: when True (workspace/cross-document QA), tag each evidence item
    with the document it came from, and let the LLM see document labels in context.
    """
    if not _retrieval_gate(evidence):
        return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []}

    evidence = sort_by_relevance(evidence)
    prompt = _build_prompt(question, evidence, history_messages, show_document)

    try:
        answer_text = generate(prompt, system_instruction=system_instruction(show_document), temperature=0.1)
    except RuntimeError:
        return {"answer": UNAVAILABLE_MESSAGE, "grounded": False, "evidence": []}

    return _finish(answer_text, evidence, show_document)


def stream_answer_question(
    question: str, evidence: list, history_messages: list | None = None, show_document: bool = False
):
    """Generator variant of answer_question for the streaming /ask endpoint.

    Yields ("delta", text_chunk) events as tokens arrive, then a final
    ("done", {"answer", "grounded", "evidence"}) event with the same shape
    answer_question() returns — so callers can persist/display it identically.
    The final answer may differ from the concatenated deltas by a stripped
    preamble; clients should replace the streamed text with it.
    """
    if not _retrieval_gate(evidence):
        yield ("delta", NOT_FOUND_MESSAGE)
        yield ("done", {"answer": NOT_FOUND_MESSAGE, "grounded": False, "evidence": []})
        return

    evidence = sort_by_relevance(evidence)
    prompt = _build_prompt(question, evidence, history_messages, show_document)

    full_text = ""
    try:
        for chunk_text in generate_stream(prompt, system_instruction=system_instruction(show_document), temperature=0.1):
            full_text += chunk_text
            yield ("delta", chunk_text)
    except RuntimeError:
        yield ("delta", UNAVAILABLE_MESSAGE)
        yield ("done", {"answer": UNAVAILABLE_MESSAGE, "grounded": False, "evidence": []})
        return

    yield ("done", _finish(full_text.strip(), evidence, show_document))
