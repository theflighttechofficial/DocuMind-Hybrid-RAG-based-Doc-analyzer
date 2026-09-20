"""Hierarchical summarization: chunk -> section summaries -> combined summary.
Avoids blindly stuffing the whole document into one LLM call.
"""
from app.qa.llm_client import generate

# _chunk_text_by_chars only splits *between* paragraphs — a document with no
# blank-line breaks (some plain-text exports, OCR output) comes back as one
# unbounded "section", which would otherwise go straight into a single LLM
# call with no size limit. Past this size, extractively pre-filter down to
# the most representative sentences before summarizing — see
# _extractive_prefilter. Normal, well-formed sections never hit this path,
# so ordinary documents are unaffected.
_MAX_SECTION_CHARS_FOR_LLM = 12_000

_SECTION_SYSTEM = (
    "You summarize a portion of a larger document accurately and concisely. "
    "Do not add information that is not present in the text."
)

_FINAL_SYSTEM = """You produce a structured summary of a document from partial section
summaries. Respond in this exact format with plain text, no markdown headers:

EXECUTIVE SUMMARY:
<2-4 sentences>

KEY POINTS:
- <point>
- <point>

IMPORTANT FINDINGS:
- <finding>

IMPORTANT TERMS:
- <term>: <one-line definition or role in the document>

CONCLUSIONS:
<1-3 sentences>
"""

_LENGTH_GUIDANCE = {
    "brief": "Keep the executive summary to 2 sentences and each list to at most 3 items.",
    "standard": "Keep the executive summary to 3-4 sentences and each list to 4-6 items.",
    "detailed": "Write a thorough executive summary (5-7 sentences) and each list with 6-10 items.",
}


def _chunk_text_by_chars(text: str, max_chars: int = 6000) -> list[str]:
    parts, current = [], []
    current_len = 0
    for para in text.split("\n\n"):
        if current_len + len(para) > max_chars and current:
            parts.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)
    if current:
        parts.append("\n\n".join(current))
    return parts or [text]


def _extractive_prefilter(text: str, max_sentences: int = 40) -> str:
    """Centroid-based extractive filtering: embeds every sentence (reusing the
    same sentence-transformers model already loaded for retrieval — no second
    model), keeps the ones closest to the section's own mean embedding (i.e.
    most representative of the section as a whole), and returns them in their
    original order so the LLM still gets coherent, sequential text.
    """
    import numpy as np
    from nltk import sent_tokenize
    from app.retrieval.embeddings import embed_texts

    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if len(sentences) <= max_sentences:
        return text

    vectors = embed_texts(sentences)
    centroid = vectors.mean(axis=0)
    similarities = vectors @ centroid
    keep_idx = set(np.argsort(-similarities)[:max_sentences])
    return " ".join(s for i, s in enumerate(sentences) if i in keep_idx)


def summarize_document(full_text: str, length: str = "standard") -> dict:
    sections = _chunk_text_by_chars(full_text)

    section_summaries = []
    for section_text in sections:
        if not section_text.strip():
            continue
        if len(section_text) > _MAX_SECTION_CHARS_FOR_LLM:
            section_text = _extractive_prefilter(section_text)
        try:
            summary = generate(
                f"Summarize this document excerpt in 3-5 sentences, preserving key facts:\n\n{section_text}",
                system_instruction=_SECTION_SYSTEM,
                temperature=0.2,
            )
            section_summaries.append(summary)
        except RuntimeError:
            continue

    if not section_summaries:
        return {
            "raw": "The answer service is temporarily unavailable. Please try again shortly.",
            "available": False,
        }

    combined_input = "\n\n".join(f"Section {i+1} summary: {s}" for i, s in enumerate(section_summaries))
    guidance = _LENGTH_GUIDANCE.get(length, _LENGTH_GUIDANCE["standard"])

    try:
        final = generate(
            f"{guidance}\n\nSection summaries:\n{combined_input}",
            system_instruction=_FINAL_SYSTEM,
            temperature=0.2,
        )
    except RuntimeError:
        return {
            "raw": "The answer service is temporarily unavailable. Please try again shortly.",
            "available": False,
        }

    return {"raw": final, "available": True, "section_count": len(section_summaries)}
