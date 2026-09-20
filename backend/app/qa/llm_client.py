"""Thin wrapper around the Gemini API. Centralized so QA, summarization, and
comparison all go through one place, and so failures degrade gracefully.
"""
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client_configured = False


def _ensure_configured():
    global _client_configured
    if _client_configured:
        return
    import google.generativeai as genai
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to backend/.env (see .env.example)."
        )
    genai.configure(api_key=settings.gemini_api_key)
    _client_configured = True


def generate(prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> str:
    """Calls Gemini and returns plain text. Raises RuntimeError on failure —
    callers must catch this and surface the 'answer service unavailable' message.
    """
    import google.generativeai as genai
    settings = get_settings()

    try:
        _ensure_configured()
        model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=system_instruction,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return (response.text or "").strip()
    except Exception as exc:
        logger.exception("Gemini call failed")
        raise RuntimeError(f"LLM generation failed: {exc}") from exc


def generate_stream(prompt: str, system_instruction: str | None = None, temperature: float = 0.2):
    """Calls Gemini's streaming API and yields plain-text chunks as they arrive.
    Raises RuntimeError on failure — callers must catch this the same way as generate().
    """
    import google.generativeai as genai
    settings = get_settings()

    try:
        _ensure_configured()
        model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=system_instruction,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        logger.exception("Gemini streaming call failed")
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
