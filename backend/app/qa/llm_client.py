"""Single entry point for every LLM call (QA, summarization, comparison,
settings check), so the backend can be switched in one place.

Default provider is a local model served by Ollama on this machine
(http://127.0.0.1:11434): no document text leaves the computer. The model is
DocuMind's own fine-tune of Qwen2.5 (see backend/finetune/README.md), with
the stock Ollama model as fallback when the fine-tune isn't installed.

The Gemini code path is kept only for comparison runs and is used solely
when LLM_PROVIDER=gemini is set explicitly; it is never a silent fallback.
"""
import json
import logging

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# A stalled remote call once held a request for 10+ minutes. Local CPU
# generation is slower than an API, so the local limit is more generous.
LLM_TIMEOUT_S = 60
LOCAL_TIMEOUT_S = 300

_resolved_model: str | None = None


# ----------------------------------------------------------------- Ollama

def _ollama_url(path: str) -> str:
    return get_settings().ollama_url.rstrip("/") + path


def ollama_models() -> list[str]:
    r = requests.get(_ollama_url("/api/tags"), timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def resolve_local_model() -> str:
    """The configured model if Ollama has it, else the fallback model."""
    global _resolved_model
    if _resolved_model:
        return _resolved_model
    s = get_settings()
    try:
        available = ollama_models()
    except Exception as exc:
        raise RuntimeError(
            f"Local LLM server not reachable at {s.ollama_url} ({exc}). Start it with `ollama serve`."
        ) from exc
    for name in (s.ollama_model, s.ollama_fallback_model):
        if name in available or f"{name}:latest" in available:
            if name != s.ollama_model:
                logger.warning("Ollama model %s not installed; using fallback %s", s.ollama_model, name)
            _resolved_model = name
            return name
    raise RuntimeError(
        f"Neither {s.ollama_model} nor {s.ollama_fallback_model} is installed in Ollama. "
        f"Run backend/finetune/build_ollama_model.ps1 or `ollama pull {s.ollama_fallback_model}`."
    )


def reset_model_cache():
    global _resolved_model
    _resolved_model = None


def _ollama_payload(prompt, system_instruction, temperature, stream):
    s = get_settings()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    options = {"temperature": temperature, "num_ctx": s.ollama_num_ctx, "num_predict": s.llm_max_tokens}
    if s.ollama_num_thread:
        options["num_thread"] = s.ollama_num_thread
    return {
        "model": resolve_local_model(),
        "messages": messages,
        "stream": stream,
        "keep_alive": s.ollama_keep_alive,
        "options": options,
    }


def _ollama_generate(prompt, system_instruction, temperature) -> str:
    r = requests.post(_ollama_url("/api/chat"), json=_ollama_payload(prompt, system_instruction, temperature, False),
                      timeout=LOCAL_TIMEOUT_S)
    r.raise_for_status()
    return (r.json().get("message", {}).get("content") or "").strip()


def _ollama_stream(prompt, system_instruction, temperature):
    with requests.post(_ollama_url("/api/chat"), json=_ollama_payload(prompt, system_instruction, temperature, True),
                       timeout=LOCAL_TIMEOUT_S, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(event["error"])
            text = event.get("message", {}).get("content")
            if text:
                yield text
            if event.get("done"):
                break


# ----------------------------------------------------------------- Gemini (opt-in only)

def _gemini_model(system_instruction):
    import google.generativeai as genai
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
    genai.configure(api_key=s.gemini_api_key)
    return genai, genai.GenerativeModel(s.gemini_model, system_instruction=system_instruction)


def _gemini_generate(prompt, system_instruction, temperature) -> str:
    genai, model = _gemini_model(system_instruction)
    response = model.generate_content(
        prompt, generation_config=genai.types.GenerationConfig(temperature=temperature),
        request_options={"timeout": LLM_TIMEOUT_S},
    )
    return (response.text or "").strip()


def _gemini_stream(prompt, system_instruction, temperature):
    genai, model = _gemini_model(system_instruction)
    response = model.generate_content(
        prompt, generation_config=genai.types.GenerationConfig(temperature=temperature),
        stream=True, request_options={"timeout": LLM_TIMEOUT_S},
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ----------------------------------------------------------------- public API

def active_backend() -> dict:
    s = get_settings()
    if s.llm_provider == "gemini":
        return {"provider": "gemini", "model": s.gemini_model, "local": False}
    try:
        model = resolve_local_model()
    except RuntimeError as exc:
        return {"provider": "ollama", "model": None, "local": True, "error": str(exc)}
    return {"provider": "ollama", "model": model, "local": True, "fine_tuned": model == s.ollama_model}


def generate(prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> str:
    """Returns plain text. Raises RuntimeError on failure — callers must catch
    this and surface the 'answer service unavailable' message."""
    try:
        if get_settings().llm_provider == "gemini":
            return _gemini_generate(prompt, system_instruction, temperature)
        return _ollama_generate(prompt, system_instruction, temperature)
    except Exception as exc:
        logger.exception("LLM call failed")
        raise RuntimeError(f"LLM generation failed: {exc}") from exc


def generate_stream(prompt: str, system_instruction: str | None = None, temperature: float = 0.2):
    """Yields plain-text chunks as they arrive. Raises RuntimeError on failure."""
    try:
        if get_settings().llm_provider == "gemini":
            yield from _gemini_stream(prompt, system_instruction, temperature)
        else:
            yield from _ollama_stream(prompt, system_instruction, temperature)
    except Exception as exc:
        logger.exception("LLM streaming call failed")
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
