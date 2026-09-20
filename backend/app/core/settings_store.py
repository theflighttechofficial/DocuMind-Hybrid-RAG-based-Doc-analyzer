"""Lets the Settings page update GEMINI_API_KEY at runtime by rewriting
backend/.env and clearing the cached Settings object, instead of requiring a
server restart every time a key changes.
"""
import os

from app.core.config import get_settings

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")


def _upsert_env_var(key: str, value: str):
    lines = []
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")

    with open(_ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def update_gemini_api_key(new_key: str):
    _upsert_env_var("GEMINI_API_KEY", new_key)
    get_settings.cache_clear()


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 6:
        return "•" * len(key)
    return f"{'•' * (len(key) - 4)}{key[-4:]}"
