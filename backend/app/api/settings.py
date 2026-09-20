from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.core.settings_store import mask_key, update_gemini_api_key
from app.models.models import User
from app.models.schemas import SettingsOut, TestConnectionOut, UpdateGeminiKeyRequest

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsOut)
def get_app_settings(current_user: User = Depends(get_current_user)):
    settings = get_settings()
    return SettingsOut(
        gemini_api_key_set=bool(settings.gemini_api_key),
        gemini_api_key_masked=mask_key(settings.gemini_api_key),
        gemini_model=settings.gemini_model,
    )


@router.post("/gemini-key", response_model=SettingsOut)
def set_gemini_key(payload: UpdateGeminiKeyRequest, current_user: User = Depends(get_current_user)):
    key = payload.api_key.strip()
    if not key:
        raise HTTPException(400, "API key cannot be empty.")

    update_gemini_api_key(key)
    settings = get_settings()
    return SettingsOut(
        gemini_api_key_set=bool(settings.gemini_api_key),
        gemini_api_key_masked=mask_key(settings.gemini_api_key),
        gemini_model=settings.gemini_model,
    )


@router.post("/test-connection", response_model=TestConnectionOut)
def test_connection(current_user: User = Depends(get_current_user)):
    from app.qa.llm_client import generate

    try:
        text = generate("Reply with exactly one word: OK", temperature=0.0)
        return TestConnectionOut(ok=True, message=f"Connected — Gemini replied: {text[:80]!r}")
    except RuntimeError as exc:
        return TestConnectionOut(ok=False, message=str(exc))
