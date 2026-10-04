"""Runtime configuration and model status, so the UI and the eval runner can
record exactly which models and settings produced a result."""
from fastapi import APIRouter

from app.core.config import get_settings
from app.core.model_loader import loaded_models_info
from app.nlp.extraction import ocr_engine
from app.qa.llm_client import active_backend

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/config")
def system_config():
    s = get_settings()
    return {
        "settings": s.model_dump(exclude={"gemini_api_key", "jwt_secret", "database_url"}),
        "ocr_engine": ocr_engine(),
        "llm": active_backend(),
        "loaded_models": loaded_models_info(),
    }
