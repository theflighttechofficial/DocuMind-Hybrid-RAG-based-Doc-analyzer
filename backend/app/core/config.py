"""Central app configuration, loaded from environment variables / .env.

Every heavy model sits behind a setting here, with a lighter fallback that is
used automatically if the primary model can't be loaded (download failure, not
enough RAM). Defaults were chosen for the development machine: 4-core
i7-1165G7, 16 GB RAM, no CUDA GPU. See README "Models" for sizes.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# TensorFlow is installed on this machine (tf-keras / Keras 3 conflict), and
# transformers will import it unless told not to. This project is PyTorch-only.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Used to sign JWTs. The default is fine for local dev only — set a real
    # secret via JWT_SECRET in backend/.env before deploying anywhere shared.
    jwt_secret: str = "dev-insecure-secret-change-me"

    database_url: str = "sqlite:///./storage/documind.db"
    upload_dir: str = "./storage/uploads"
    vector_dir: str = "./storage/vector_index"

    # --- parsing
    # "auto": tesseract if its binary is installed, else EasyOCR, else no OCR.
    ocr_engine: str = "auto"  # auto | tesseract | easyocr | none
    ocr_min_chars: int = 20  # pages with less extractable text than this get OCR

    # --- embeddings / retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    top_k_retrieval: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
