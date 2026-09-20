"""Central app configuration, loaded from environment variables / .env."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    top_k_retrieval: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
