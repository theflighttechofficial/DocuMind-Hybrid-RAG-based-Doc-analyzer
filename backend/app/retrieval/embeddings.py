"""Local sentence-transformers embedding model, loaded once and reused."""
from functools import lru_cache
import numpy as np

from app.core.config import get_settings


@lru_cache
def get_embedder():
    from sentence_transformers import SentenceTransformer
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_embedder()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
