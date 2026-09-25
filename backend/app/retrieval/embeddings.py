"""Local sentence-transformers embedding model, loaded once and reused."""
import numpy as np

from app.core.config import get_settings
from app.core.model_loader import load_model


def get_embedder():
    from sentence_transformers import SentenceTransformer
    settings = get_settings()
    names = [settings.embedding_model]
    if settings.embedding_fallback_model not in names:
        names.append(settings.embedding_fallback_model)
    return load_model("embedder", [(n, lambda n=n: SentenceTransformer(n, device="cpu")) for n in names])


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
