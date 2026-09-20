"""Per-document FAISS index, persisted to disk under VECTOR_DIR/<document_id>.index.

Since normalized embeddings are used, inner product == cosine similarity.
"""
import os
import faiss
import numpy as np

from app.core.config import get_settings

settings = get_settings()
os.makedirs(settings.vector_dir, exist_ok=True)


def _index_path(document_id: str) -> str:
    return os.path.join(settings.vector_dir, f"{document_id}.index")


def build_index(document_id: str, embeddings: np.ndarray) -> None:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if embeddings.shape[0] > 0:
        index.add(embeddings)
    faiss.write_index(index, _index_path(document_id))


def load_index(document_id: str):
    path = _index_path(document_id)
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)


def search(document_id: str, query_embedding: np.ndarray, top_k: int = 6):
    """Returns (scores, indices) arrays, or (None, None) if no index exists."""
    index = load_index(document_id)
    if index is None or index.ntotal == 0:
        return None, None
    query = np.expand_dims(query_embedding, axis=0)
    scores, indices = index.search(query, min(top_k, index.ntotal))
    return scores[0], indices[0]


def delete_index(document_id: str) -> None:
    path = _index_path(document_id)
    if os.path.exists(path):
        os.remove(path)
