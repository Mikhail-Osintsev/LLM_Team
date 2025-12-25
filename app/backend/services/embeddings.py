# app/backend/services/embeddings.py

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.backend.core.config import get_settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Ленивая загрузка модели эмбеддингов (кэшируется на процесс)."""
    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Вернёт L2-нормированные эмбеддинги float32 формы (N, D).

    Подходит для cosine similarity в FAISS через IndexFlatIP.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")