# app/backend/services/embeddings.py
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.backend.core.config import get_settings

_settings = get_settings()

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Ленивая загрузка и кэш модели эмбеддингов по имени из .env."""
    return SentenceTransformer(_settings.EMBEDDING_MODEL)

def embed(texts: list[str]) -> np.ndarray:
    """
    Возвращает L2-нормированные эмбеддинги (float32), форма (N, D).
    Ничего не требует кроме списка строк.
    """
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")