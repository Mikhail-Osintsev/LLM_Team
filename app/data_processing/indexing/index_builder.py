import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from app.backend.services.embeddings import embed


def _normalize_inplace(x: np.ndarray) -> np.ndarray:
    """L2-нормализация эмбеддингов для cosine-sim через Inner Product."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (n, dim), got {x.shape}")
    # faiss.normalize_L2 работает in-place
    faiss.normalize_L2(x)
    return x


def build_faiss(chunks: list[str], index_dir: str) -> None:
    """Старая функция для обратной совместимости"""
    os.makedirs(index_dir, exist_ok=True)
    vecs = embed(chunks).astype("float32")
    if len(chunks) == 0:
        raise ValueError("No chunks provided to build_faiss")
    if vecs.size == 0:
        raise ValueError("Embedding model returned empty vectors")
    vecs = _normalize_inplace(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
    index.add(vecs)

    print(f"[FAISS] Built index: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "store.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks}, f)


def build_faiss_with_metadata(chunks_data: List[Dict[str, Any]], index_dir: str) -> None:
    """
    Строит FAISS индекс с метаданными для каждого чанка
    chunks_data: список словарей с полями text, page_number, book_name, filename
    """
    os.makedirs(index_dir, exist_ok=True)

    # Для E5 важно: документы эмбеддим с префиксом `passage:`,
    # но в store.pkl сохраняем "чистый" текст без префикса.
    texts_for_embed = [f"passage: {chunk['text']}" for chunk in chunks_data]
    texts_raw = [chunk["text"] for chunk in chunks_data]

    if len(chunks_data) == 0:
        raise ValueError("chunks_data is empty — nothing to index")

    vecs = embed(texts_for_embed).astype("float32")
    if vecs.size == 0:
        raise ValueError("Embedding model returned empty vectors")
    vecs = _normalize_inplace(vecs)

    dim = vecs.shape[1]

    # Создаем и заполняем индекс
    index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
    index.add(vecs)

    print(f"[FAISS] Built index with metadata: {index.ntotal} vectors, dim={dim}")

    # Сохраняем индекс
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    # Сохраняем метаданные
    metadata = {
        "chunks": texts_raw,
        "metadata": [
            {
                "page_number": chunk["page_number"],
                "book_name": chunk["book_name"],
                "filename": chunk["filename"]
            }
            for chunk in chunks_data
        ]
    }

    with open(os.path.join(index_dir, "store.pkl"), "wb") as f:
        pickle.dump(metadata, f)