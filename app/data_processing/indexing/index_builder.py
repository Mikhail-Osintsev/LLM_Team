import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from app.backend.services.embeddings import embed


def build_faiss(chunks: list[str], index_dir: str) -> None:
    """Старая функция для обратной совместимости"""
    os.makedirs(index_dir, exist_ok=True)
    vecs = embed(chunks).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "store.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks}, f)


def build_faiss_with_metadata(chunks_data: List[Dict[str, Any]], index_dir: str) -> None:
    """
    Строит FAISS индекс с метаданными для каждого чанка
    chunks_data: список словарей с полями text, page_number, book_name, filename
    """
    os.makedirs(index_dir, exist_ok=True)

    # Извлекаем тексты для эмбеддингов
    texts = [chunk["text"] for chunk in chunks_data]

    # Создаем эмбеддинги
    vecs = embed(texts).astype("float32")
    dim = vecs.shape[1]

    # Создаем и заполняем индекс
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Сохраняем индекс
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    # Сохраняем метаданные
    metadata = {
        "chunks": texts,
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