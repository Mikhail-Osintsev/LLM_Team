import numpy as np                              # массивы
from app.backend.services.embeddings import embed     # функция эмбеддинга
from app.backend.services.vector_store import store  # глобальный FAISS стор

def retrieve(query: str, top_k: int = 4):            # главная функция ретривера
    qv = embed([query])                               # считаем вектор вопроса
    # нормировка (если индекс строили на нормированных векторах)
    norms = np.linalg.norm(qv, axis=1, keepdims=True) # считаем длину вектора
    qv = qv / np.where(norms == 0, 1, norms)          # нормируем безопасно
    return store.search(qv.astype("float32"), k=top_k) # ищем в FAISS и возвращаем (текст, скор)