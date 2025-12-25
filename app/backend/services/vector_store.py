import os                                 # работа с путями
import pickle                             # сохранение метаданных
import faiss                              # FAISS индекс
import numpy as np                        # массивы
from typing import List, Dict, Any, Tuple
from app.backend.core.config import get_settings  # настройки

_settings = get_settings()                # грузим настройки


class FaissStore:                         # класс-обёртка над FAISS
    def __init__(self, index_path: str, meta_path: str):  # конструктор принимает пути
        self.index_path = index_path                      # путь к index.faiss
        self.meta_path = meta_path                        # путь к store.pkl
        self.index = None                                 # тут будет индекс
        self.chunks: list[str] = []                       # тут будут тексты чанков
        self.metadata: List[Dict[str, Any]] = []          # метаданные для каждого чанка

    def load(self) -> None:                               # метод загрузки индекса
        # Явно проверяем, что файлы индекса существуют — иначе будет понятная ошибка
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index file not found: {self.index_path}. "
                f"You need to build the index first or set INDEX_DIR correctly."
            )
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"FAISS metadata file not found: {self.meta_path}. "
                f"You need to build the index first or set INDEX_DIR correctly."
            )

        self.index = faiss.read_index(self.index_path)    # читаем FAISS с диска
        with open(self.meta_path, "rb") as f:             # открываем метаданные
            meta = pickle.load(f)                         # читаем словарь
            self.chunks = meta["chunks"]                  # берём список чанков
            # Загружаем метаданные если они есть
            self.metadata = meta.get("metadata", [])

    def _ensure_loaded(self) -> None:
        """Ленивая загрузка индекса перед поиском."""
        if self.index is None:
            self.load()

    def search(self, q_vec: np.ndarray, k: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Поиск по эмбеддингу запроса
        Возвращает список кортежей (текст, скор, метаданные)
        """
        self._ensure_loaded()
        scores, idx = self.index.search(q_vec, k)         # обращаемся к FAISS
        hits = []                                         # сюда сложим результаты
        for j, i in enumerate(idx[0]):
            if i == -1:
                continue

            # Получаем метаданные если они есть
            chunk_metadata = {}
            if self.metadata and i < len(self.metadata):
                chunk_metadata = self.metadata[i]

            hits.append((
                self.chunks[i],
                float(scores[0][j]),
                chunk_metadata
            ))
        return hits

    def search_legacy(self, q_vec: np.ndarray, k: int = 4):
        """Старый метод для обратной совместимости, возвращает только (текст, скор)"""
        self._ensure_loaded()
        scores, idx = self.index.search(q_vec, k)
        hits = []
        for j, i in enumerate(idx[0]):
            if i == -1:
                continue
            hits.append((self.chunks[i], float(scores[0][j])))
        return hits


store = FaissStore(
    index_path=os.path.join(_settings.INDEX_DIR, "index.faiss"),
    meta_path=os.path.join(_settings.INDEX_DIR, "store.pkl"),
)