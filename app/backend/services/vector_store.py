import os                                 # работа с путями
import pickle                             # сохранение метаданных
import faiss                              # FAISS индекс
import numpy as np                        # массивы
from app.backend.core.config import get_settings  # настройки

_settings = get_settings()                # грузим настройки

class FaissStore:                         # класс-обёртка над FAISS
    def __init__(self, index_path: str, meta_path: str):  # конструктор принимает пути
        self.index_path = index_path                      # путь к index.faiss
        self.meta_path = meta_path                        # путь к store.pkl
        self.index = None                                 # тут будет индекс
        self.chunks: list[str] = []                       # тут будут тексты чанков

    def load(self) -> None:                               # метод загрузки индекса
        self.index = faiss.read_index(self.index_path)    # читаем FAISS с диска
        with open(self.meta_path, "rb") as f:             # открываем метаданные
            meta = pickle.load(f)                         # читаем словарь
            self.chunks = meta["chunks"]                  # берём список чанков

    def search(self, q_vec: np.ndarray, k: int = 4):      # поиск по эмбеддингу запроса
        scores, idx = self.index.search(q_vec, k)         # обращаемся к FAISS
        hits = []                                         # сюда сложим (текст, скор)
        for j, i in enumerate(idx[0]):                   
            if i == -1:                               
                continue                                  
            hits.append((self.chunks[i], float(scores[0][j]))) 
        return hits                                     

store = FaissStore(
    index_path=os.path.join(_settings.INDEX_DIR, "index.faiss"),
    meta_path=os.path.join(_settings.INDEX_DIR, "store.pkl"),
)