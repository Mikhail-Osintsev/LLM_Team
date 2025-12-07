from fastapi import FastAPI                                # импортируем FastAPI
from pydantic import BaseModel                             # валидация входа/выхода
from app.backend.services.vector_store import store        # доступ к FAISS
from app.backend.core.config import get_settings           # настройки
from app.backend.core.rag_graph import run_rag
import os
from typing import List



app = FastAPI(title="Book RAG API")                        # создаём приложение FastAPI
settings = get_settings()                                  # читаем настройки

class PassageMetadata(BaseModel):
    page_number: int = 0
    book_name: str = ""
    filename: str = ""


class PassageWithMetadata(BaseModel):
    text: str
    score: float
    metadata: PassageMetadata


class AskRequest(BaseModel):                               # схема входа POST /ask
    question: str                                          # поле вопроса
    top_k: int = 4                                         # число пассажей


class AskResponse(BaseModel):                              # схема ответа
    answer: str                                            # финальный текст
    passages: List[PassageWithMetadata]                    # список пассажей с метаданными

@app.on_event("startup")                                   # хук на старт приложения
def _load_index():                                         # функция загрузки индекса
    store.load()                                           # загружаем FAISS + метаданные

@app.get("/health")                                        # эндпоинт проверки
def health():                                              # функция обработчик
    return {"status": "ok"}                                # простой JSON


class BookInfo(BaseModel):
    filename: str
    title: str
    path: str


@app.get("/books", response_model=List[BookInfo])
def get_books() -> List[BookInfo]:
    """Возвращает список доступных книг"""
    books = []
    raw_dir = settings.RAW_DIR

    if os.path.exists(raw_dir):
        for filename in sorted(os.listdir(raw_dir)):
            if filename.lower().endswith('.pdf'):
                # Извлекаем читаемое название из имени файла
                title = filename.replace('.pdf', '').replace('_', ' ')
                books.append(BookInfo(
                    filename=filename,
                    title=title,
                    path=os.path.join(raw_dir, filename)
                ))

    return books


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    final_state = run_rag(req.question, top_k=req.top_k)

    # Преобразуем passages в новый формат с метаданными
    passages_with_metadata = []
    for passage_data in final_state.get("passages", []):
        if len(passage_data) == 3:
            # Новый формат: (text, score, metadata)
            text, score, metadata = passage_data
            passages_with_metadata.append(
                PassageWithMetadata(
                    text=text,
                    score=score,
                    metadata=PassageMetadata(
                        page_number=metadata.get("page_number", 0),
                        book_name=metadata.get("book_name", ""),
                        filename=metadata.get("filename", "")
                    )
                )
            )
        elif len(passage_data) == 2:
            # Старый формат для обратной совместимости: (text, score)
            text, score = passage_data
            passages_with_metadata.append(
                PassageWithMetadata(
                    text=text,
                    score=score,
                    metadata=PassageMetadata()
                )
            )

    return AskResponse(
        answer=final_state["answer"],
        passages=passages_with_metadata,
    )