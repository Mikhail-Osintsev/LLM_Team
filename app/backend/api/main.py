from fastapi import FastAPI                                # импортируем FastAPI
from pydantic import BaseModel                             # валидация входа/выхода
from app.backend.services.vector_store import store        # доступ к FAISS
from app.backend.core.config import get_settings           # настройки
from app.backend.core.query_processor import answer_question  # логика ответа

app = FastAPI(title="Book RAG API")                        # создаём приложение FastAPI
settings = get_settings()                                  # читаем настройки

class AskRequest(BaseModel):                               # схема входа POST /ask
    question: str                                          # поле вопроса
    top_k: int = 4                                         # число пассажей

class AskResponse(BaseModel):                              # схема ответа
    answer: str                                            # финальный текст
    passages: list[tuple[str, float]]                      # список (пассаж, скор)

@app.on_event("startup")                                   # хук на старт приложения
def _load_index():                                         # функция загрузки индекса
    store.load()                                           # загружаем FAISS + метаданные

@app.get("/health")                                        # эндпоинт проверки
def health():                                              # функция обработчик
    return {"status": "ok"}                                # простой JSON

@app.post("/ask", response_model=AskResponse)              # основной эндпоинт
def ask(req: AskRequest):                                  # функция обработчик
    result = answer_question(req.question, top_k=req.top_k)  # вызываем логику
    return result                                          # отдаём результат клиенту