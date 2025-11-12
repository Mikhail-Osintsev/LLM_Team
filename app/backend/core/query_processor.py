from app.backend.core.retriever import retrieve           # импортируем ретривер
from app.backend.core.generator import generate_extractive_answer  # импортируем генератор

def answer_question(query: str, top_k: int = 4) -> dict:
    passages = retrieve(query, top_k=top_k)               # находим релевантные чанки
    answer = generate_extractive_answer(query, passages)  # формируем ответ
    return {"answer": answer, "passages": passages}       # возвращаем структуру для API