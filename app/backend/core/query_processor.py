# app/backend/core/query_processor.py

from app.backend.core.rag_graph import run_rag


def answer_question(query: str, top_k: int = 4) -> dict:
    """
    Обёртка над графом RAG, которую вызывает FastAPI.

    На вход:
      - query: вопрос пользователя
      - top_k: сколько чанков вытаскивать из индекса

    На выход:
      - dict с полями:
         - "answer": финальный ответ LLM
         - "passages": список (текст, скор)
    """
    state = run_rag(question=query, top_k=top_k)

    return {
        "answer": state["answer"],
        "passages": state["passages"],
    }