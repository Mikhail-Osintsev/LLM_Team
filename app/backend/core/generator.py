# app/backend/core/generator.py

"""
Генератор ответа: одна функция, которая:
- принимает LLM (любой BaseChatModel),
- вопрос,
- список чанков (текст, скор),
- возвращает финальный ответ как строку.
"""

from typing import List, Tuple
from langchain_core.language_models.chat_models import BaseChatModel


def build_rag_prompt(question: str, passages: List[Tuple[str, float]]) -> str:
    """
    Строим простой текстовый промпт для RAG.

    passages: список (текст чанка, скор).
    Мы аккуратно форматируем контекст, чтобы LLM было понятно,
    откуда взялись выдержки.
    """
    if not passages:
        return (
            "Вопрос пользователя:\n"
            f"{question}\n\n"
            "В базе знаний не найдено ни одного подходящего фрагмента. "
            "Объясни, что информации недостаточно."
        )

    context_lines = []
    for i, (text, score) in enumerate(passages, start=1):
        snippet = text.strip().replace("\n", " ")
        context_lines.append(f"[{i}] score={score:.3f}: {snippet}")

    context_block = "\n".join(context_lines)

    prompt = (
        "Ты ассистент, который отвечает только по предоставленным фрагментам книги.\n\n"
        "Контекст:\n"
        f"{context_block}\n\n"
        "Инструкция:\n"
        f"- Ответь на вопрос: \"{question}\"\n"
        "- Не выдумывай факты, используй только приведённые выдержки.\n"
        "- Если информации недостаточно, так и скажи честно.\n"
        "- Ответ дай на русском языке.\n"
    )

    return prompt


def generate_answer_from_passages(
    llm: BaseChatModel,
    question: str,
    passages: List[Tuple[str, float]],
) -> str:
    """
    Основная функция генерации:

    - строит промпт по question + passages,
    - вызывает LLM,
    - возвращает text-ответ.
    """
    if not passages:
        return "Я не нашёл подходящих фрагментов в книге, поэтому не могу ответить уверенно."

    prompt = build_rag_prompt(question, passages)

    # Простейший однократный вызов LLM без цепочек
    response = llm.invoke(prompt)

    # У разных моделей может быть .content или .text – берём content по умолчанию
    answer_text = getattr(response, "content", None) or str(response)

    return answer_text.strip()