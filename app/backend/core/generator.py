def generate_extractive_answer(query: str, passages: list[tuple[str, float]]) -> str:
    # Простая экстрактивная «генерация»: показываем контекст и короткую подпись
    bullets = "\n\n".join(
        [f"[{i+1}] score={score:.3f}\n{txt[:600]}{'...' if len(txt)>600 else ''}"
         for i, (txt, score) in enumerate(passages)]
    )
    return f"Вопрос: {query}\n\nКонтекст:\n{bullets}\n\nОтвет: см. выдержки выше."