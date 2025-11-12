from typing import List                   # типы

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    chunks: List[str] = []               # список чанков
    i, n = 0, len(text)                  # i — позиция, n — длина текста
    while i < n:                         # пока не дошли до конца
        j = min(i + size, n)             # правая граница
        piece = text[i:j].strip()        # кусок без пробелов по краям
        if piece:                        # если непусто
            chunks.append(piece)         # добавляем
        i = j - overlap if j - overlap > i else j  # шагаем с перекрытием
    return chunks                        # отдаём список