from typing import List, Dict, Any
from pypdf import PdfReader
import os


def read_pdf_with_metadata(path: str, book_name: str) -> List[Dict[str, Any]]:
    """
    Читает PDF и возвращает список страниц с метаданными
    """
    r = PdfReader(path)
    pages_data = []

    for page_num, page in enumerate(r.pages, start=1):
        text = page.extract_text() or ""
        pages_data.append({
            "text": text,
            "page_number": page_num,
            "book_name": book_name
        })

    return pages_data


def read_pdf(path: str) -> str:
    """Старая функция для обратной совместимости"""
    r = PdfReader(path)
    pages = [p.extract_text() or "" for p in r.pages]
    return "\n".join(pages)


def load_raw_texts(raw_dir: str) -> List[str]:
    """Старая функция для обратной совместимости"""
    texts: List[str] = []
    for name in sorted(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, name)
        low = name.lower()
        if low.endswith(".pdf"):
            texts.append(read_pdf(p))
    return texts


def load_raw_texts_with_metadata(raw_dir: str) -> List[Dict[str, Any]]:
    """
    Загружает все PDF файлы с метаданными
    Возвращает список словарей с полями: text, page_number, book_name, filename
    """
    all_pages = []

    for name in sorted(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, name)
        low = name.lower()
        if low.endswith(".pdf"):
            # Извлекаем читаемое имя книги из имени файла
            book_name = name.replace('.pdf', '').replace('_', ' ')
            pages_data = read_pdf_with_metadata(p, book_name)

            for page_data in pages_data:
                page_data["filename"] = name
                all_pages.append(page_data)

    return all_pages                       