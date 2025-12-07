import os
from typing import List, Dict, Any
from app.backend.core.config import get_settings
from app.data_processing.ingestion.book_parser import load_raw_texts_with_metadata
from app.data_processing.indexing.index_builder import build_faiss_with_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Загружаем страницы с метаданными
    pages = load_raw_texts_with_metadata(s.RAW_DIR)

    # Создаем чанки с сохранением метаданных
    chunks_with_metadata: List[Dict[str, Any]] = []

    for page in pages:
        # Разбиваем текст страницы на чанки
        text_chunks = splitter.split_text(page["text"])

        # Для каждого чанка сохраняем метаданные
        for chunk_text in text_chunks:
            chunks_with_metadata.append({
                "text": chunk_text,
                "page_number": page["page_number"],
                "book_name": page["book_name"],
                "filename": page["filename"]
            })

    print(f"Создано {len(chunks_with_metadata)} чанков из {len(pages)} страниц")
    build_faiss_with_metadata(chunks_with_metadata, s.INDEX_DIR)
    print("Индексация завершена")


if __name__ == "__main__":
    main()                                                   