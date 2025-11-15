# Book RAG (FastAPI + Streamlit + FAISS)

Небольшой учебный проект: локальный RAG по книгам «Властелин колец».

- backend: FastAPI  
- frontend: Streamlit  
- векторное хранилище: FAISS  
- эмбеддинги: `intfloat/multilingual-e5-base` (SentenceTransformers)  
- книги лежат как PDF в `data/raw/` и хранятся в Git LFS  
- индексы (`data/indexes/*`) и производные файлы не коммитятся, строятся на месте

Идея простая:  
кладёшь книги в `data/raw/` → запускаешь сборку индекса → задаёшь вопросы по содержанию и смотришь релевантные выдержки из текста.

---

## 1. Требования

Минимум для запуска через Docker:

- Docker
- Docker Compose
- Git
- Git LFS

Дополнительно (если хочешь запускать всё без Docker):

- Python 3.11
- Poetry 2.x

---

## 2. Клонирование репозитория

Репозиторий: `Mikhail-Osintsev/LLM_Team`  
Этот проект лежит в подкаталоге `book-rag`.

SSH:

```bash
git clone git@github.com:Mikhail-Osintsev/LLM_Team.git
cd LLM_Team/book-rag


make lfs-setup

make lfs-pull

ls -lh data/raw

cp .env.example .env

make up


```
book-rag/
  app/
    backend/
      api/
        main.py             # FastAPI, роут /ask
      core/
        config.py           # настройки (BaseSettings)
        retriever.py        # поиск по FAISS
        generator.py        # сбор "ответа" из чанков
        query_processor.py  # оркестрация: retrieve + generate
      services/
        embeddings.py       # обёртка над SentenceTransformer
        vector_store.py     # класс FaissStore, загрузка index.faiss
    data_processing/
      ingestion/
        book_parser.py      # чтение PDF/ePub/txt
        text_chunker.py     # нарезка текста на чанки
      indexing/
        index_builder.py    # построение FAISS индекса
    frontend/
      streamlit_app.py      # основное Streamlit-приложение
      components/           # UI-компоненты (чат, сайдбар, цитаты)
        chat_interface.py
        citation_display.py
        sidebar.py

  data/
    raw/                    # сырьевые книги (PDF), под Git LFS
    indexes/                # FAISS-индекс (index.faiss + store.pkl)
    processed/              # потенциальные промежуточные данные
    metadata/               # доп. метаданные (зарезервировано)

  docker/
    backend.Dockerfile      # образ backend
    frontend.Dockerfile     # образ frontend

  scripts/
    ingest.py               # entry-point для построения индекса

  tests/
    test_api.py             # базовый тест API
    test_retrieval.py       # базовый тест ретривера

  docker-compose.yml        # описание сервисов backend + frontend
  makefile                  # команды для сборки/запуска
  pyproject.toml            # зависимости проекта (Poetry)
  poetry.lock               # зафиксированные версии зависимостей