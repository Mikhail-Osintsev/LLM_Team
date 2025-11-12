# Book RAG (FastAPI + Streamlit + FAISS)

Локальный RAG по книгам. Бэкенд — FastAPI, фронт — Streamlit, векторное хранилище — FAISS.
Raw книги храним в Git LFS. Индексы и производные файлы не коммитим.

## Требования
- Docker + Docker Compose
- Git LFS установлен локально

## Быстрый старт

```bash
# 1) клон
git clone <URL_репозитория>
cd book-rag

# 2) Git LFS (один раз на машине)
make lfs-setup

# 3) докачать большие файлы (raw книги) из LFS
make lfs-pull

# 4) переменные окружения
cp .env.example .env
# при необходимости поменяй порты/пути, BACKEND_URL для фронта выставляется в docker-compose

# 5) поднять сервисы
make up
# если индекса нет, make перед up вызовет ingest и построит индекс

# фронт: http://localhost:8501
# бэк:   http://localhost:8001, swagger: http://localhost:8001/docs