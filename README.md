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
- Python 3.11
- Poetry 2.x

## 2. Клонирование репозитория

Репозиторий: `Mikhail-Osintsev/LLM_Team`  
Этот проект лежит в подкаталоге `book-rag`.

SSH:

```bash
git clone git@github.com:Mikhail-Osintsev/LLM_Team.git
cd LLM_Team/book-rag


Шаг 1. Инициализация Git LFS 
make lfs-setup

Шаг 2. Докачать большие файлы из LFS
make lfs-pull

Шаг 3. Переменные окружения
cp .env.example .env

Шаг 4. Поднятие сервисов
make up

Шаг 5. Открыть интерфейс
	•	фронтенд:
http://localhost:8501
	•	бэкенд:
	•	API: http://localhost:8001
	•	Swagger: http://localhost:8001/docs
