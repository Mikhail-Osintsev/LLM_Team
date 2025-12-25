# Book RAG (FastAPI + Streamlit + FAISS)

Учебный проект: интеллектуальная система вопросов-ответов по книгам с использованием RAG (Retrieval-Augmented Generation) и агентного подхода на базе LangGraph.

## Основные технологии

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Векторное хранилище**: FAISS
- **Эмбеддинги**: `intfloat/multilingual-e5-base` (SentenceTransformers)
- **LLM**: Mistral Small (через API)
- **Агентный граф**: LangGraph
- **Контейнеризация**: Docker Compose

## Идея проекта

Система позволяет задавать вопросы по содержанию книг (например, «Властелин колец») и получать ответы, основанные на релевантных фрагментах текста. Используется агентный подход с планировщиком, который решает, какие инструменты вызвать для получения информации.

**Рабочий процесс:**
1. Загружаете PDF-книги в `data/raw/`
2. Запускаете индексацию
3. Задаёте вопросы через web-интерфейс
4. Получаете ответы с указанием источников (книга, страница)

---

## 1. Архитектура системы

### 1.1. Общая схема

Проект построен на микросервисной архитектуре с разделением на backend и frontend:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                         │
│  - Чат-интерфейс                                                │
│  - Отображение источников с метаданными                         │
│  - Боковая панель с настройками                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP REST API
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              RAG AGENT GRAPH (LangGraph)                  │  │
│  │                                                           │  │
│  │  START → Planner → Tools/Generate → END                   │  │
│  │           ↑           │                                   │  │
│  │           └───────────┘                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ MCP Tools    │  │  Retriever   │  │  Generator   │           │
│  │ (retrieve)   │  │  (FAISS +    │  │  (Mistral    │           │
│  │              │  │  embeddings) │  │   LLM)       │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  FAISS Index │  │  Metadata    │  │  PDF Books   │           │
│  │  (vectors)   │  │  (JSON)      │  │  (Git LFS)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2. Архитектура RAG-пайплайна

#### Агентный граф (LangGraph)

Система использует агентный подход с планировщиком и инструментами:

**Узлы графа:**

1. **Planner Node** ([rag_graph.py:51-146](app/backend/core/rag_graph.py#L51-L146))
   - Анализирует вопрос пользователя
   - Решает, нужно ли вызвать инструмент или можно сразу ответить
   - Формирует JSON с решением (`tool` или `answer`)
   - Использует Mistral LLM для планирования

2. **Tools Node** ([rag_graph.py:149-208](app/backend/core/rag_graph.py#L149-L208))
   - Вызывает MCP-инструменты (сейчас только `retrieve`)
   - Обрабатывает результаты и добавляет их в историю сообщений
   - Поддерживает метаданные (название книги, номер страницы)

3. **Generate Node** ([rag_graph.py:211-223](app/backend/core/rag_graph.py#L211-L223))
   - Генерирует финальный ответ на основе найденных фрагментов
   - Использует Mistral LLM для синтеза ответа

**Граф переходов:**
```
START → plan → [tools | generate]
              tools → plan (цикл до MAX_TOOL_CALLS)
              generate → END
```

#### Компоненты системы

**1. MCP Tools** ([mcp_tools.py](app/backend/core/mcp_tools.py))
- Реализация паттерна MCP (Model Context Protocol)
- `MCPServer`: регистрирует инструменты
- `MCPClient`: вызывает инструменты
- Инструмент `retrieve`: поиск релевантных фрагментов

**2. Retriever** ([retriever.py](app/backend/core/retriever.py))
- Создаёт эмбеддинги запроса
- Выполняет поиск по FAISS индексу
- Возвращает top-k релевантных фрагментов с метаданными

**3. Generator** ([generator.py](app/backend/core/generator.py))
- Строит промпт из вопроса и найденных фрагментов
- Вызывает LLM для генерации ответа
- Форматирует метаданные источников

**4. Vector Store** ([vector_store.py](app/backend/services/vector_store.py))
- Управление FAISS индексом
- Загрузка/сохранение индекса и метаданных
- Поиск по векторам

**5. Embeddings** ([embeddings.py](app/backend/services/embeddings.py))
- Создание векторных представлений текста
- Модель: `intfloat/multilingual-e5-base`

### 1.3. Взаимодействие компонентов

**Поток обработки запроса:**

1. **Пользователь** отправляет вопрос через Streamlit UI
2. **Frontend** делает POST запрос к `/ask` эндпоинту
3. **FastAPI** ([main.py:70-105](app/backend/api/main.py#L70-L105)) запускает RAG граф
4. **Planner** анализирует вопрос и решает вызвать `retrieve`
5. **Tools Node** вызывает `retrieve` через MCP:
   - **Retriever** создаёт эмбеддинг запроса
   - **Vector Store** ищет похожие векторы в FAISS
   - Возвращаются top-k фрагментов с метаданными
6. **Planner** получает результаты и решает перейти к генерации
7. **Generator** создаёт промпт и вызывает Mistral LLM
8. **LLM** генерирует ответ на основе контекста
9. **FastAPI** возвращает ответ и источники клиенту
10. **Frontend** отображает ответ с цитатами

### 1.4. Структура данных

**Директории:**
- `data/raw/` - исходные PDF книги (Git LFS)
- `data/processed/` - обработанные текстовые фрагменты
- `data/indexes/` - FAISS индексы и метаданные (не в Git)

**Формат метаданных:**
```python
{
    "page_number": int,     # номер страницы
    "book_name": str,       # название книги
    "filename": str         # имя файла
}
```

### 1.5. Список зависимостей

**Backend (Python):**
- `fastapi` - REST API сервер
- `uvicorn` - ASGI сервер
- `pydantic` - валидация данных
- `streamlit` - web-интерфейс
- `sentence-transformers` - создание эмбеддингов
- `faiss-cpu` - векторный поиск
- `pypdf` - парсинг PDF
- `langchain` - фреймворк для LLM
- `langgraph` - граф агентов
- `langchain-mistralai` - интеграция с Mistral AI
- `numpy` - работа с векторами

**Инфраструктура:**
- Docker & Docker Compose - контейнеризация
- Git LFS - хранение больших файлов
- Poetry - управление зависимостями

**Внешние сервисы:**
- Mistral AI API - LLM для генерации ответов

---

## 2. Требования

Минимум для запуска через Docker:

- Docker
- Docker Compose
- Git
- Git LFS
- Python 3.11+
- Poetry 2.x
- Mistral API ключ (для работы LLM)

---

## 3. Быстрый старт

### 3.1. Клонирование репозитория

Репозиторий: `Mikhail-Osintsev/LLM_Team`
Этот проект лежит в подкаталоге `book-rag`.

```bash
git clone git@github.com:Mikhail-Osintsev/LLM_Team.git
cd LLM_Team/book-rag
```

### 3.2. Установка и запуск

**Шаг 1.** Инициализация Git LFS
```bash
make lfs-setup
```

**Шаг 2.** Докачать большие файлы из LFS
```bash
make lfs-pull
```

**Шаг 3.** Настройка переменных окружения
```bash
cp .env.example .env
```

Отредактируйте `.env` и добавьте ваш Mistral API ключ:
```
MISTRAL_API_KEY=your_api_key_here
```

**Шаг 4.** Поднятие сервисов
```bash
make up
```

**Шаг 5.** Открыть интерфейс
- **Frontend (Streamlit)**: [http://localhost:8501](http://localhost:8501)
- **Backend API**: [http://localhost:8001](http://localhost:8001)
- **API Docs (Swagger)**: [http://localhost:8001/docs](http://localhost:8001/docs)

---

## 4. Структура проекта

```
book-rag/
├── app/
│   ├── backend/
│   │   ├── api/
│   │   │   └── main.py              # FastAPI приложение, эндпоинты
│   │   ├── core/
│   │   │   ├── config.py            # Конфигурация приложения
│   │   │   ├── rag_graph.py         # LangGraph граф агента
│   │   │   ├── mcp_tools.py         # MCP инструменты
│   │   │   ├── retriever.py         # Поиск по FAISS
│   │   │   ├── generator.py         # Генерация ответов
│   │   │   └── query_processor.py   # Обработка запросов
│   │   └── services/
│   │       ├── embeddings.py        # Создание эмбеддингов
│   │       └── vector_store.py      # Управление FAISS индексом
│   ├── frontend/
│   │   ├── streamlit_app.py         # Главный файл Streamlit
│   │   └── components/
│   │       ├── chat_interface.py    # Чат-интерфейс
│   │       ├── citation_display.py  # Отображение источников
│   │       └── sidebar.py           # Боковая панель
│   └── data_processing/
│       ├── ingestion/
│       │   ├── book_parser.py       # Парсинг PDF
│       │   └── text_chunker.py      # Разбиение на чанки
│       └── indexing/
│           └── index_builder.py     # Построение FAISS индекса
├── data/
│   ├── raw/                         # PDF книги (Git LFS)
│   ├── processed/                   # Обработанные тексты
│   └── indexes/                     # FAISS индексы (не в Git)
├── docker/
│   ├── backend.Dockerfile           # Dockerfile для backend
│   └── frontend.Dockerfile          # Dockerfile для frontend
├── scripts/
│   └── ingest.py                    # Скрипт индексации
├── tests/
│   ├── test_api.py                  # Тесты API
│   └── test_retrieval.py            # Тесты поиска
├── docker-compose.yml               # Оркестрация контейнеров
├── pyproject.toml                   # Зависимости Poetry
├── Makefile                         # Команды для управления
└── README.md                        # Этот файл
```

---

## 5. Использование

### 5.1. Работа через Web-интерфейс

1. Откройте [http://localhost:8501](http://localhost:8501)
2. Введите вопрос в поле чата
3. Система автоматически:
   - Найдёт релевантные фрагменты в книгах
   - Сгенерирует ответ на основе контекста
   - Покажет источники с указанием книги и страницы

### 5.2. Работа через API

**Пример запроса:**
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Кто такой Гэндальф?",
    "top_k": 4
  }'
```

**Пример ответа:**
```json
{
  "answer": "Гэндальф - один из волшебников (истари)...",
  "passages": [
    {
      "text": "Фрагмент текста из книги...",
      "score": 0.856,
      "metadata": {
        "page_number": 42,
        "book_name": "Властелин колец",
        "filename": "lotr_part1.pdf"
      }
    }
  ]
}
```

### 5.3. Добавление новых книг

1. Поместите PDF-файлы в `data/raw/`
2. Запустите индексацию:
```bash
docker-compose exec backend python scripts/ingest.py
```
3. Перезапустите сервисы:
```bash
make rebuild
```

---

## 6. Полезные команды

```bash
# Запуск сервисов
make up

# Остановка
make down

# Пересборка и запуск
make rebuild

# Просмотр логов
docker-compose logs -f

# Просмотр логов только backend
docker-compose logs -f backend

# Запуск тестов
docker-compose exec backend pytest

# Индексация книг
docker-compose exec backend python scripts/ingest.py
```

---

## 7. Разработка

### 7.1. Локальная разработка без Docker

```bash
# Установка зависимостей
poetry install

# Активация виртуального окружения
poetry shell

# Запуск backend
uvicorn app.backend.api.main:app --reload --host 0.0.0.0 --port 8001

# Запуск frontend (в отдельном терминале)
streamlit run app/frontend/streamlit_app.py
```

### 7.2. Расширение функциональности

**Добавление нового MCP инструмента:**

Отредактируйте [app/backend/core/mcp_tools.py](app/backend/core/mcp_tools.py):

```python
def my_tool(arg1: str, arg2: int) -> Dict[str, Any]:
    # Ваша логика
    return {"result": "..."}

mcp_server.register_tool(
    name="my_tool",
    description="Описание инструмента",
    parameters={
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "..."},
            "arg2": {"type": "integer", "description": "..."}
        },
        "required": ["arg1"]
    },
    func=my_tool
)
```

---

## 8. Лицензия

MIT

---

## 9. Авторы

- Mikhail Osintsev - [a89150242288@gmail.com](mailto:a89150242288@gmail.com)
- Andrey Ivanov - [cashman2100@gmail.com](mailto:cashman2100@gmail.com)
- Dmitriy Zolotukhin - [zolotukhind44@gmail.com](mailto:zolotukhind44@gmail.com)

---

## 10. Благодарности

- [LangChain](https://github.com/langchain-ai/langchain) - фреймворк для LLM
- [LangGraph](https://github.com/langchain-ai/langgraph) - граф агентов
- [FAISS](https://github.com/facebookresearch/faiss) - векторный поиск
- [Mistral AI](https://mistral.ai/) - LLM провайдер
- [Sentence Transformers](https://www.sbert.net/) - эмбеддинги
