FROM python:3.12-slim

WORKDIR /app

# Быстрые колёса и меньше шума
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_VIRTUALENVS_CREATE=false

# Poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Ставим зависимости до копирования кода (чтобы кэшировалось)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root


# Теперь код приложения
COPY app ./app
COPY scripts ./scripts

ENV PYTHONPATH=/app
EXPOSE 8001

# Только сервер. Индексацию будем запускать из Makefile (разово)
CMD ["poetry","run","uvicorn","app.backend.api.main:app","--host","0.0.0.0","--port","8001"]