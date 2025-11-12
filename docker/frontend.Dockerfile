FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_VIRTUALENVS_CREATE=false


RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY app ./app
ENV PYTHONPATH=/app
EXPOSE 8501

# BACKEND_URL получаем из docker-compose
CMD ["poetry","run","streamlit","run","app/frontend/streamlit_app.py","--server.address","0.0.0.0","--server.port","8501"]