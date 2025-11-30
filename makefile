SHELL := /bin/bash

PY = poetry run
COMPOSE = docker compose

# пути к файлам индекса
INDEX_FAISS = data/indexes/index.faiss
INDEX_META  = data/indexes/store.pkl

.PHONY: help lfs-setup lfs-pull setup ingest up down logs rebuild test clean

help:
	@echo "Команды:"
	@echo "  make lfs-setup  - инициализировать Git LFS (один раз на машине)"
	@echo "  make lfs-pull   - докачать большие файлы LFS (raw книги)"
	@echo "  make setup      - установить зависимости Poetry локально"
	@echo "  make ingest     - построить FAISS-индекс в контейнере backend"
	@echo "  make up         - поднять backend и frontend (если индекса нет - соберём)"
	@echo "  make down       - остановить контейнеры"
	@echo "  make logs       - логи сервисов"
	@echo "  make rebuild    - пересобрать образы и поднять"
	@echo "  make test       - pytest локально"
	@echo "  make clean      - удалить __pycache__ и *.pyc"

# 0) Git LFS
lfs-setup:
	# git lfs install - регистрирует LFS-хуки в текущем git
	git lfs install

lfs-pull:
	# git lfs pull - докачивает все объекты LFS под текущую ревизию
	git lfs pull

# 1) локальная разработка
setup:
	poetry install

# 2) построить индекс в одинаковой среде (через контейнер backend)
ingest:
	$(COMPOSE) run --rm backend poetry run python scripts/ingest.py

# 3) поднять сервисы; если индекса нет - соберём его разово
up:
	@if [ ! -f "$(INDEX_FAISS)" ] || [ ! -f "$(INDEX_META)" ]; then \
		echo "Индекс не найден - строю..."; \
		$(COMPOSE) run --rm backend poetry run python scripts/ingest.py; \
	fi
	$(COMPOSE) up 

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

rebuild:
	$(COMPOSE) build --no-cache
	$(COMPOSE) up 

test:
	$(PY) pytest -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +; \
	find . -type f -name "*.pyc" -delete