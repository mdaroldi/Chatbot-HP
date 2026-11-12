.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

BACKEND := backend
TEST_COMPOSE := docker compose -f docker-compose.test.yml
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f

.PHONY: help setup lint format typecheck test test-int security up down ingest eval load check

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	uv sync --frozen --project $(BACKEND)

lint: ## ruff check + ruff format --check
	cd $(BACKEND) && uv run --frozen ruff check . && uv run --frozen ruff format --check .

format: ## Apply ruff fixes and formatting
	cd $(BACKEND) && uv run --frozen ruff check --fix . && uv run --frozen ruff format .

typecheck: ## mypy --strict
	cd $(BACKEND) && uv run --frozen mypy

test: ## Unit tests with branch coverage gate (fail under 90)
	cd $(BACKEND) && uv run --frozen pytest tests/unit --cov --cov-report=term-missing

test-int: ## Integration tests against real Postgres and Qdrant (needs Docker)
	$(TEST_COMPOSE) up -d --wait
	cd $(BACKEND) && uv run --frozen pytest tests/integration -m integration; \
	status=$$?; cd .. && $(TEST_COMPOSE) down -v; exit $$status

security: ## bandit, pip-audit, gitleaks
	cd $(BACKEND) && uv run --frozen bandit -q -c pyproject.toml -r app
	cd $(BACKEND) && uv run --frozen pip-audit --strict --progress-spinner off
	docker run --rm -v "$(CURDIR)":/repo $(GITLEAKS_IMAGE) git /repo --redact --no-banner

up: ## docker compose up
	@echo "not implemented yet"; exit 1

down: ## docker compose down
	@echo "not implemented yet"; exit 1

ingest: ## Build the vector index
	@echo "not implemented yet"; exit 1

eval: ## Run the quality benchmark
	@echo "not implemented yet"; exit 1

load: ## Run the load test
	@echo "not implemented yet"; exit 1

check: lint typecheck test security ## Run before every commit
