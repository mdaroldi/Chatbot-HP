.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

BACKEND := backend
FRONTEND := frontend
PROJECTS := $(BACKEND) $(FRONTEND)
TEST_COMPOSE := docker compose -f docker-compose.test.yml
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f

.PHONY: help setup lint format typecheck test test-int security up down ingest eval load check

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	for project in $(PROJECTS); do uv sync --frozen --project $$project; done

lint: ## ruff check + ruff format --check (backend and frontend)
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen ruff check . && uv run --frozen ruff format --check .); \
	done

format: ## Apply ruff fixes and formatting (backend and frontend)
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen ruff check --fix . && uv run --frozen ruff format .); \
	done

typecheck: ## mypy --strict (backend and frontend)
	for project in $(PROJECTS); do (cd $$project && uv run --frozen mypy); done

test: ## Unit tests with branch coverage gate, fail under 90 (backend and frontend)
	cd $(BACKEND) && uv run --frozen pytest tests/unit --cov --cov-report=term-missing
	cd $(FRONTEND) && uv run --frozen pytest --cov --cov-report=term-missing

test-int: ## Integration tests against real Postgres and Qdrant (needs Docker)
	$(TEST_COMPOSE) up -d --wait
	status=0; (cd $(BACKEND) && uv run --frozen pytest tests/integration -m integration) \
		|| status=$$?; $(TEST_COMPOSE) down -v; exit $$status

security: ## bandit, pip-audit (both lock files), gitleaks
	cd $(BACKEND) && uv run --frozen bandit -q -c pyproject.toml -r app
	cd $(FRONTEND) && uv run --frozen bandit -q -r app.py hpdocs_ui
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen pip-audit --strict --progress-spinner off); \
	done
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
