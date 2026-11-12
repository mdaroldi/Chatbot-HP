.DEFAULT_GOAL := help
SHELL := /bin/bash
# GNU Make < 3.82 (macOS ships 3.81) ignores .SHELLFLAGS, so recipes must not rely on
# -e: every loop below stops explicitly with `|| exit 1` on the first failure.
.SHELLFLAGS := -eu -o pipefail -c

BACKEND := backend
FRONTEND := frontend
PROJECTS := $(BACKEND) $(FRONTEND)
TEST_COMPOSE := docker compose -f docker-compose.test.yml
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f
TRIVY_IMAGE := aquasec/trivy:0.76.1@sha256:790fd66d1697395d8693d1acaa1ea702406a20f109b6fc7bac8db623eca6c06c
ACTIONLINT_IMAGE := rhysd/actionlint:1.7.12@sha256:b1934ee5f1c509618f2508e6eb47ee0d3520686341fec936f3b79331f9315667
IMAGES := hpdocs-backend:local hpdocs-frontend:local
TRIVY := docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$(CURDIR)":/src:ro \
	-v trivy-cache:/root/.cache $(TRIVY_IMAGE)

.PHONY: help install setup lint lint-workflows format typecheck test coverage-summary test-int \
	security images scan-images scan-config up down ingest eval load check

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install locked dependencies for both projects (what CI runs)
	for project in $(PROJECTS); do uv sync --frozen --project $$project || exit 1; done

setup: install ## install + pre-commit hooks (local development)
	uv run --frozen --project $(BACKEND) pre-commit install

lint: ## ruff check + ruff format --check (backend and frontend)
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen ruff check . && uv run --frozen ruff format --check .) \
			|| exit 1; \
	done

lint-workflows: ## actionlint on .github/workflows
	docker run --rm -v "$(CURDIR)":/repo --workdir /repo $(ACTIONLINT_IMAGE) -color

format: ## Apply ruff fixes and formatting (backend and frontend)
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen ruff check --fix . && uv run --frozen ruff format .) \
			|| exit 1; \
	done

typecheck: ## mypy --strict (backend and frontend)
	for project in $(PROJECTS); do (cd $$project && uv run --frozen mypy) || exit 1; done

test: ## Unit tests with branch coverage gate, fail under 90 (backend and frontend)
	cd $(BACKEND) && uv run --frozen pytest tests/unit --cov --cov-report=term-missing
	cd $(FRONTEND) && uv run --frozen pytest --cov --cov-report=term-missing

coverage-summary: ## Markdown coverage tables (used for the CI job summary)
	@for project in $(PROJECTS); do \
		echo "### Unit test coverage: $$project"; echo; \
		(cd $$project && uv run --frozen coverage report --format=markdown) || exit 1; echo; \
	done

test-int: ## Integration tests against real Postgres and Qdrant (needs Docker)
	$(TEST_COMPOSE) up -d --wait
	status=0; (cd $(BACKEND) && uv run --frozen pytest tests/integration -m integration) \
		|| status=$$?; $(TEST_COMPOSE) down -v; exit $$status

security: ## bandit, pip-audit (both lock files), gitleaks
	cd $(BACKEND) && uv run --frozen bandit -q -c pyproject.toml -r app
	cd $(FRONTEND) && uv run --frozen bandit -q -r app.py hpdocs_ui
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen pip-audit --strict --progress-spinner off) || exit 1; \
	done
	docker run --rm -v "$(CURDIR)":/repo $(GITLEAKS_IMAGE) git /repo --redact --no-banner

images: ## Build the backend and frontend images (no .env needed, unlike compose)
	docker build --tag hpdocs-backend:local $(BACKEND)
	docker build --tag hpdocs-frontend:local $(FRONTEND)

scan-images: ## Trivy image scan: fail on fixable HIGH/CRITICAL vulnerabilities
	for image in $(IMAGES); do \
		$(TRIVY) image --quiet --skip-version-check --severity HIGH,CRITICAL \
			--ignore-unfixed --exit-code 1 $$image || exit 1; \
	done

scan-config: ## Trivy misconfiguration scan of the Dockerfiles (compose is not supported)
	$(TRIVY) config --quiet --skip-version-check --exit-code 1 \
		--skip-dirs /src/backend/.venv --skip-dirs /src/frontend/.venv /src

up: ## Build and start the stack, waiting until every service is healthy
	docker compose up --build --detach --wait

down: ## Stop the stack (named volumes are kept)
	docker compose down

ingest: ## Build the vector index
	@echo "not implemented yet"; exit 1

eval: ## Run the quality benchmark
	@echo "not implemented yet"; exit 1

load: ## Run the load test
	@echo "not implemented yet"; exit 1

check: lint typecheck test security ## Run before every commit
