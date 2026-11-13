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

.PHONY: help install setup lint lint-workflows format typecheck test docs-config coverage-summary test-int smoke-llm \
	fetch-test-assets models \
	security images scan-images scan-config up down ingest inspect-chunks model-pins \
	purge eval-retrieval eval load check

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

docs-config: ## Print the README configuration table generated from Settings
	@cd $(BACKEND) && uv run --frozen python -c \
		"import sys; from app.core.settings_docs import settings_markdown_table as t; sys.stdout.write(t())"

coverage-summary: ## Markdown coverage tables (used for the CI job summary)
	@for project in $(PROJECTS); do \
		echo "### Unit test coverage: $$project"; echo; \
		(cd $$project && uv run --frozen coverage report --format=markdown) || exit 1; echo; \
	done

# Marker expression for test-int; CI runs the default. Override it locally to run a
# subset, for example TEST_INT_MARKERS=integration to skip the HP manuals and models.
TEST_INT_MARKERS ?= integration or corpus or model

# The corpus is fetched live from HP and verified against backend/corpus.toml; models go
# to EMBEDDING_CACHE_DIR. Failures read CORPUS/MODEL UNAVAILABLE or CHANGED UPSTREAM.
fetch-test-assets: ## Download and verify the HP manuals and pinned models for test-int (3 attempts)
	cd $(BACKEND) && uv run --frozen python -m tests.downloads --attempts 3 --backoff-seconds 10

# The image bakes its own copy (ADR 0008); this fills backend/models for model tests.
models: ## Download and verify every pinned model into backend/models (3 attempts)
	cd $(BACKEND) && uv run --frozen python -m tests.downloads --models-only --attempts 3 --backoff-seconds 10

test-int: ## Integration tests (real Postgres and Qdrant, needs Docker), corpus and model tests
	$(TEST_COMPOSE) up -d --wait
	status=0; (cd $(BACKEND) && uv run --frozen pytest tests/integration tests/corpus tests/embeddings -m "$(TEST_INT_MARKERS)") \
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

# Models to pin; empty means the models already in backend/models.toml.
MODELS ?=

model-pins: ## Regenerate backend/models.toml from a fresh download and print the diff (manual only)
	cd $(BACKEND) && uv run --frozen python -c \
		"import sys; from app.infra.model_pins import main; sys.exit(main(sys.argv[1:]))" $(MODELS)

# Settings require a database password, but this target never opens the database.
inspect-chunks: ## Write every chunk to ingest-review/chunks.md for local review (git-ignored)
	cd $(BACKEND) && POSTGRES_PASSWORD=unused-by-inspect-chunks uv run --frozen \
		python -m app.ingestion --inspect-chunks ../ingest-review/chunks.md

# `docker compose up` builds the index only when it is not current (--if-needed);
# this target always builds a new one and moves the alias (ADR 0007). The backend
# service builds the image that ingest runs, so `run --build` would not rebuild it.
ingest: ## Rebuild the vector index now (forced) through compose
	docker compose build backend
	docker compose run --rm ingest --force

# Retrieval benchmark (ADR 0009): in process in the backend image, against the running
# stack's index, with the baked models; no deps and no profiles (ADR 0007). EVAL_ARGS is
# the command: `preflight`, or `run` with options, e.g. EVAL_ARGS="run --config dense".
# It runs as the caller's user so a Linux bind mount of eval/ stays writable (the image
# user is 10001); GIT_SHA ends in -dirty when tracked files have uncommitted changes.
EVAL_ARGS ?= run
SMOKE_ARGS ?= --runs 5

smoke-llm: ## Manual, billed: real-model answers and one rewrite; TTFT p50, stops, rewrite outcome (needs the key)
	docker compose run --rm --no-deps --env LLM_PROVIDER=anthropic \
		--entrypoint python backend -m app.smoke_llm $(SMOKE_ARGS)

purge: ## Delete conversations past CHAT_RETENTION_DAYS now (needs `make up`); prints counts
	docker compose run --rm --no-deps --entrypoint python backend -m app.retention

eval-retrieval: ## Retrieval benchmark via compose (needs `make up`); EVAL_ARGS=preflight|run ...
	docker compose run --rm --no-deps --user "$$(id -u):$$(id -g)" \
		--volume "$(CURDIR)/eval:/eval" \
		--env GIT_SHA="$$(git rev-parse HEAD)$$(git diff --quiet HEAD || echo -dirty)" \
		--entrypoint python backend -m app.evaluation $(EVAL_ARGS)

eval: ## Run the quality benchmark
	@echo "not implemented yet"; exit 1

load: ## Run the load test
	@echo "not implemented yet"; exit 1

check: lint typecheck test security ## Run before every commit
