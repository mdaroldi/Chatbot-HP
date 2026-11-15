.DEFAULT_GOAL := help
SHELL := /bin/bash
# GNU Make < 3.82 (macOS ships 3.81) ignores .SHELLFLAGS, so recipes must not rely on
# -e: every loop below stops explicitly with `|| exit 1` on the first failure.
.SHELLFLAGS := -eu -o pipefail -c

BACKEND := backend
FRONTEND := frontend
LOADTEST := loadtest
PROJECTS := $(BACKEND) $(FRONTEND) $(LOADTEST)
TEST_COMPOSE := docker compose -f docker-compose.test.yml
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f
TRIVY_IMAGE := aquasec/trivy:0.76.1@sha256:790fd66d1697395d8693d1acaa1ea702406a20f109b6fc7bac8db623eca6c06c
ACTIONLINT_IMAGE := rhysd/actionlint:1.7.12@sha256:b1934ee5f1c509618f2508e6eb47ee0d3520686341fec936f3b79331f9315667
IMAGES := hpdocs-backend:local hpdocs-frontend:local
TRIVY := docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$(CURDIR)":/src:ro \
	-v trivy-cache:/root/.cache $(TRIVY_IMAGE)

.PHONY: help install setup lint lint-workflows format typecheck test docs-config coverage-summary test-int test-ui-int test-eval-int smoke-llm \
	fetch-test-assets models \
	security images scan-images scan-config up down ingest inspect-chunks model-pins \
	purge eval-retrieval eval-estimate eval-answers eval-ablation eval-judge \
	eval-label-sheet eval-agreement eval-report eval-summary eval load load-up load-reset load-down check

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install locked dependencies for every project (what CI runs)
	for project in $(PROJECTS); do uv sync --frozen --project $$project || exit 1; done

setup: install ## install + pre-commit hooks (local development)
	uv run --frozen --project $(BACKEND) pre-commit install

lint: ## ruff check + ruff format --check (every project)
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen ruff check . && uv run --frozen ruff format --check .) \
			|| exit 1; \
	done

lint-workflows: ## actionlint on .github/workflows
	docker run --rm -v "$(CURDIR)":/repo --workdir /repo $(ACTIONLINT_IMAGE) -color

format: ## Apply ruff fixes and formatting (every project)
	for project in $(PROJECTS); do \
		(cd $$project && uv run --frozen ruff check --fix . && uv run --frozen ruff format .) \
			|| exit 1; \
	done

typecheck: ## mypy --strict (every project)
	for project in $(PROJECTS); do (cd $$project && uv run --frozen mypy) || exit 1; done

test: ## Unit tests with branch coverage gate, fail under 90 (every project)
	cd $(BACKEND) && uv run --frozen pytest tests/unit --cov --cov-report=term-missing
	cd $(FRONTEND) && uv run --frozen pytest --cov --cov-report=term-missing
	cd $(LOADTEST) && uv run --frozen pytest --cov --cov-report=term-missing

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

# Local only, not in test-int or CI (ADR 0011): it needs both images built and the
# index ingested. Start the stub stack first (README, testing); the tests fail when the
# backend is down and abort before sending anything unless /ready reports the stub.
UI_IT_BACKEND_URL ?= http://127.0.0.1:$${API_PORT:-8000}
test-ui-int: ## UI integration tests against the running stub stack (local only)
	cd $(FRONTEND) && UI_IT_BACKEND_URL="$(UI_IT_BACKEND_URL)" uv run --frozen pytest tests/integration -m integration

security: ## bandit, pip-audit (every lock file), gitleaks
	cd $(BACKEND) && uv run --frozen bandit -q -c pyproject.toml -r app
	cd $(FRONTEND) && uv run --frozen bandit -q -r app.py hpdocs_ui
	cd $(LOADTEST) && uv run --frozen bandit -q -r hpdocs_load
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
			--secret-config /src/trivy-secret.yaml \
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

# Files only (matrix, frozen datasets, git-ignored ledger): no stack, no key, no spend.
eval-estimate: ## Answer benchmark budget plan: run estimates, funded runs, spend measured/estimated
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers estimate --eval-dir ../eval

# Answer benchmark (ADR 0012), billed with LLM_PROVIDER=anthropic: the host driver
# restarts the backend with the run's settings, checks its env (printenv, never the
# key), runs `ask` in a harness container without the key, then restores the defaults.
# ANSWER_ARGS: --run NAME [--limit N] [--resume] [--retry-errors] [--allow-heldout].
ANSWER_ARGS ?=

eval-answers: ## Billed: one answer run via the host driver (needs `make up`); ANSWER_ARGS="--run reference-1 --limit 10"
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers drive \
		--eval-dir "$(CURDIR)/eval" $(ANSWER_ARGS)

# Billed: the matrix rows in priority order (every unreserved row, or ABLATION_ARGS="--runs
# NAME ..."). A finished row only gets its report again; a row the budget plan cuts is
# skipped; the naive row builds its own alias first (ingest --if-needed, CHUNKER=naive);
# a run that stops stops the sweep. Rebuild the backend image after a code change first.
# ABLATION_ARGS="--provider stub --limit N ..." is a free check of the sweep.
ABLATION_ARGS ?=

eval-ablation: ## Billed: sweep the matrix rows in priority order (needs `make up`); ABLATION_ARGS="--runs dense naive"
	cd $(BACKEND) && GIT_SHA="$$(git rev-parse HEAD)$$(git diff --quiet HEAD || echo -dirty)" \
		uv run --frozen python -m app.evaluation.answers ablate --eval-dir "$(CURDIR)/eval" $(ABLATION_ARGS)

# Billed: the judge pass of one run through the Message Batches API, in a one-off
# backend container (Qdrant for the gold passages; ANTHROPIC_API_KEY from .env through
# compose, which `ask` never gets). As the caller's user, like eval-retrieval. A second
# call collects an open batch or reuses recorded verdicts; it never pays twice.
# JUDGE_ARGS: --run NAME [--limit N] [--allow-heldout].
JUDGE_ARGS ?=

eval-judge: ## Billed: judge one run's answers via the Batch API (needs `make up`); JUDGE_ARGS="--run reference-1 --limit 3"
	docker compose run --rm --no-deps --user "$$(id -u):$$(id -g)" \
		--volume "$(CURDIR)/eval:/eval" \
		--entrypoint python backend -m app.evaluation.answers judge $(JUDGE_ARGS)

# Files only, on the host (no stack, no key, no spend): the label sheet of a judged run,
# built from the inputs the judge pass saved, and the judge's agreement with the labels.
LABEL_RUN ?= reference-1

eval-label-sheet: ## Write the human label sheet of a judged run (eval/answer-runs/RUN/labels.csv); LABEL_RUN=reference-1
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers label-sheet \
		--eval-dir ../eval --run $(LABEL_RUN)

eval-agreement: ## Judge agreement with the human labels (raw and Cohen's kappa); LABEL_RUN=reference-1
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers agreement \
		--eval-dir ../eval --run $(LABEL_RUN)

# Local only, like test-ui-int: the answer harness end to end on the stub provider, no
# key and no spend. The driver pins the stub and the stub matrix's models, so it runs on
# any .env, and puts the backend back on .env after. Its files (with their own ledger)
# live in eval/answer-runs/stub/, cleared first. The second call resumes the same run.
test-eval-int: ## Answer harness pipeline on the stub provider (local, needs `make up`): restart, checks, ask, resume
	rm -rf "$(CURDIR)/eval/answer-runs/stub"
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers drive --provider stub \
		--eval-dir "$(CURDIR)/eval" --run reference-1 --limit 3
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers drive --provider stub \
		--eval-dir "$(CURDIR)/eval" --run reference-1 --limit 2 --resume

# Files only, on the host (no stack, no key, no spend): one finished run's report in its
# git-ignored run directory, and the committed cross-run summary built from the reports.
# REPORT_ARGS: --run NAME [--provider stub] [--allow-heldout].
REPORT_ARGS ?= --run reference-1

eval-report: ## Write one finished run's report (eval/answer-runs/RUN/report.md); REPORT_ARGS="--run reference-1"
	cd $(BACKEND) && GIT_SHA="$$(git rev-parse HEAD)$$(git diff --quiet HEAD || echo -dirty)" \
		uv run --frozen python -m app.evaluation.answers report --eval-dir ../eval $(REPORT_ARGS)

eval-summary: ## Write eval/reports/answers-summary.md from the run reports (files only)
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers summary --eval-dir ../eval

# The reference config once on dev (resumed if it exists, so a finished run pays nothing
# again), then its report and the summary. Billed and inside the ledger's cap: needs the
# key (in the environment or .env) and `make up`; without it, exit 2 before anything runs.
# EVAL_PROVIDER=stub is the free pipeline check: test-eval-int's 5 stub items, then their
# report, which says it makes no quality claims.
EVAL_PROVIDER ?= anthropic

eval: ## Quality benchmark: reference run on dev, report, summary (billed); EVAL_PROVIDER=stub for the free pipeline check
ifeq ($(EVAL_PROVIDER),stub)
	$(MAKE) test-eval-int
	$(MAKE) eval-report REPORT_ARGS="--run reference-1 --provider stub"
else
	@[ -n "$$ANTHROPIC_API_KEY" ] || grep -Eq '^ANTHROPIC_API_KEY=[^[:space:]]+' .env 2>/dev/null || { \
		echo "make eval needs ANTHROPIC_API_KEY (EVAL_PROVIDER=stub runs the free pipeline check)" >&2; \
		exit 2; }
	cd $(BACKEND) && uv run --frozen python -m app.evaluation.answers drive \
		--eval-dir "$(CURDIR)/eval" --run reference-1 --resume
	$(MAKE) eval-report REPORT_ARGS="--run reference-1"
	$(MAKE) eval-summary
endif

# Load stack (Phase 6, ADR 0013): own project, resource limits, load profile after .env.
LOAD_COMPOSE := docker compose -p hpdocs-load -f docker-compose.yml -f docker-compose.load.yml --env-file .env --env-file loadtest/load.env
LOAD_PROVIDER ?= stub
LOAD_API_URL := http://127.0.0.1:8100
# /ready is 503 until the models are warm (ADR 0008); 3 minutes covers a first start.
LOAD_WAIT_READY := for i in $$(seq 90); do curl -fs -o /dev/null $(LOAD_API_URL)/ready && exit 0; sleep 2; done; echo "load stack not ready"; exit 1

load-up: ## Start the load stack: backend and dependencies only (LOAD_PROVIDER=anthropic: Profile B)
	LLM_PROVIDER=$(LOAD_PROVIDER) $(LOAD_COMPOSE) up --build --detach --wait backend
	@$(LOAD_WAIT_READY)

load-reset: ## Same starting state: clear conversations (index kept), fresh backend, wait for /ready
	$(LOAD_COMPOSE) exec -T postgres sh -c 'psql -q -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -c "TRUNCATE messages, conversations"'
	LLM_PROVIDER=$(LOAD_PROVIDER) $(LOAD_COMPOSE) up --detach --wait --no-deps --force-recreate backend
	@$(LOAD_WAIT_READY)

load-down: ## Stop the load stack (its volumes are kept)
	$(LOAD_COMPOSE) down

load: ## Run the load test
	@echo "not implemented yet"; exit 1

check: lint typecheck test security ## Run before every commit
