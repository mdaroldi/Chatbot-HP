# CLAUDE.md

Guidance for Claude Code in this repository. Read it fully before any task.

## Project

HP Docs Chatbot: a chatbot that answers questions about two public HP documents using an LLM with Retrieval-Augmented Generation (RAG). It is a take-home challenge for a Lead Machine Learning Engineer position, so it is judged on engineering quality as much as on features. The repository is public. All code, comments, commits and docs are in English.

### Hard requirements from the challenge (never break these)

- Backend in Python with FastAPI.
- Backend unit tests with at least 90% coverage.
- Graphical frontend for user interaction.
- Open-source vector database for RAG.
- The vector database is built ONLY from the two HP documents listed below.
- An explicit chunking strategy and an explicit search strategy.
- Conversation with chat history; chats and history stored in the backend.
- Everything runs through Docker Compose, configured by environment variables.
- Scalability shown with load tests (requests per minute supported).
- Quality of LLM answers is benchmarked.

### Source documents (the only allowed corpus)

- HP ENVY 6000 All-in-One series User Guide: https://h10032.www1.hp.com/ctg/Manual/c06584210.pdf
- OMEN 17.3 inch Gaming Laptop PC Maintenance and Service Guide: https://kaas.hpcloud.hp.com/pdf-public/pdf_10301542_en-US-1.pdf

The PDFs are copyrighted by HP. Never commit them or large verbatim extracts. Ingestion downloads them from these URLs and verifies pinned SHA-256 checksums.

## Architecture decisions (approved, do not change without asking)

| Area | Decision |
|---|---|
| Language and tooling | Python 3.12, uv for dependencies (lock file committed) |
| API | FastAPI, fully async, Pydantic v2, pydantic-settings |
| Frontend | Streamlit app in its own container; talks to the backend only over HTTP |
| Vector DB | Qdrant (dense + sparse vectors in one collection) |
| Relational DB | PostgreSQL, SQLAlchemy 2 async, asyncpg, Alembic migrations |
| Embeddings | Local open-source models on CPU via fastembed: dense `BAAI/bge-small-en-v1.5`, sparse BM25 |
| Reranker | Local cross-encoder on CPU via fastembed, switchable by env var |
| Chunking | Structure-aware: follow the heading hierarchy, one procedure or table per chunk, at most 450 tokens including the "document > chapter > section" prefix (bge-small truncates at 512 tokens); see ADR 0005 |
| Search | Hybrid dense + BM25 with reciprocal rank fusion, then rerank top 20 down to 5 |
| LLM | Anthropic API behind a provider interface; a deterministic stub provider for tests and load tests |
| Chat history | Stored in Postgres; follow-up questions are rewritten into standalone queries before retrieval |
| Users | Anonymous client ID per browser, kept by the Streamlit app in a URL query parameter and sent to the API as `X-Client-ID`; schema ready for real auth later |
| Rate limiting | Chat endpoints limited per `X-Client-ID` (validated like `X-Request-ID`), falling back to client IP when absent |
| Answers | English only, always with citations (document, section, page); explicit refusal when the answer is not in the documents; WARNING and CAUTION notes from the source are carried into answers |
| Icons in PDFs | No vision pass. Button glyphs lost in text extraction are a documented known limitation |
| Evaluation | Versioned golden dataset; retrieval metrics and LLM-judged generation metrics; ablation of chunking and search variants |
| Load testing | Locust, two profiles: stub LLM (system capacity) and real LLM (end to end) |

Model IDs, keys, URLs and tuning values come from environment variables. Never hardcode them. Before pinning an Anthropic model ID in `.env.example`, verify it in the current Anthropic documentation.

## Repository layout

```
backend/
  app/
    api/          # routers, request and response schemas, dependencies
    core/         # settings, logging, errors
    domain/       # pure business logic and interfaces (no I/O)
    services/     # chat, retrieval, generation orchestration
    infra/        # Qdrant, Postgres, LLM providers, embedding adapters
    ingestion/    # download, parse, clean, chunk, embed, upsert (CLI)
  tests/
    unit/         # no network, no containers
    integration/  # real Qdrant and Postgres, marked `integration`
  alembic/
frontend/         # Streamlit app
eval/             # golden dataset, benchmark runner, reports
loadtest/         # Locust files and results
docs/adr/         # architecture decision records
docker-compose.yml
Makefile
.env.example
```

Startup order (ADR 0007): `postgres` and `qdrant` healthy -> one-shot `migrate`
(`alembic upgrade head`) -> one-shot `ingest` (`--if-needed`) -> `backend` -> `frontend`.
Migrations do not run in the backend entrypoint. There are no compose profiles: every
service runs on a plain `docker compose up`.

Dependency rule: `api` -> `services` -> `domain` <- `infra`. `domain` imports nothing from the other layers. External systems are reached only through interfaces defined in `domain`, injected with FastAPI dependencies.

## Commands

All common tasks go through the Makefile. Keep it up to date.

```
make setup        # install dependencies and pre-commit hooks
make lint         # ruff check + ruff format --check
make typecheck    # mypy --strict
make test         # unit tests with coverage gate (branch coverage, fail under 90)
make test-int     # integration, corpus and model tests (needs Docker)
make fetch-test-assets  # download and verify the HP manuals and models for test-int (CI step)
make models       # download and verify every pinned model into backend/models (model tests)
make security     # bandit, pip-audit, gitleaks
make up / down    # docker compose up / down
make ingest       # force a new index build through compose (up builds only when needed)
make model-pins   # regenerate backend/models.toml and print the diff (manual, review before committing)
make eval         # run the quality benchmark
make load         # run the load test
make check        # lint + typecheck + test + security (run before every commit)
```

## Development workflow: strict TDD

1. Write a failing test that describes the behavior. Run it and see it fail.
2. Write the minimum code to make it pass.
3. Refactor with tests green.
4. Run `make check` before committing.

Rules:

- No production code without a test that required it.
- Never weaken, skip or delete a test to make the suite pass. Never add `# pragma: no cover`, lower the coverage threshold, or exclude files from coverage without asking first.
- Unit tests are fast and isolated: no network, no real databases, no real LLM calls. Use fakes that implement the `domain` interfaces instead of deep mocking.
- Test behavior, not implementation details. Cover error paths and edge cases, not only the happy path.
- Integration tests use real Qdrant and Postgres containers and are marked `@pytest.mark.integration`.
- The 90% gate applies to unit tests alone.

## Code standards

- Type hints everywhere; `mypy --strict` must pass. No `Any` without a comment explaining why.
- Ruff for linting and formatting. No unused code, no commented-out code.
- Small functions and modules with one responsibility. Prefer plain functions and dataclasses or Pydantic models over deep class hierarchies.
- Async all the way in the request path. No blocking calls inside async functions; CPU-heavy work (embedding, reranking) runs in a thread pool.
- Errors: domain-specific exceptions mapped to HTTP responses in one place. Never swallow exceptions. Never return stack traces to clients.
- Logging: structured JSON logs with a request ID. Never use `print`.
- Docstrings on public functions explain why, not what.
- Add a dependency only when needed; pin it through the lock file and say why in the commit.

## Security

- Secrets only through environment variables. `.env` is git-ignored; `.env.example` holds placeholders only. Never write a real key anywhere in the repo, tests, logs or commit messages.
- Validate all input with Pydantic, including length limits on user messages.
- Retrieved document text and user messages are untrusted data. Keep them clearly delimited in prompts and never let them act as instructions.
- CORS restricted to the frontend origin. Rate limiting on chat endpoints.
- Containers run as non-root, from pinned slim base images, with health checks. Only the frontend and API ports are published.
- SQL only through SQLAlchemy with bound parameters.
- CI runs gitleaks, bandit, pip-audit and a Trivy image scan.

## Data governance

- Corpus allowlist: ingestion accepts only the two URLs above and fails if a checksum does not match.
- Every index build writes a manifest: document hashes, chunking version, embedding model, chunk counts, build time.
- Every chunk carries metadata: document ID, product, section path, page range, content type.
- Each assistant message stores the IDs of the chunks used, the model, token counts and latency, so answers are traceable.
- Chat content is personal data. Do not log message content by default (a debug flag may enable it locally). Provide an endpoint to delete a conversation and a configurable retention period.
- The README states that, with the cloud provider enabled, user questions and retrieved excerpts are sent to the Anthropic API.

## Git conventions

- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`, `ci:`).
- Small, focused commits that each pass `make check`. Test and implementation may be in the same commit.
- Never commit generated artifacts, model caches, PDFs, `.env`, or load test raw output.
- Do not push, force-push or rewrite history unless asked.

## Conventions learned so far

- Red runs: run new tests against a behavior-free skeleton and report how many
  passed and why (expected: zero). Tighten tests that pass vacuously.
- Every fix is its own `fix:` commit. When a line is uncovered, ask whether the
  branch is needed before adding a test.
- macOS ships GNU Make 3.81, which ignores .SHELLFLAGS: loops end with `|| exit 1`,
  and Makefile changes are proven with a failure in the first project, not the last.
- Never log user content: no request bodies, query strings, exception messages,
  source lines in tracebacks, validation inputs or SQL parameters.
- Pin everything: images by digest, actions by commit SHA, pre-commit hooks by SHA,
  dependencies through lock files. Do not adopt a release in its first 5 days
  (matches the Dependabot cooldown), except for security fixes.
- Relax container hardening only after showing it breaks, and document why.
- CI must not need .env: build images with `docker build`, not `docker compose build`.
- Integration tests fail, never skip, when their services are missing.
- Set the author identity repo-locally; never rely on the global git config.

## Working agreement

- Stay inside the scope of the current phase. If something outside it looks necessary, stop and ask.
- If a requirement is ambiguous or a decision above seems wrong, ask before coding. Do not silently pick a different library or design.
- Record each significant decision as a short ADR in `docs/adr/` (context, decision, consequences).
- Keep the README accurate: a fresh clone must run with `cp .env.example .env`, adding the API key, and `docker compose up`.
- At the end of each task, report: what was done, how it was verified (commands and results), and what is left or uncertain.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation: repo, tooling, CI, compose skeleton, health endpoints | done |
| 1 | Ingestion: download, parse, clean, chunk, embed, upsert, manifest | done |
| 2 | Retrieval: hybrid search, reranker, golden set, retrieval metrics | pending |
| 3 | Chat API: LLM gateway, prompt, citations, streaming, history | pending |
| 4 | Frontend: Streamlit chat UI with conversations and sources | pending |
| 5 | Benchmark: generation metrics, ablation, report | pending |
| 6 | Load testing: Locust profiles, tuning, capacity report | pending |
| 7 | Hardening: security review, docs, demo script | pending |

Update the status column when a phase is completed.
