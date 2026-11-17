# ADR 0001: Architecture overview

- **Status:** accepted (Phase 0)
- **Date:** 2026-11-12

## Context

A take-home challenge asks for a RAG chatbot over two HP PDFs: a Python FastAPI
backend with at least 90% unit coverage, a graphical frontend, an open-source vector
database, chat history stored in the backend, Docker Compose with environment-based
configuration, load tests and an answer-quality benchmark. It is judged on
engineering quality as much as features, so the foundation must make later phases
cheap to build and easy to verify.

## Decision

- Four containers: Streamlit **frontend**, FastAPI **backend**, **PostgreSQL**
  (chats, history, traceability) and **Qdrant** (vectors). The frontend reaches the
  backend only over HTTP.
- Backend layers `api` → `services` → `domain` ← `infra`. `domain` defines
  interfaces (for example `DependencyProbe`) and imports no other layer; `infra`
  implements them; routes receive them through FastAPI dependencies, so unit tests
  inject fakes instead of mocking libraries.
- Fully async request path (SQLAlchemy 2 async with asyncpg, httpx), Alembic for the
  schema, pydantic-settings for configuration read only from environment variables.
- Operability built in from the start: JSON logs with a request ID, one error shape,
  liveness (`/health`) separate from readiness (`/ready`).
- Python 3.12 with uv and committed lock files; strict TDD; `make check` as the
  single local and CI gate.

## Alternatives considered

- **One container serving UI and API:** simpler, but couples release cycles and
  hides the HTTP contract the frontend must respect.
- **A single vector-plus-relational store (pgvector):** fewer services, but weaker
  hybrid dense+BM25 search; see ADR 0002.
- **Synchronous FastAPI with a thread pool:** simpler code, worse concurrency under
  the load tests the challenge requires.

## Consequences

- Each layer is unit-tested in isolation; real Postgres and Qdrant are exercised
  only by marked integration tests.
- Two uv projects (backend, frontend) mean two lock files and Makefile loops.
- The edge/data network split and read-only, capability-free containers are part
  of the baseline, not a later hardening step.

## Known limitations and deferred decisions

- **CORS on unhandled 500s:** the request ID middleware is outside CORS so every
  response, including preflights, carries `X-Request-ID`. Its generic 500 therefore
  has no CORS headers, and a browser calling cross-origin could not read it. No
  effect today, since Streamlit calls the API server-side.
- **Migrations at startup:** the backend entrypoint runs `alembic upgrade head`.
  Fine for one replica; with several (Phase 6) a single migration job is needed.
  Resolved in Phase 1 by ADR 0007: a one-shot `migrate` service runs them.
- **Pool sizing and `pool_timeout`:** the pool is per process (default 5 + 5
  overflow) and replicas × (size + overflow) must stay under Postgres
  `max_connections`. `pool_timeout` is not set (SQLAlchemy's 30 s default); add an
  explicit setting in Phase 6.
- **Streamlit calls the API server-side:** browsers never talk to the backend, so
  CORS is defence in depth only, and the per-browser client ID must be passed by the
  frontend explicitly (planned: a query parameter, sent as `X-Client-ID`).
- **Frontend image of about 830 MB:** mostly Streamlit's dependencies (pyarrow,
  pandas, numpy). Shrinking it would mean another UI framework, outside the
  approved stack.
- **`apt-get upgrade` in the runtime images:** added for CVE-2026-76844 and
  CVE-2026-85091 (zlib1g), which the pinned base image still ships. Remove it once
  the base digest is bumped past the fixed zlib1g and `make scan-images` passes
  without it.

## Update (Phase 7, 2026-11-17)

The text above is kept as written; this section records where its deferred items stand
at the end of Phase 7. The security review is in the
[threat model](../security/threat-model.md).

- **Frontend image:** measured on the Phase 7 builds at 632 MB unpacked and 185 MB
  compressed (decimal MB; method in
  [ADR 0008](0008-baked-models-and-inference.md#update-phase-7-2026-11-17-image-sizes)),
  not "about 830 MB". The cause is unchanged: Streamlit's dependencies.
- **`apt-get upgrade`:** still needed. The pinned base digest still ships the `+b1`
  build of `zlib1g`; the fix is in `+deb13u1`. Trivy passes with the upgrade in place.
- **Next step: `pool_timeout`.** Still not set (SQLAlchemy's 30 s default). In the Phase 6
  load test the bottleneck was the reranker and Postgres stayed at about 1% CPU
  ([ADR 0013](0013-load-testing.md)), so it stayed deferred. Add an
  explicit `POSTGRES_POOL_TIMEOUT_SECONDS`, shorter than the request deadline, before
  running several replicas.
- **Next step: an injected clock in the Anthropic adapter.** Two adapter tests check
  that a deadline fires, against the real clock. After the wall-clock audit that followed a
  flaky CI run, they were given 1 s of slack (`6c97d05`). A clock passed into the adapter would make them
  deterministic, like the access-log duration test (`0a3d309`). The adapter was not
  changed in Phase 7.
