# HP Docs Chatbot

A chatbot that will answer questions about two public HP documents using
Retrieval-Augmented Generation (RAG):

- HP ENVY 6000 All-in-One series User Guide
- OMEN 17.3 inch Gaming Laptop PC Maintenance and Service Guide

> **Status: Phase 1 (ingestion) built.** `docker compose up` downloads the two
> manuals, verifies them against pinned checksums and builds a hybrid (dense + BM25)
> Qdrant index of them. The API still serves only health and readiness checks, and
> the Streamlit page is a placeholder: retrieval and chat come in Phases 2 to 4; see
> [Status](#status).

## Architecture

```mermaid
flowchart LR
    user([Browser]) -->|":8501"| frontend
    subgraph edge [network: edge]
        frontend["frontend<br/>Streamlit"]
        backend["backend<br/>FastAPI + uvicorn"]
        ingest["ingest (one-shot)<br/>python -m app.ingestion"]
    end
    subgraph data [network: data, internal]
        migrate["migrate (one-shot)<br/>alembic upgrade head"]
        postgres[("postgres<br/>PostgreSQL 17")]
        qdrant[("qdrant<br/>Qdrant 1.20")]
    end
    hp[["HP download hosts<br/>(the two manuals)"]]
    hub[["Hugging Face hub<br/>(pinned models)"]]
    frontend -->|"HTTP: /health, /ready"| backend
    operator([Operator]) -->|":8000"| backend
    migrate -->|"schema"| postgres
    ingest -->|"build manifest"| postgres
    ingest -->|"collection + alias"| qdrant
    ingest -->|"HTTPS, first run only"| hp
    ingest -->|"HTTPS, first run only"| hub
    backend --> postgres
    backend --> qdrant
```

Startup order ([ADR 0007](docs/adr/0007-startup-order.md)): `postgres` and `qdrant`
healthy → `migrate` → `ingest` → `backend` → `frontend`. The backend starts only after
both one-shot services have completed successfully.

- **frontend** talks to the backend only over HTTP, from the Streamlit server.
- **backend** is the only service on both networks. `GET /health` is liveness and
  never touches dependencies; `GET /ready` probes Postgres (`SELECT 1`) and Qdrant
  (`/readyz`) concurrently and returns `503` with a per-dependency status document
  when one is down.
- **migrate** applies the Alembic migrations once per `up`; the backend entrypoint
  never migrates.
- **ingest** builds the vector index when it is missing or stale (see
  [Ingestion](#ingestion)). It is the only service that reaches the internet, and only
  to download the manuals and the models on its first run.
- **postgres** and **qdrant** sit on an internal network with no route out and are
  never published.

Layering inside the backend: `api` → `services` → `domain` ← `infra`. `domain`
holds pure types and interfaces (for example `DependencyProbe`); `infra` holds the
real adapters; unit tests use fakes of the domain interfaces. The decisions behind
this are recorded in [docs/adr](docs/adr).

## Quick start

Requirements: Docker with Compose v2.

```bash
cp .env.example .env        # then set POSTGRES_PASSWORD in .env
docker compose up --build   # or: make up (waits until every service is healthy)
```

- Frontend: http://localhost:8501 (shows backend health and readiness)
- API: http://localhost:8000 (`/health`, `/ready`, `/docs`)

Compose refuses to start without `POSTGRES_PASSWORD`. Only the API and the frontend
are published, on `127.0.0.1` by default. On a remote VM set `BIND_ADDRESS=0.0.0.0`
in `.env` to reach them from outside, and restrict access with the VM's firewall.

**The first `up` needs network access and takes a minute or more:** `ingest`
downloads the two manuals from HP and the embedding models from the Hugging Face hub,
then builds the index. Measured on a fresh clone with empty volumes (Apple M2, Docker
Desktop with 8 CPUs and 4 GB, November 2026): image build 38 s (`--no-cache`), first
`docker compose up --wait` 70 s, of which ingestion (download, parse, embed, upsert)
is about 47 s. Later runs need no network: the verified files stay in the
`corpus-cache` and `model-cache` volumes, and a second `up` took 21 s (1.6 s of it in
`ingest`, which found the index current and skipped the build). If a download or a
checksum check fails, `ingest` exits non-zero and **the backend does not start**, by
design: the API never runs on a missing or unverified index. `docker compose logs
ingest` shows the one structured error line; fix the network and run `docker compose
up` again.

## Ingestion

`ingest` runs `python -m app.ingestion --if-needed` on every `docker compose up`:

1. **Download and verify.** Only the two URLs in
   [`backend/corpus.toml`](backend/corpus.toml) are allowed. Each PDF must match its
   pinned SHA-256; a mismatch fails the build (the upstream file changed and needs a
   review). Verified files are kept in `CORPUS_DIR` and reused offline.
2. **Parse and clean** with pdfplumber: bookmarks give the section hierarchy, footers
   give the printed page numbers, ruled tables keep each row whole, and the table of
   contents, the index, navigation lists, footers and non-English text are dropped
   with per-rule counts ([ADR 0004](docs/adr/0004-pdf-parsing.md)).
3. **Chunk** by structure: one section, procedure or table per chunk, at most 450
   bge-small tokens including the "document > chapter > section" prefix; procedures
   split at step boundaries and tables at row boundaries with the header repeated;
   WARNING and CAUTION notes stay with their step and flag the chunk
   ([ADR 0005](docs/adr/0005-chunking-v1.md)). Every chunk carries its document,
   section path, PDF and printed page ranges and content type.
4. **Embed** each chunk twice, with a dense model and BM25 (see
   [Embedding models](#embedding-models)), on CPU.
5. **Publish** into a new versioned Qdrant collection, check that its point count
   equals the chunk count, then move the `hpdocs_chunks` alias to it. The previous
   collection is kept for rollback (`INDEX_KEEP_PREVIOUS`); older ones are deleted.
   Each build writes a manifest row to Postgres (`index_builds`): document hashes,
   chunking version, model revisions and file digests, chunk counts per document, drop
   statistics, status.

The current index holds 287 chunks: 149 from the ENVY 6000 user guide and 138 from the
OMEN 17.3 service guide (chunking version v2).

`--if-needed` skips the build (no download, no model load) only when the live build
matches the pinned document hashes, the chunking version, both models and their pins,
and the collection's point count equals the manifest's chunk count; any difference
rebuilds ([ADR 0007](docs/adr/0007-startup-order.md)). **`make ingest`** always builds
a new index and moves the alias (`docker compose build backend`, then `docker compose
run --rm ingest --force`; 47 s measured, with the same 287 point IDs in the new
collection); the API keeps serving the old collection until the swap. `make
inspect-chunks` writes every chunk to the git-ignored `ingest-review/chunks.md` for
local review.

Search over this index (hybrid dense + BM25 with reciprocal rank fusion, then a
cross-encoder rerank) is Phase 2.

### Embedding models

| Role | Model | Files from | License |
|---|---|---|---|
| Dense (384 dimensions) | `BAAI/bge-small-en-v1.5` | [`Qdrant/bge-small-en-v1.5-onnx-Q`](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q), Qdrant's quantized ONNX build | MIT |
| Sparse (BM25, IDF in Qdrant) | `Qdrant/bm25` | [`Qdrant/bm25`](https://huggingface.co/Qdrant/bm25) | Apache-2.0 |

Both run locally on CPU through fastembed; no embedding API is called. Each model is
pinned in [`backend/models.toml`](backend/models.toml) to a hub revision and the
SHA-256 of every file it loads; every build checks the digests, and a model without a
pin is never downloaded or loaded ([ADR 0006](docs/adr/0006-embedding-model-files.md)).
`make model-pins` regenerates the pins for review. The dense vectors come from the
quantized ONNX build, not the original weights, so published bge-small benchmark
scores are not a direct baseline for this index.

## Configuration

Everything is configured through environment variables (`.env` for Compose).
Compose-only keys: `BIND_ADDRESS`, `API_PORT`, `FRONTEND_PORT`. The frontend reads
`BACKEND_URL` (default `http://backend:8000`). Backend settings:

<!-- settings-table:start (generated by `make docs-config`) -->
| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test` or `production`. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `LOG_INCLUDE_EXCEPTION_MESSAGES` | `false` | Log exception messages (they may contain user text). Local debugging only; rejected in production. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:8501` | Comma-separated browser origins allowed by CORS; `*` is rejected. |
| `API_DOCS_ENABLED` | `true` | Serve `/docs`, `/redoc` and `/openapi.json`. |
| `POSTGRES_HOST` | `postgres` | Postgres host (fixed in Compose). |
| `POSTGRES_PORT` | `5432` | Postgres port (fixed in Compose). |
| `POSTGRES_DB` | `hpdocs` | Database name. |
| `POSTGRES_USER` | `hpdocs` | Database user. |
| `POSTGRES_PASSWORD` | *required* | Database password. Required; never has a default. |
| `POSTGRES_POOL_SIZE` | `5` | Pooled connections kept per API process. |
| `POSTGRES_MAX_OVERFLOW` | `5` | Extra connections beyond the pool; the cap is pool size plus this. |
| `POSTGRES_CONNECT_TIMEOUT_SECONDS` | `5.0` | Timeout for opening a Postgres connection. |
| `QDRANT_URL` | `http://qdrant:6333/` | Qdrant REST base URL (fixed in Compose). |
| `QDRANT_COLLECTION_ALIAS` | `hpdocs_chunks` | Alias that always names the live chunk collection. |
| `INDEX_KEEP_PREVIOUS` | `true` | Keep the previous build's collection after a swap, for rollback; older builds are always deleted. |
| `READINESS_TIMEOUT_SECONDS` | `2.0` | Per-dependency timeout for `GET /ready`. |
| `CORPUS_DIR` | `corpus` | Cache directory for the verified source PDFs; reused offline. |
| `CORPUS_MAX_BYTES` | `33554432` | Largest accepted source PDF, in bytes. |
| `CORPUS_DOWNLOAD_TIMEOUT_SECONDS` | `120.0` | Per-phase HTTP timeout and total deadline for one PDF download. |
| `EMBEDDING_DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed dense model; its tokenizer also measures chunk size. |
| `EMBEDDING_SPARSE_MODEL` | `Qdrant/bm25` | fastembed sparse model. |
| `EMBEDDING_CACHE_DIR` | `models` | Cache directory for the downloaded embedding models; reused offline. |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding batch (bounds memory). |
| `CHUNK_MAX_TOKENS` | `450` | Largest chunk in dense-model tokens, prefix included; bge-small reads at most 512 (ADR 0005). |
| `CHUNK_MIN_TOKENS` | `65` | Sections below this many tokens merge with the next one under the same parent (ADR 0005). |
| `RETRIEVAL_MODE` | `hybrid` | `dense`, `sparse` or `hybrid` (dense and BM25 fused with RRF). |
| `RETRIEVAL_RERANK_ENABLED` | `true` | Rerank the candidates with the cross-encoder. |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per question. |
| `RETRIEVAL_RERANK_CANDIDATES` | `20` | Candidates the reranker scores; at least `RETRIEVAL_TOP_K`. |
| `RETRIEVAL_PREFETCH_DENSE_LIMIT` | `20` | Dense candidates before fusion; at least `RETRIEVAL_RERANK_CANDIDATES`. |
| `RETRIEVAL_PREFETCH_SPARSE_LIMIT` | `20` | BM25 candidates before fusion; at least `RETRIEVAL_RERANK_CANDIDATES`. |
| `RETRIEVAL_RRF_K` | `60` | Reciprocal rank fusion constant k (Qdrant's default is 2). |
| `RETRIEVAL_PRODUCT_FILTER_ENABLED` | `false` | Restrict the search to one manual when the question names its product. |
| `RETRIEVAL_RELEVANCE_THRESHOLD` | *unset* | Lowest top reranker score that counts as relevant; unset until calibrated (ADR 0009). Belongs to one reranker model. |
| `RETRIEVAL_DENSE_QUERY_PREFIX` | *empty* | Text put before the question for the dense query only (BGE instruction). |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed cross-encoder; must be pinned in `models.toml`. |
| `INFERENCE_WORKERS` | `2` | Threads that run embedding and reranking for requests. |
| `INFERENCE_ONNX_THREADS` | `2` | ONNX intra-op threads per model call; workers times threads should fit the container's CPUs. |
| `QDRANT_SEARCH_TIMEOUT_SECONDS` | `2.0` | Deadline for one Qdrant search. |
| `QUERY_MAX_CHARS` | `1000` | Longest accepted question, in characters. |
| `SEARCH_ENDPOINT_ENABLED` | `false` | Serve the `POST /v1/search` debug endpoint; rejected in production. |
<!-- settings-table:end -->

A unit test fails if this table, `.env.example` and the `Settings` class disagree.

## Make targets

| Target | What it does |
|---|---|
| `make install` | Install the locked dependencies of both projects (used by CI) |
| `make setup` | `install` plus the pre-commit hooks (local development) |
| `make lint` / `make format` | ruff check and format, backend and frontend |
| `make typecheck` | mypy `--strict`, backend and frontend |
| `make test` | Unit tests with a branch-coverage gate of 90%, both projects |
| `make test-int` | Integration tests against real Postgres and Qdrant (Docker), plus the corpus and model tests on the real manuals and models |
| `make fetch-test-assets` | Download and verify the manuals and models for `test-int` (3 attempts; the CI download step) |
| `make security` | bandit, pip-audit on both lock files, gitleaks on the history |
| `make images` / `scan-images` / `scan-config` | Build images; Trivy image and Dockerfile scans |
| `make lint-workflows` | actionlint on the GitHub Actions workflow |
| `make up` / `make down` | Start (waiting for health) or stop the stack |
| `make docs-config` | Print the configuration table above |
| `make check` | lint + typecheck + test + security; run before every commit |
| `make ingest` | Build a new vector index now and move the alias (`up` builds only when needed) |
| `make inspect-chunks` | Write every chunk to `ingest-review/chunks.md` (git-ignored) for review |
| `make model-pins` | Regenerate `backend/models.toml` from the hub and print the diff (manual, review before committing) |
| `make eval` / `load` | Later phases; print "not implemented yet" and fail |

## Repository layout

```
backend/            FastAPI app (uv project, own lock file)
  app/api/          routes and FastAPI dependencies
  app/core/         settings, JSON logging, request ID middleware, errors
  app/domain/       pure types and interfaces (no I/O)
  app/services/     orchestration (readiness)
  app/infra/        adapters: Postgres, Qdrant, PDF parser, downloads, fastembed
  app/ingestion/    ingestion stages and CLI (python -m app.ingestion)
  alembic/          migrations 0001-0003 (baseline, index_builds, error_type)
  corpus.toml       the two allowed manuals with their pinned SHA-256
  models.toml       the pinned embedding model revisions and file digests
  tests/unit/       no network, no containers; coverage gate
  tests/integration/  real Postgres and Qdrant
  tests/corpus/     the real manuals (marker `corpus`)
  tests/embeddings/ the real models (marker `model`)
frontend/           Streamlit app (uv project, own lock file)
eval/, loadtest/    placeholders for Phases 2, 5 and 6
docs/adr/           architecture decision records
docker-compose.yml       full stack
docker-compose.test.yml  throwaway Postgres and Qdrant for integration tests
```

## Development and testing

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker.

```bash
make setup
make check       # what CI runs on every push, minus integration and images
make test-int    # starts docker-compose.test.yml on 127.0.0.1:55432/56333, tears down
```

`make test-int` also runs the corpus and model tests, which download the manuals and
the models on their first run (into `backend/corpus/` and `backend/models/`, both
git-ignored) and reuse them afterwards. A download problem fails as CORPUS UNAVAILABLE
or MODEL UNAVAILABLE, and a file that no longer matches its pin as CORPUS (or MODEL)
CHANGED UPSTREAM, never as a test assertion.

The project is built test-first. Unit tests use fakes that implement the domain
interfaces (for example `FakeProbe`), never real databases or the network; the
backend and the frontend each have a 90% branch-coverage gate (currently 100%).
Integration tests run Alembic and the probes against real containers and fail,
rather than skip, when those containers are not running.

Run the pieces without Docker (needs the variables from `.env.example`):

```bash
cd backend && uv run uvicorn app.main:create_app --factory --no-server-header
cd frontend && BACKEND_URL=http://localhost:8000 uv run streamlit run app.py
```

CI (GitHub Actions) runs lint, typecheck, unit tests with a coverage summary,
integration, corpus and model tests, security scans and a Docker build with Trivy.
Every job calls the Makefile, so CI and local runs execute the same commands, and no
job needs a `.env`. The integration job downloads the manuals live from HP on every
run and verifies them against the pinned checksums; **no HP content is stored by
GitHub**: the PDFs are never cached or uploaded as artifacts, and corpus-test failures
print page numbers, counts and identifiers, never document text. Only the open-source
model files are kept in the Actions cache, keyed by `backend/models.toml`. Mark the
integration job as a required check in the branch protection settings.

## Security and data governance

- **Secrets** come only from environment variables. `.env` is git-ignored and
  `.env.example` holds placeholders. The database password is a `SecretStr`, the
  DSN is unwrapped in exactly one place, and neither appears in logs, images or
  `docker history`.
- **Logs** are one JSON object per line with a request ID. Request bodies and query
  strings are never logged; exception messages are left out of logs by default,
  because they can contain user text (`LOG_INCLUDE_EXCEPTION_MESSAGES` re-enables
  them locally and is rejected in production).
- **Errors** share one JSON shape and never include stack traces or internals;
  validation errors omit the submitted input.
- **Containers** run as non-root numeric users, with all capabilities dropped,
  `no-new-privileges` and a read-only root filesystem. Base images are pinned by
  digest and scanned with Trivy; dependencies are locked and audited with
  pip-audit; Dependabot proposes updates weekly.
- **Corpus**: the HP PDFs are copyrighted by HP and are not in this repository.
  Ingestion downloads them from their public URLs, accepts only those two, and verifies
  pinned checksums. Your copies stay in the local `corpus-cache` volume (and, for
  tests, the git-ignored `backend/corpus/`).
- **Telemetry**: usage telemetry is disabled in every container through each
  component's documented setting: onnxruntime (`ORT_DISABLE_TELEMETRY` and its
  `disable_telemetry_events()` call), the Hugging Face hub (`HF_HUB_DISABLE_TELEMETRY`),
  Qdrant (`QDRANT__TELEMETRY_DISABLED`) and Streamlit (`gatherUsageStats` off). This was
  set from the documentation, not verified by capturing traffic.
- **Outbound traffic**: the only service that reaches the internet is `ingest`, to
  download the two manuals from HP and the pinned models from the Hugging Face hub,
  on its first run or after a pin changes. From Phase 3, the backend will also call
  the Anthropic API.
- **Third-party processing**: no LLM is called yet. From Phase 3, when the cloud
  provider (Anthropic) is enabled, user questions and the retrieved document
  excerpts will be sent to the Anthropic API to generate answers.

## Third-party licenses

The backend and frontend depend only on open-source packages with permissive
licenses (MIT, BSD, Apache-2.0, PSF and similar), with two file-level copyleft
exceptions, both used unmodified from PyPI:

- **tqdm** (MPL-2.0 AND MIT), pulled in by fastembed;
- **certifi** (MPL-2.0), the CA bundle used by httpx and requests.

MPL-2.0 applies to those packages' own files, which can be obtained from PyPI; it does
not extend to this project's code. The embedding model licenses are in
[Embedding models](#embedding-models). The two HP manuals are copyrighted by HP and
are not redistributed.

## Known limitations

- Unhandled server errors (500) carry no CORS headers, because the request ID
  middleware sits outside CORS. The frontend calls the API from its server, so it
  is unaffected.
- The Postgres pool is sized per process and has no explicit `pool_timeout` yet
  (SQLAlchemy's 30 s default applies).
- The frontend image is about 830 MB, mostly Streamlit's own dependencies. The
  backend image carries the ingestion stack (pdfplumber, fastembed, onnxruntime):
  +92 MB compressed (155 MB against 63 MB) and +362 MB on disk (659 MB against
  297 MB) over Phase 0; a separate ingest image is on the Phase 7 list.
- **Button glyphs are lost:** the ENVY manual draws many buttons as inline images. No
  vision pass reads them; ingestion puts an `[icon]` placeholder where one sat inside
  a line (87 in ENVY, 3 in OMEN), so an answer can say a button is pictured but not
  which one ([ADR 0004](docs/adr/0004-pdf-parsing.md)).
- A few chunking artifacts are known and kept by design; they are listed in
  [ADR 0005](docs/adr/0005-chunking-v1.md#known-residue-phase-1-commit-10b-review).
- The Hugging Face hub prints its warning about unauthenticated requests twice
  during the first download: once as plain text, outside the JSON log format, and
  once as a JSON log line.
- The runtime images run `apt-get upgrade` to pick up two zlib security fixes that
  the pinned base image lacks; it goes away with the next base image bump.

The full list, with the reasoning, is in [ADR 0001](docs/adr/0001-architecture-overview.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation: repo, tooling, CI, compose, health endpoints | done |
| 1 | Ingestion: download, parse, clean, chunk, embed, upsert, manifest | done |
| 2 | Retrieval: hybrid search, reranker, golden set, retrieval metrics | pending |
| 3 | Chat API: LLM gateway, prompt, citations, streaming, history | pending |
| 4 | Frontend: Streamlit chat UI with conversations and sources | pending |
| 5 | Benchmark: generation metrics, ablation, report | pending |
| 6 | Load testing: Locust profiles, tuning, capacity report | pending |
| 7 | Hardening: security review, docs, demo script | pending |

## License

MIT, see [LICENSE](LICENSE).
