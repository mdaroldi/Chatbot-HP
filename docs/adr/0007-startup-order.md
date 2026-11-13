# ADR 0007: Startup order with one-shot migrate and ingest services

- **Status:** accepted (Phase 1, commit 10b)
- **Date:** 2026-11-13

## Context

The challenge asks for one command, `docker compose up`, to run everything. The API
needs a migrated database, and from Phase 2 it needs a live index, so a fresh stack
must migrate Postgres and build the Qdrant index before the backend serves traffic.
Until now the backend entrypoint ran `alembic upgrade head`, which every replica
would repeat (ADR 0001), and ingestion had no compose service at all.

An index build downloads the two PDFs and the embedding models, then parses, embeds
and upserts: about a minute of work. It must not run on every `up` when nothing has
changed, and it must never leave the API on a partial index.

## Decision

**Order:**

1. `postgres` and `qdrant` start and become healthy.
2. **`migrate`** (backend image, `alembic upgrade head`) runs once after Postgres is
   healthy.
3. **`ingest`** (backend image, `python -m app.ingestion --if-needed`) runs once after
   `migrate` completed successfully and Qdrant is healthy.
4. **`backend`** waits for both with `service_completed_successfully`; its entrypoint
   only starts uvicorn.
5. `frontend` waits for a healthy backend.

**The `--if-needed` rule:** the build is skipped (exit 0, no download, no model load)
only when the live alias's latest successful manifest matches all of: the document
hashes in `corpus.toml`, the chunking version, the dense and sparse model names,
their pins in `models.toml` (revision and every file digest), and a collection point
count equal to the manifest's chunk count. Any difference rebuilds, and the run logs
the first one as its reason. `make ingest` runs the same service with `--force`: it
always builds a new collection and moves the alias, while the API keeps serving the
old one until the swap.

**One-shot services** use the backend image and the shared hardening (non-root
10001, read-only root, all capabilities dropped, no new privileges, `/tmp` tmpfs),
with two overrides: `restart: "no"`, because a finished job must stay finished, and
the image healthcheck disabled, because it probes the API, which these containers
never start. `migrate` is on the internal data network only. `ingest` is on both
networks, since it downloads from HP and the Hugging Face hub, and it keeps the
verified PDFs and model files in two named volumes, `corpus-cache` and
`model-cache`. The image creates both mount points owned by uid 10001: a named volume
copies its mount point's owner on first use. huggingface-hub and hf-xet keep their
own cache and logs under `HF_HOME`, which points to the tmpfs. With the default
(`~/.cache`) the first download failed, because the root is read-only and the user
has no home. The verified files still go to `EMBEDDING_CACHE_DIR`.

**Settings** reach both services as environment variables. `ingest` gets every
setting it reads; the hosts and the two cache paths are fixed by the stack. A unit
test derives the list from `Settings` and checks that the two match.

## Alternatives considered

- **Migrations and ingestion in the backend entrypoint:** one service fewer, but
  every replica migrates and builds, and the API container needs internet access.
- **Ingestion only on demand (`make ingest`):** a fresh `up` would start an API with
  no index, and the README promise of one command would not hold.
- **Ingest in the background while the backend starts:** the API would answer before
  the index exists and would have to report not ready for minutes; the
  `service_completed_successfully` gate is simpler and visible in `docker compose ps`.
- **A compose `profiles: [tools]` service for ingestion:** it would not run on a
  plain `up`; the stack has no profiles.

## Consequences

- `docker compose up` on a fresh clone ends with a migrated database and a live,
  verified index; later runs skip the build in a few seconds and need no network.
- **Trade-off accepted:** the first `up` needs network access and takes minutes. If
  a download or checksum fails, `ingest` exits 1 and the backend does not start, by
  design: no API on an unverified or missing index. `docker compose logs ingest`
  shows the one structured error line. The README states both, with the measured
  first-up time (Phase 1, commit 12).
- With several API replicas (Phase 6), migrations and ingestion still run once per
  `up`, not once per replica.
- Two named volumes hold downloaded data; `docker compose down -v` removes them and
  the next `up` downloads again.
