# ADR 0002: Qdrant as the vector database

- **Status:** accepted (Phase 0; used from Phase 1)
- **Date:** 2026-11-12

## Context

The challenge requires an open-source vector database built only from the two HP
documents. The search strategy (CLAUDE.md) is hybrid: dense embeddings
(`BAAI/bge-small-en-v1.5`) plus BM25 sparse vectors, fused with reciprocal rank
fusion, then reranked. The manuals are full of exact terms (error codes, part
names, button labels) where keyword matching matters as much as semantics. Chunks
carry metadata (document, product, section path, page range, content type) that
retrieval must filter on.

## Decision

Use **Qdrant** (v1.20, unprivileged image), with dense and sparse vectors as named
vectors in one collection and payload metadata on each point. It runs in its own
container on the internal data network, with a persistent volume. Phase 0 only
checks readiness through its REST `/readyz` endpoint, using httpx with explicit
timeouts; the Qdrant client library arrives with ingestion in Phase 1.

## Alternatives considered

- **pgvector in the existing Postgres:** one fewer service, but no native sparse
  vectors or built-in fusion; BM25 would mean a separate full-text index and fusion
  in application code.
- **Weaviate / Milvus:** capable, but heavier to run for a corpus of two documents.
- **Chroma:** simple, but weaker hybrid search and filtering, and less suited to
  showing production-style operations.
- **FAISS in process:** no server, no payload filtering, and the index would live
  inside every API replica.

## Consequences

- Hybrid search and fusion happen in one query instead of glue code.
- One more stateful service to operate: it needs its storage and snapshots volumes
  (the read-only root filesystem forced the snapshots volume) and has telemetry
  disabled.
- Readiness depends on Qdrant answering 200 on `/readyz`; any other status, a
  timeout or a connection error reports the backend as not ready.

## Index builds (Phase 1, commit 8)

- **One collection per build**, `hpdocs_chunks_v1_<build id>` (`v1` is the chunking
  version; see Corrections after implementation). Named vectors `dense` (cosine) and `sparse` with `modifier=IDF`, because BM25
  vectors carry term frequencies only. Keyword payload indexes on `doc_id` and
  `content_type`.
- **Readers use an alias** (`QDRANT_COLLECTION_ALIAS`). A build goes live only after its
  point count equals its chunk count. A lower count also catches duplicate chunk IDs,
  since Qdrant keeps one point per ID. The alias then moves in one atomic request.
- **Failure:** before the swap, the new collection is deleted and the alias is left
  untouched. If the swap reports an error, the collection is deleted only when the
  alias does not point to it. A swap that succeeded on the server but lost its reply
  stays live.
- **Rollback:** with `INDEX_KEEP_PREVIOUS=true`, the previous alias target survives the
  swap. Every other collection under the prefix is deleted, and nothing outside the
  prefix is touched.
- **Local mode:** qdrant-client's in-memory mode supports aliases and the IDF modifier,
  so unit tests run on it. It ignores payload indexes, which integration tests check
  on the server.

## Corrections after implementation

### Collection names and pruning (Phase 1, commit 10)

- **The version in the name is the chunking version**, not a collection schema:
  `hpdocs_chunks_<chunking_version>_<build id>`, with the build ID as the lowercase UTC
  start time (`20261112t143005z`). A new chunking version gets a new prefix. Pruning
  stays inside the new build's prefix, so collections of an older chunking version are
  never deleted automatically; the alias still moves, and they are removed by hand.
- **A failed prune is a warning.** Pruning runs after the swap, so the new build is
  already live. A collection that cannot be deleted is logged (collection and error
  type, never the message), recorded as a `prune_failed` warning in the build's
  manifest row, and tried again by the next build; the other old builds are still
  pruned and ingest exits 0.
- **No client compatibility check at runtime.** The ingest CLI builds the client with
  `check_compatibility=False`: the check is a background request that would run even
  when the index is never used, and it reports outside the JSON logs. The server
  image and the client are pinned together, and an integration test checks them.
