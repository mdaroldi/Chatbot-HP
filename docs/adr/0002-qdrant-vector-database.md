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
