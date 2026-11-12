# ADR 0003: Anthropic behind a provider interface, local embeddings

- **Status:** accepted (Phase 0; implemented from Phases 1 and 3)
- **Date:** 2026-11-12

## Context

Answers need a capable LLM, but tests and load tests must run without network
access, cost or non-determinism. Retrieval needs embeddings and reranking for every
chunk and every question; sending all of that to a cloud API would add latency, cost
and another place where document and user text leave the system.

## Decision

- Generation goes through an **LLM provider interface** defined in `domain`. The
  real adapter calls the **Anthropic API**; a **deterministic stub provider**
  serves unit tests and the stub load-test profile. The provider is chosen by
  environment variable.
- The **Anthropic model ID is chosen and verified against the current Anthropic
  documentation in Phase 3**, and set only through an environment variable. Phase 0
  defines no model and no API key setting.
- **Embeddings and reranking run locally on CPU** with fastembed: dense
  `BAAI/bge-small-en-v1.5`, sparse BM25, and a cross-encoder reranker that can be
  switched by environment variable. CPU-heavy work runs in a thread pool, off the
  event loop.

## Alternatives considered

- **Call Anthropic directly from services:** less code, but every test would need
  network mocks, and the load test could not separate system capacity from LLM
  latency.
- **Hosted embedding APIs:** better models are available, but they add per-request
  cost, latency and data exposure, for a corpus small enough to embed locally.
- **A self-hosted LLM:** keeps all data local, but needs a GPU to answer in
  acceptable time, which this deployment does not assume.

## Consequences

- Unit tests and the stub load-test profile need no network or API key, and they
  are deterministic.
- With the Anthropic provider enabled, user questions and the retrieved excerpts
  are sent to the Anthropic API; the README states this.
- Embedding models are downloaded at build or first start and cached; they must
  never be committed.
- Swapping or adding an LLM provider means one new adapter, not changes in services.
