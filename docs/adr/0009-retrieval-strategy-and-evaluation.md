# ADR 0009: Retrieval strategy and evaluation method

- **Status:** in progress (Phase 2). Commits 3 to 8 set the strategy below; the frozen
  golden set, the baseline, the ablations and the chosen defaults are added at commits
  11 to 14.
- **Date:** 2026-11-13

## Context

Every chat turn (Phase 3) retrieves the chunks the answer must cite, and its relevance
signal decides whether the assistant refuses. The manuals mix prose with exact terms
(part numbers, error codes, button names), so neither dense nor keyword search alone is
expected to be enough. Every knob must be an environment variable, so that Phase 5 can
rerun the ablations without code changes, and the defaults must be chosen on a golden
set frozen before any tuning.

## Decision (strategy, so far)

- **Modes:** `dense`, `sparse` (BM25) and `hybrid` (both, fused with reciprocal rank
  fusion), set by `RETRIEVAL_MODE`, each optionally followed by a cross-encoder rerank
  of `RETRIEVAL_RERANK_CANDIDATES` chunks down to `RETRIEVAL_TOP_K`.
- **Query encoding:** dense through fastembed's `query_embed`, which adds no BGE
  instruction for bge-small (measured: identical to `embed`); an optional instruction is
  the setting `RETRIEVAL_DENSE_QUERY_PREFIX` (empty by default), applied by the
  retrieval service to the dense query only, and ablated after the freeze. BM25 queries
  go through `query_embed` (unit weights; the collection applies IDF).
- **RRF k:** a setting (`RETRIEVAL_RRF_K`), starting at the paper's 60; Qdrant's own
  default of 2 is an ablation.
- **Document filter:** a request may name one manual; otherwise a product-name rule
  (off by default, `RETRIEVAL_PRODUCT_FILTER_ENABLED`) filters when a question names
  exactly one product: a capitalized `ENVY` or `Omen`, or a name in any case followed
  by its model number (`envy 6000`, `omen 17`). Lowercase `envy` and `omen` alone are
  English words and never fire it. The names live in `corpus.toml`.
- **Filter placement:** the filter goes into every prefetch and at the top level of the
  hybrid query. **Finding (Qdrant 1.20, integration test):** with every nearest
  neighbour in doc A and a filter on doc B, the filtered hybrid search returned 10 of
  10 chunks from doc B, and a top-level-only filter also reached the prefetches (10 of
  10 from doc B). The store does not rely on the propagation; it states the filter in
  each prefetch.
- **Relevance signal:** the top reranker score (raw cross-encoder logit) against
  `RETRIEVAL_RELEVANCE_THRESHOLD`, unset until calibrated. RRF scores are rank-based
  and are not a signal; without rerank the signal is unavailable. The threshold belongs
  to one reranker model and revision.
- **Search response:** `POST /v1/search` returns the fused or store score and the rerank
  score per chunk, not per-mode dense and sparse ranks: Qdrant's fused query returns
  only the fused score, and the ranks would cost two more searches per request. H1
  (part numbers under dense-only and hybrid) is shown by the benchmark runner, which
  runs each mode. The router is mounted only when `SEARCH_ENDPOINT_ENABLED` is set
  (otherwise 404; rejected in production). Failures are 503 `retrieval_unavailable` or
  503 `models_not_ready` in the shared error shape, with fixed messages; a 422 never
  echoes input, including the name of an unknown field.
- **Index compatibility:** the API is ready only when the alias names a collection with
  the dense vector of the query model's size (cosine) and the IDF-weighted sparse
  vector, a succeeded build in the manifest, and the build's model files equal to the
  files the API verified. The BM25 `avg_len` affects only document vectors, so any
  recorded value is accepted (null before Phase 2 commit 10b means fastembed's
  defaults).
- **Failures:** every Qdrant failure (down, timeout, missing alias, unexpected payload)
  is `RetrievalUnavailableError`, logged with the stage and the exception type only.

## Evaluation method (to be completed at commit 11)

Golden set file `eval/golden/retrieval-v1.toml`; version, SHA-256, metrics,
selection rule, hypotheses H1 to H3 and their results follow with the freeze.
