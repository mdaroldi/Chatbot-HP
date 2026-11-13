# ADR 0009: Retrieval strategy and evaluation method

- **Status:** in progress (Phase 2). Commits 3 to 8 set the strategy below; the golden
  set is frozen at commit 11b; the baseline, the ablations and the chosen defaults
  follow at commit 12.
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
  recorded value is accepted.
- **BM25 avg_len (commit 10b):** `EMBEDDING_BM25_AVG_LEN` is a number or `auto`, the
  mean BM25 length of the chunks being built: stemmed tokens without stopwords or
  punctuation, the `|d|` of fastembed's formula, measured with fastembed's own steps.
  The default stays at fastembed's 256, so the index is unchanged; `auto` measures
  107.26 (30,783 tokens over 287 chunks, chunking v2). Every build passes k, b and
  avg_len to fastembed and records them in `index_builds.sparse_params` (migration
  0004, with an `avg_len_auto` flag); NULL on older rows means fastembed's defaults.
  `--if-needed` compares a fixed value with the recorded one. `auto` cannot be measured
  on the skip path, which loads no model, so it reuses only a build that was measured
  too: its mean follows from the documents, chunking version and pins already compared.
  The API ignores the setting (queries carry unit weights). Verified live: a forced
  rebuild with the default kept the chunk-ID digests and all 287 stored sparse vectors
  of the previous build.
- **Failures:** every Qdrant failure (down, timeout, missing alias, unexpected payload)
  is `RetrievalUnavailableError`, logged with the stage and the exception type only.

## Relevance threshold (calibration record)

Calibrated on `Xenova/ms-marco-MiniLM-L-6-v2` @ `a09144355adeed5f58c8ed011d209bf8ee5a1fec`, threshold `-6.54`
(run `20261113t153722z-06bf4c0280ab`, dev split, build `20261113t153253z`), reproduced
exactly after the tie fix by runs `20261113t170151z-e417067988f3` and
`20261113t171404z-e417067988f3`. A unit test fails when the reranker pin moves without
this record. Results follow below.

## Finding: RRF ties at the cutoff (fixed at `e417067`)

Qdrant broke score ties at its own `limit` differently between runs; RRF scores are
rank-based, so ties are common. The store now asks for the union of both prefetch
limits (in hybrid mode, every fused point) and cuts client side by score descending,
then chunk id ascending, in every mode. Two identical post-fix full-matrix runs differ
on 0 of 56 candidate lists for every configuration. Pre-fix evidence of the noise
(runs on `1ae6ccc` and `06bf4c0`, kept in `eval/reports/runs/`, git-ignored):

- Between identical runs, the reference's candidate set differed on 10 to 12 of 56 dev
  questions: one chunk swapped at position 20 among equal fused scores. No MiniLM
  score changed.
- jina-turbo's Recall@5 moved from 0.948 to 0.927 between two runs (post-fix: 0.948).
- The reference's total p95 over the five pre-fix runs: 2275, 3577, 2392, 2348 and
  2170 ms. The first three set the 1302 ms tie margin.
- The noise also reached the order of plain `hybrid` (47 of 56 lists reordered between
  two runs) and `sparse` (7), and inflated the "changed by the setting" counts of RRF
  k = 2 (10 and 8, post-fix 0) and the product filter (14 and 17, post-fix 7).

The post-fix runs are the record in `eval/reports/retrieval-summary.md`. Only plain
`hybrid` moved (Recall@5 0.878 to 0.889, MRR 0.841 to 0.852); the selection, the
threshold and every default are unchanged.

## Evaluation method

**Frozen golden set (commit 11b):** `eval/golden/retrieval-v1.toml`, version **1.0.1**,
SHA-256 `91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416`. A unit test
pins the hash and the version; changing the file needs a new version, a new record
here and the pin moved with it. 70 questions: 56 dev (48 in scope, 8 out of scope: 4
unrelated, 4 near-domain) and 14 held-out. Preflight on build `20261113t134027z` (287
chunks): 74 anchors, match counts {1: 58, 2: 13, 3: 3}, 16 anchors with two or more
matches, none unresolved or over five. Metrics, selection rule and the results of H1
to H3 follow with the baseline (commit 12).

**Before the freeze (record):**

- **Commit of v1.0.0.** The file was committed unchanged at `fa0526d`, before the
  review of its multi-match anchors and before any metric was read. Committing it is
  not the freeze: the freeze is this ADR's version and SHA-256 record plus the pin
  test (commit 11b). A v1.0.1, if the review asks for one, lands as its own commit
  before 11b.
- **One pre-freeze smoke run.** In commit 10, `hybrid-rerank` ran once on the 56 dev
  questions to prove the Docker path of the benchmark. Its metrics were not read and
  its output was deleted. No other configuration ran on dev before the freeze.
- **v1.0.0 replaced by v1.0.1.** After the multi-match review, the planner judged 15
  of the 18 multi-match anchors legitimate and tightened three false positives
  (omen-prereq-005 anchor 0, both-ambiguous-003 anchor 0, both-ambiguous-005 anchor 1;
  match counts 2 to 1, 2 to 1 and 3 to 2). v1.0.0 (`fa0526d`, SHA-256
  `d4caaefff162f689055ccfb1ffc14d7bb5ec29885110235b890104551c4fd588`) was replaced
  unchanged by the delivered v1.0.1 at commit `69cb00b`. No metric was read for either
  version.
- **Multi-match review (commit 11a).** The preflight also lists every anchor that
  matches two or more chunks, with its question id, anchor index, match count and the
  section path of each matching chunk (no chunk text), so false positive coverage can
  be reviewed before the freeze.
