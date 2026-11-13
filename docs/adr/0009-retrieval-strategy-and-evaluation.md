# ADR 0009: Retrieval strategy and evaluation method

- **Status:** accepted (Phase 2). Commits 3 to 8 set the strategy below; the golden
  set was frozen at commit 11b; the baseline, the ablations and the chosen defaults are
  in [Results](#results-phase-2-record), from
  [the benchmark summary](../../eval/reports/retrieval-summary.md).
- **Date:** 2026-11-13

## Context

Every chat turn (Phase 3) retrieves the chunks the answer must cite, and its relevance
signal decides whether the assistant refuses. The manuals mix prose with exact terms
(part numbers, error codes, button names), so neither dense nor keyword search alone is
expected to be enough. Every knob must be an environment variable, so that Phase 5 can
rerun the ablations without code changes, and the defaults must be chosen on a golden
set frozen before any tuning.

## Decision (strategy)

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
  `RETRIEVAL_RELEVANCE_THRESHOLD`, calibrated at -6.54 (record below). RRF scores are
  rank-based and are not a signal; without rerank the signal is unavailable. The threshold belongs
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
threshold and every default are unchanged. After the fix, RRF k = 2 changes 0
candidate lists: the earlier 10 and 8 were tie noise.

**Deterministic per index build, not across builds.** Within one build, two
full-matrix runs differ on 0 of 56 candidate lists for every configuration. Across two
builds with the same chunk IDs and vectors (`20261113t153253z` and
`20261113t173108z`, one informational run on the second, deleted after its counts
were recorded), 4 of 56 plain `hybrid` lists differ, and no metric changes. Dense
lists are identical. Sparse lists are identical in order and set, but 80 of the 1,060
sparse scores differ, by at most 2.1e-7 relative (float32). The denominator is 1,060,
not 56 × 20 = 1,120, because BM25 returns only chunks that share a term with the
question: 51 sparse lists hold 20 chunks, one 18, one 16 and three 2 (the kept runs
`170151z` and `171404z` show the same lengths). Nine sparse lists hold tied scores,
and Qdrant orders those ties inside its own prefetch, which the client-side cut
cannot reach: the likely source of the 4 hybrid differences.

- **Candidate, not implemented (Phase 5 or 6):** client-side fusion, that is, a dense
  and a sparse search with RRF computed in the store adapter under the same
  tie-break. It costs a second search per request. Ask before doing it.
- **Rule:** every final evaluation runs on a single index build, and the build id is
  recorded in the metadata of each run. The runner already writes `index.build_id` to
  `summary.json` and to the summary header; the rule makes it a requirement.

## Evaluation method

**Frozen golden set (commit 11b):** `eval/golden/retrieval-v1.toml`, version **1.0.1**,
SHA-256 `91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416`. A unit test
pins the hash and the version; changing the file needs a new version, a new record
here and the pin moved with it. 70 questions: 56 dev (48 in scope, 8 out of scope: 4
unrelated, 4 near-domain) and 14 held-out. Preflight on build `20261113t134027z` (287
chunks): 74 anchors, match counts {1: 58, 2: 13, 3: 3}, 16 anchors with two or more
matches, none unresolved or over five; unchanged on the live build
`20261113t173108z`. Metrics, selection rule and the results of H1 to H3 are in
[Results](#results-phase-2-record). The held-out split was never scored in Phase 2:
the runner refuses `--split heldout` without `--allow-heldout`
(`test_the_heldout_split_is_refused_without_the_flag` in `test_eval_runner.py` and
`test_eval_cli.py`), and every kept run records the dev split.

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

## Results (Phase 2 record)

Every number is from run `20261113t170151z-e417067988f3` (code `e417067`, build
`20261113t153253z`, dev split, 56 questions) unless stated otherwise. The tables, the
per-category scores and the timings are in
[`eval/reports/retrieval-summary.md`](../../eval/reports/retrieval-summary.md).

**Selected default: `hybrid-rerank`.** Hybrid dense + BM25, RRF k 60, then the
MiniLM-L6 cross-encoder reranks 20 candidates down to 5. No dense query prefix,
product filter off, BM25 avg_len 256.

- **Recall@5 is a tie, not equal scores.** "All rows tie on Recall@5" means a tie by
  the selection rule (overlapping bootstrap intervals). The point estimates range from
  0.889 (`hybrid`) to 0.948 (`hybrid-rerank-jina-turbo`).
- **MRR:** five configurations share the top MRR, 0.898 [0.826, 0.958]: the reference,
  the product filter, the BGE prefix, RRF k = 2 and the avg_len `auto` build (run
  `172817z`). Their p95 values fall within the tie margin, so matrix order decides, and
  the reference comes first.
- **Selection amendment, made after seeing results.** The p95 tie margin was added to
  the rule (commit `06bf4c0`) after the first runs had been read: a p95 gap within
  `[selection] p95_tie_margin_ms` (1302 ms) is a tie, and the earliest configuration
  within the margin of the fastest wins. 1302 ms is the spread of the reference's p95
  over its first three runs (2275, 3577 and 2392 ms). Later runs widened that spread to
  1407 ms (eight runs, 2170 to 3577 ms). The selection is the same with either value;
  the margin was not moved.
- **jina-turbo:** removed at commit 13; the record is in ADR 0008, decision 5.

**Hypotheses:**

- **H1 (sparse and hybrid beat dense on `part_number` and `table`): partly supported,
  weak evidence (n = 6 each).** `part_number`: Recall@5 is at the ceiling (1.000 for
  every mode); MRR is 1.000 for sparse and hybrid against 0.833 for dense. `table`:
  sparse 1.000 / 0.833 (Recall@5 / MRR) against dense 0.833 / 0.667; hybrid ties dense
  on Recall@5 (0.833 / 0.769).
- **H2 (rerank improves MRR over hybrid): the direction holds, not significant.** MRR
  0.898 [0.826, 0.958] against plain hybrid 0.852 [0.769, 0.927]; the intervals
  overlap. Plain hybrid's Recall@5 is 0.889.
- **H3 (the product filter does not lower Recall@5): holds, with no effect.** The rule
  fires on 12 dev questions (9 in scope, 3 near-domain) and changes the candidate list
  on 7 of the 12; no question's score or top rerank score changes.

**Calibration:** -6.539, set as -6.54. In scope, 2 of 48 are refused (one `table`, one
`ambiguous`); 4 of 4 unrelated questions are caught, 0 of 4 near-domain ones. Fragile:
it rests on 8 out-of-scope questions and belongs to one reranker model and revision.

**RRF k (informational check).** k changes only the order before the rerank, which the
reranker masks. Plain hybrid without rerank, k = 60 against k = 2: the top-20 order
differs on 49 of 56 dev questions, the top-5 order on 44, the top-20 set on 0. Without
rerank, k = 60 scores Recall@5 0.889 / MRR 0.852 and k = 2 scores 0.927 / 0.849; 8
questions change score. This was a single informational run on dev, made after the
selection and deleted after its counts were recorded. It was not used for any default:
k stays 60, and the shipped pipeline always reranks.

## Consequences

- **Masked ablations.** The BGE prefix, RRF k, avg_len and the product filter were
  measured behind the reranker, which reorders the same top 20 and can hide their
  effect. If a later phase turns reranking off for latency, re-measure them.
- **The reranker's value rests on weak evidence** (H2 not significant, the intervals
  overlap) and it costs about 2 s p50 per query: the rerank is about 97% of retrieval
  time (p50 1986 of 2039 ms). Revisit it against answer quality in Phase 5 and against
  load numbers in Phase 6. Latency decisions are deferred to Phase 6.
- **The threshold catches unrelated questions (4 of 4), not near-domain ones (0 of
  4).** Near-domain refusal must come from the Phase 3 prompt, and its rate is a Phase 5
  metric. Changing the reranker means recalibrating.
- **Long queries (Phase 3).** A 1,000-character query is truncated together with 55 of
  the 287 chunks at the reranker's 512 tokens (ADR 0008); the chat message limit takes
  this as an input.
- **Evaluation hygiene (Phase 5).** Every final evaluation runs on a single index build
  with its build id recorded (rule above). The held-out split is scored only then.
