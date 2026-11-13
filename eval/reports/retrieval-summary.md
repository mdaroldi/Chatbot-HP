# Retrieval benchmark: Phase 2 summary

Assembled from the run summaries under `eval/reports/runs/` (git-ignored). It holds
identifiers, scores and timings only. Method and decisions:
[ADR 0009](../../docs/adr/0009-retrieval-strategy-and-evaluation.md).

## Setup

- **Golden set:** `eval/golden/retrieval-v1.toml` v1.0.1, SHA-256
  `91e754f19cbafb80ea0e1a120f386e5ce654c97daf1e5b5f0d8102f549e70416`. Dev split only:
  56 questions (48 in scope, 8 out of scope: 4 unrelated, 4 near-domain). The held-out
  split (14) was not scored.
- **Index:** chunking v2, 287 chunks, BM25 k 1.2, b 0.75. The avg_len is 256 (fixed)
  unless a row says otherwise. Every build used here held the same chunk IDs.
- **Models:** `BAAI/bge-small-en-v1.5` @ `aa8f8b060edb`, `Qdrant/bm25` @ `22b8d2af71a7`,
  `Xenova/ms-marco-MiniLM-L-6-v2` @ `a09144355ade`, `jinaai/jina-reranker-v1-turbo-en`
  @ `b8c14f4e723d`.
- **Hardware:** Apple M2, Docker Desktop, aarch64, 8 CPUs; inference workers 2, ONNX threads 2.
- **Scoring:** over the candidate list (top 20). Bootstrap 95% intervals: seed
  20261113, 2000 resamples. Latency is wall time per retrieval in process (embed,
  search, rerank). Each reranker is warmed with 20 pairs at its 512-token limit, then
  one uncounted retrieval runs per configuration.
- **Selection rule:** highest Recall@5, where overlapping intervals count as tied; then
  higher MRR; then lower p95, where a gap of up to 1302 ms is a tie; then matrix order.

Every number below is from the post-fix runs (code `e417067`, deterministic candidate
lists; see "Latency and determinism"). The pre-fix runs are kept as the evidence of the
tie noise and of the p95 tie margin; their numbers are recorded in ADR 0009.

| Run | Code | Build | Configs |
|---|---|---|---|
| `20261113t143340z-1ae6cccf3956` | `1ae6ccc` | `20261113t134027z` | pre-fix: four-way baseline |
| `20261113t143602z-1ae6cccf3956` | `1ae6ccc` | `20261113t134027z` | pre-fix: reference + four single changes |
| `20261113t144854z-1ae6cccf3956` | `1ae6ccc` | `20261113t134027z` | pre-fix: the three tied changes + reference, reverse order |
| `20261113t153000z-1ae6cccf3956` | `1ae6ccc` | `20261113t145757z` (avg_len `auto`), by `--collection` | pre-fix: reference |
| `20261113t153722z-06bf4c0280ab` | `06bf4c0` | `20261113t153253z` | pre-fix: full matrix, amended rule |
| `20261113t170151z-e417067988f3` | `e417067` | `20261113t153253z` | full matrix (**the record for the tables below**) |
| `20261113t171404z-e417067988f3` | `e417067` | `20261113t153253z` | full matrix again (determinism check) |
| `20261113t172817z-e417067988f3` | `e417067` | `20261113t172701z` (avg_len `auto`, 107.26), by `--collection` | reference |

## Metrics (run `20261113t170151z-e417067988f3`)

| config | Recall@5 [95% CI] | MRR [95% CI] | Recall@20 | nDCG@5 | p50 / p95 ms |
|---|---|---|---|---|---|
| `dense` | 0.899 [0.819, 0.965] | 0.827 [0.742, 0.910] | 0.927 | 0.821 | 11 / 15 |
| `sparse` | 0.906 [0.826, 0.969] | 0.793 [0.704, 0.877] | 0.944 | 0.795 | 3 / 16 |
| `hybrid` | 0.889 [0.805, 0.962] | 0.852 [0.769, 0.927] | 0.976 | 0.831 | 11 / 13 |
| `hybrid-rerank` | 0.941 [0.889, 0.986] | 0.898 [0.826, 0.958] | 0.976 | 0.881 | 2039 / 2644 |
| `hybrid-rerank-jina-turbo` | 0.948 [0.899, 0.990] | 0.854 [0.779, 0.924] | 0.976 | 0.853 | 2703 / 3338 |
| `hybrid-rerank-product-filter` | 0.941 [0.889, 0.986] | 0.898 [0.826, 0.958] | 0.976 | 0.881 | 2427 / 2961 |
| `hybrid-rerank-bge-prefix` | 0.941 [0.889, 0.986] | 0.898 [0.826, 0.958] | 0.976 | 0.881 | 2514 / 3035 |
| `hybrid-rerank-rrf-k2` | 0.941 [0.889, 0.986] | 0.898 [0.826, 0.958] | 0.976 | 0.881 | 2346 / 3049 |

The second run (`171404z`) gave the same quality metrics on every row. Its p95 values
were 14, 15, 19, 2937, 3968, 3009, 3636 and 3097 ms in the order above, and it made the
same selection. The avg_len `auto` build (run `172817z`, `hybrid-rerank` only) scored
exactly the same as the reference: Recall@5 0.941, MRR 0.898, Recall@20 0.976, nDCG@5
0.881, p50 / p95 2595 / 2951 ms.

**Selected: `hybrid-rerank`.** All eight rows tie on Recall@5 (overlapping intervals).
Five share the top MRR (0.898): the reference, product filter, BGE prefix, RRF k = 2
and the avg_len build, which was run separately. Their p95 values fall within the
margin, so matrix order decides. jina-turbo has the highest Recall@5 point estimate
(0.948, tied) and loses on MRR (0.854).

## Recall@5 / MRR by category, four-way baseline (run `170151z`)

| category | n | `dense` R@5 / MRR | `sparse` R@5 / MRR | `hybrid` R@5 / MRR | `hybrid-rerank` R@5 / MRR |
|---|---|---|---|---|---|
| factual | 10 | 1.000 / 0.853 | 0.800 / 0.554 | 0.900 / 0.681 | 1.000 / 0.875 |
| procedure | 8 | 1.000 / 1.000 | 1.000 / 0.938 | 1.000 / 1.000 | 1.000 / 1.000 |
| table | 6 | 0.833 / 0.667 | 1.000 / 0.833 | 0.833 / 0.769 | 1.000 / 1.000 |
| part_number | 6 | 1.000 / 0.833 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| warning | 5 | 0.800 / 0.667 | 1.000 / 0.717 | 0.800 / 0.695 | 1.000 / 0.750 |
| prerequisite | 5 | 0.433 / 0.800 | 0.500 / 0.900 | 0.533 / 0.900 | 0.433 / 0.900 |
| ambiguous | 4 | 1.000 / 0.875 | 1.000 / 0.688 | 1.000 / 1.000 | 1.000 / 0.708 |
| follow_up | 4 | 1.000 / 0.833 | 1.000 / 0.800 | 1.000 / 0.875 | 1.000 / 0.812 |

Against the pre-fix baseline, only `hybrid` without rerank moved (factual MRR 0.679 to
0.681, prerequisite Recall@5 0.433 to 0.533, ambiguous MRR 0.875 to 1.000): its order
among tied fused scores is now fixed by chunk id.

## Hypotheses

- **H1 (sparse and hybrid beat dense on `part_number` and `table`): partly supported,
  weak evidence (n = 6 each).** On `part_number`, Recall@5 is 1.000 for every mode
  (ceiling), and MRR is 1.000 for sparse and hybrid against 0.833 for dense. On `table`,
  sparse beats dense on Recall@5 (1.000 vs 0.833) and on MRR (0.833 vs 0.667). Hybrid
  ties dense on Recall@5 (0.833) and is slightly ahead on MRR (0.769 vs 0.667). Rerank
  lifts `table` to 1.000 / 1.000.
- **H2 (rerank improves MRR over hybrid): the direction holds, not significant.** MRR
  is 0.898 [0.826, 0.958] against 0.852 [0.769, 0.927]; the intervals overlap. Recall@5
  rises from 0.889 to 0.941, and nDCG@5 from 0.831 to 0.881.
- **H3 (the product filter does not lower Recall@5): holds, with no effect.** The rule
  fired on 12 dev questions (9 in scope, 3 near-domain). That matches the rule applied
  offline to the dev questions and a case-insensitive name search (also 12 and 9). On
  7 of the 12, the filter changed the candidate set (about 2.9 chunks dropped per
  changed question), but no question's score or top rerank score changed.

## Single changes from `hybrid-rerank`

No single change improved dev quality, so the matrix has no combination row.

- **BGE query prefix, RRF k = 2, product filter, avg_len `auto`:** every metric is
  identical to the reference.
- **jina-turbo reranker:** lower MRR (0.854 against 0.898) and higher p95.

**Check that each setting was applied.** Counts of dev questions (of 56) whose candidate
list or top 5 differs from the reference (run `170151z`):

| Change | Candidate list differs | Top 5 differs | Top rerank score differs |
|---|---|---|---|
| reference vs itself, run `171404z` (noise) | 0 | 0 | 0 |
| BGE prefix | 45 | 8 | 1 |
| product filter | 7 | 1 | 0 |
| RRF k = 2 | 0 | 0 | 0 |
| avg_len `auto` (run `172817z`) | 39 | 6 | 0 |
| jina-turbo | 0 (same set, reordered) | 55 | 56 |

- **RRF k = 2:** applied. Its fused candidate scores range from 0.048 to 1.0,
  against 0.013 to 0.033 at k = 60. The fused top 20 holds the same chunks as at k = 60
  on all 56 questions (measured; with up to 40 fused points it could differ), so only
  the order before the rerank changes, and the reranker reorders the same 20. Before
  the fix, its 10 and 8 changed lists were tie noise.
- **Product filter:** applied; it changes the candidate set on 7 of the 12 questions
  where it fired (pre-fix: 9, inflated by the noise).
- **The changed top 5s:** under BGE prefix and avg_len `auto`, the top 5 changes on
  some questions but no anchor score does.

## Relevance threshold

Calibrated on `hybrid-rerank` (MiniLM-L6 @ `a09144355ade`), run `20261113t153722z`
(pre-fix), and reproduced exactly by the post-fix runs `170151z` and `171404z`. The
rule is the highest top rerank score that keeps false refusals at or below 5% of
in-scope questions (at most 2 of 48).

- **Threshold:** -6.539, set as **-6.54**. Rounding down refuses the same questions.
- **In scope (n 48):** 2 refused (one `table`, one `ambiguous`). Top scores: min -8.152,
  p25 0.510, median 3.623, p75 5.255, max 8.693.
- **Unrelated (n 4):** 4 caught. Top scores -11.173 to -9.001, median -10.828.
- **Near-domain (n 4):** 0 caught. Top scores 2.196 to 6.563, median 2.949.

**Fragile.** It rests on 8 out-of-scope questions, and it catches only unrelated ones:
near-domain questions score like in-scope ones. It belongs to one reranker model and
revision. With jina-turbo, the threshold would be -1.546, with 0 of 4 near-domain caught.

## Latency and determinism

- **Latency is method-limited.** A laptop VM has order and load effects. The
  reference's total p95 was 2275, 3577, 2392, 2348 and 2170 ms over the five pre-fix
  runs, and 2644, 2937 and 2951 ms over the three post-fix runs; its p50 ranged from
  1833 to 2595 ms. The tie margin (1302 ms) is the spread of the first three. All eight
  spread to 1407 ms, which changes no selection. In run `170151z`, rerank was about 97%
  of the time (p50 1986 of 2039 ms). Latency decisions are deferred to the Phase 6 load
  tests.
- **Candidate lists are deterministic (fix `e417067`).** Before it, Qdrant broke score
  ties at its own limit differently between runs: the reference's candidate set
  differed on 10 to 12 of 56 questions between identical runs, and jina-turbo's
  Recall@5 moved from 0.948 to 0.927. The store now asks for the union of both
  prefetches (every fused point) and cuts client side by score descending, then chunk
  id ascending, in every mode. Between the two post-fix full-matrix runs, 0 of 56
  candidate lists differ for every configuration, in order and in every score.
