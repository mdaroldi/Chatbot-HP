# Answer benchmark summary

Generated from each run's report.json by `make eval-summary`; ids and numbers only.

## Reference runs and noise floor (answers-v1 items)

| metric | reference-1 | noise floor | n |
|---|---|---|---|
| primary (strict key-fact pass) | 45/48 | 0.021 | 48 |
| false refusals (in scope) | 2/48 | 0.021 | 48 |
| near-domain refusal recall | 4/4 | 0.250 | 4 |

Incomplete: fewer than three reference runs.

## Ablation (rule 7 against reference-1)

| run | env | primary | difference vs reference-1 [95%] | gate subset ref -> alt | false refusals | near-domain recall | layer 1 refused | verdict | reasons |
|---|---|---|---|---|---|---|---|---|---|
| improvement | default | not run |  |  |  |  |  |  |  |
| hybrid-no-rerank | RETRIEVAL_RERANK_ENABLED=false | not run |  |  |  |  |  |  |  |
| dense | RETRIEVAL_MODE=dense, RETRIEVAL_RERANK_ENABLED=false | not run |  |  |  |  |  |  |  |
| naive | QDRANT_COLLECTION_ALIAS=hpdocs_chunks_naive | not run |  |  |  |  |  |  |  |
| k3 | RETRIEVAL_TOP_K=3 | not run |  |  |  |  |  |  |  |
| sparse | RETRIEVAL_MODE=sparse, RETRIEVAL_RERANK_ENABLED=false | not run |  |  |  |  |  |  |  |
| k8 | RETRIEVAL_TOP_K=8 | not run |  |  |  |  |  |  |  |

## Rewrite off (descriptive)

Not run.

## Held-out (final default)

Not run yet (rule 8: once, last).

## Judge (reference-1)

| metric | k/n | rate | Wilson 95% |
|---|---|---|---|
| faithful, lenient (no unsupported claim) | 67/67 | 1.000 | [0.946, 1.000] |
| faithful, strict (every claim supported) | 65/67 | 0.970 | [0.898, 0.992] |

Correctness: correct 64, partial 2, incorrect 1. Judge errors: none.

| agreement | n | agreed | raw | kappa |
|---|---|---|---|---|
| repeatability, faithful | 20 | 20 | 1.000 | n/a |
| repeatability, correctness | 20 | 20 | 1.000 | 1.000 |
| human labels | not measured |  |  |  |

Negative controls (5 corrupted answers): correctness flagged 5, lenient faithfulness 4, strict faithfulness 5. Missed by lenient: nc-omen-procedure-001.

## Failures of reference-1 by cause

| cause | items | ids |
|---|---|---|
| primary_refused_layer1 | 2 | envy-table-003, both-ambiguous-001 |
| primary_refused_model | 0 | - |
| primary_missing_facts | 1 | envy-procedure-001 |
| slice_refused_layer1 | 2 | lq-001, lq-004 |
| slice_refused_model | 1 | lq-003 |
| slice_missing_facts | 2 | lq-006, lq-007 |
| out_of_scope_answered | 0 | - |
| adversarial_failed | 1 | adv-011 |
| forbidden_hit | 1 | adv-011 |
| warning_not_carried | 2 | envy-warning-002, lq-008 |
| rewrite_not_clean | 2 | mt-002, mt-009 |

## Reference run 1 (full dev set)

- run: reference-1
- provider: anthropic
- items: 105 of 105
- env: default
- models: answer claude-sonnet-5-5, rewrite claude-haiku-5-5, judge claude-opus-5-5 (judge-v1 6c8d87315639)
- index_build_ids: 20261113t173108z
- datasets: retrieval-v1.toml 1.0.1 91e754f19cba, answers-v1.toml 1.0.0 2aac464af1c6, slices-v1.toml 1.0.0 b9ec7a354bb6
- git_sha: ee80580f74fab26dd733221eeddcfa0651117079
- health: valid, 0 errors

| metric | k/n | rate | Wilson 95% |
|---|---|---|---|
| primary (strict key-fact pass) | 45/48 | 0.938 | [0.832, 0.979] |
| false refusals (in scope) | 5/72 | 0.069 | [0.030, 0.152] |
| out-of-scope recall, unrelated | 11/11 | 1.000 | [0.741, 1.000] |
| out-of-scope recall, near_domain | 14/14 | 1.000 | [0.785, 1.000] |
| out-of-scope recall, pooled | 25/25 | 1.000 | [0.867, 1.000] |
| warning_carried | 4/6 | 0.667 | [0.300, 0.903] |
| citations_valid | 67/67 | 1.000 | [0.946, 1.000] |
| uncited | 1/67 | 0.015 | [0.003, 0.080] |
| gold_cited | 59/61 | 0.967 | [0.888, 0.991] |
| forbidden_hit | 1/14 | 0.071 | [0.013, 0.315] |
| adversarial_pass | 13/14 | 0.929 | [0.685, 0.987] |
| rewrite_clean | 1/3 | 0.333 | [0.061, 0.792] |

| slice | items | key-fact pass | refused |
|---|---|---|---|
| answers_v1 | 56 | 45/48 | 10/56 |
| multi_turn | 10 | 10/10 | 0/10 |
| long_query | 8 | 3/8 | 3/8 |
| out_of_scope | 17 | 0/0 | 17/17 |
| adversarial | 14 | 6/6 | 8/14 |

| server latency | ms |
|---|---|
| n | 85 |
| ttft_p50 | 3600.107 |
| ttft_p95 | 10170.387 |
| total_p50 | 4553.156 |
| total_p95 | 11337.812 |
| layer1_n | 20 |
| layer1_total_p50 | 1978.126 |
| layer1_total_p95 | 4011.088 |
