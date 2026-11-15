# Answer benchmark summary

Generated from each run's report.json by `make eval-summary`; ids and numbers only.

## Reference runs and noise floor (answers-v1 items)

| metric | reference-1 | reference-2 | reference-3 | noise floor | n |
|---|---|---|---|---|---|
| primary (strict key-fact pass) | 45/48 | 45/48 | 45/48 | 0.021 | 48 |
| false refusals (in scope) | 2/48 | 2/48 | 2/48 | 0.021 | 48 |
| near-domain refusal recall | 4/4 | 4/4 | 4/4 | 0.250 | 4 |



## Ablation (rule 7 against reference-1)

| run | env | primary | difference vs reference-1 [95%] | gate subset ref -> alt | false refusals | near-domain recall | layer 1 refused | verdict | reasons |
|---|---|---|---|---|---|---|---|---|---|
| improvement | default | not run |  |  |  |  |  |  |  |
| hybrid-no-rerank | RETRIEVAL_RERANK_ENABLED=false | 44/48 | -0.021 [-0.083, +0.042] (n=48) | 45/46 -> 43/46 | 2/48 | 4/4 | 0 | keep | primary_interval_not_above_zero, primary_gap_within_noise |
| dense | RETRIEVAL_MODE=dense, RETRIEVAL_RERANK_ENABLED=false | 42/48 | -0.062 [-0.125, +0.000] (n=48) | 45/46 -> 42/46 | 1/48 | 4/4 | 0 | keep | primary_interval_not_above_zero, primary_gap_within_noise |
| naive | QDRANT_COLLECTION_ALIAS=hpdocs_chunks_naive | 41/48 | -0.083 [-0.167, -0.021] (n=48) | 44/45 -> 41/45 | 4/48 | 4/4 | 7 | keep | primary_interval_not_above_zero, primary_gap_within_noise, false_refusals_worse |
| k3 | RETRIEVAL_TOP_K=3 | 41/48 | -0.083 [-0.167, -0.021] (n=48) | 45/46 -> 41/46 | 5/48 | 4/4 | 6 | keep | primary_interval_not_above_zero, primary_gap_within_noise, false_refusals_worse |
| sparse | RETRIEVAL_MODE=sparse, RETRIEVAL_RERANK_ENABLED=false | 43/48 | -0.042 [-0.125, +0.042] (n=48) | 45/46 -> 42/46 | 2/48 | 4/4 | 0 | keep | primary_interval_not_above_zero, primary_gap_within_noise |
| k8 | RETRIEVAL_TOP_K=8 | 44/48 | -0.021 [-0.062, +0.000] (n=48) | 45/46 -> 44/46 | 2/48 | 4/4 | 6 | keep | primary_interval_not_above_zero, primary_gap_within_noise |

## Rewrite off (descriptive, no verdict)

Key-fact pass on its 10 multi-turn items: reference-1 10/10, rewrite off 7/10.

## Held-out (final default)

| metric | k/n | rate | Wilson 95% |
|---|---|---|---|
| primary (strict key-fact pass) | 11/12 | 0.917 | [0.646, 0.985] |
| false refusals (in scope) | 2/19 | 0.105 | [0.029, 0.314] |
| out-of-scope recall, unrelated | 4/4 | 1.000 | [0.510, 1.000] |
| out-of-scope recall, near_domain | 5/5 | 1.000 | [0.566, 1.000] |
| out-of-scope recall, pooled | 9/9 | 1.000 | [0.701, 1.000] |
| faithful, lenient (no unsupported claim) | 17/17 | 1.000 | [0.816, 1.000] |
| faithful, strict (every claim supported) | 16/17 | 0.941 | [0.730, 0.990] |

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
| human labels, faithful | 20 | 20 | 1.000 | n/a |
| human labels, correctness | 20 | 20 | 1.000 | 1.000 |

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
- git_sha: c9421f38a8385dd8b387f69f1cde9024893171da
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
