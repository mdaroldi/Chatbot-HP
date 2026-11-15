# Evaluation

Golden dataset, benchmark runner and reports. The retrieval benchmark is Phase 2
([ADR 0009](../docs/adr/0009-retrieval-strategy-and-evaluation.md)); the answer benchmark
is Phase 5 ([ADR 0012](../docs/adr/0012-answer-benchmark.md)).

| Path | What it is |
|---|---|
| `golden/retrieval-v1.toml` | The retrieval golden set: questions, splits and anchors (short facts, never HP text) |
| `golden/answers-v1.toml` | Phase 5: key facts (and expected safety notes) for each retrieval-v1 question; frozen |
| `golden/slices-v1.toml` | Phase 5: multi-turn, long-query, out-of-scope and adversarial items; frozen |
| `retrieval-matrix.toml` | Named retrieval configurations, the bootstrap seed and resamples, and the selection rule's p95 tie margin |
| `reports/runs/<utc>-<sha>/` | Raw output of one run (git-ignored): `results.jsonl`, `summary.json`, `summary.md` |
| `reports/retrieval-summary.md` | The committed summary of the Phase 2 benchmark |
| `answer-matrix.toml` | Phase 5: the budget cap, dated prices, token bounds, models, bootstrap seed and the named answer runs |
| `answer-runs/` | Phase 5 raw output (git-ignored): one directory per run, the ledger and the judge cache |
| `reports/answers-summary.md` | The committed summary of the Phase 5 benchmark |

## Running the retrieval benchmark

The stack must be up (`make up`), so that Qdrant holds the index and Postgres its
manifest. The benchmark runs in the backend image, in process, with the models baked
into the image, against the index the alias names:

```
make eval-retrieval EVAL_ARGS=preflight                      # resolve every anchor; ids and counts only
make eval-retrieval                                          # every config in the matrix, dev split
make eval-retrieval EVAL_ARGS="run --config hybrid --config hybrid-rerank"
make eval-retrieval EVAL_ARGS="run --collection <kept build>" # a kept earlier build instead of the alias
```

- **Preflight.** Every run starts with it. Each anchor of both splits must match between
  one and five indexed chunks; otherwise the run stops and logs the question id, the
  anchor index and the count.
- **Held-out split.** It is scored only in Phase 5. `--split heldout` is refused unless
  `--allow-heldout` is also given.
- **Reproducibility.** `summary.json` records the golden set's version and SHA-256, the
  index build, the git SHA (with `-dirty` for uncommitted changes to tracked files), the
  model revisions, the hardware and every resolved setting. Reports hold identifiers,
  scores and timings only.
- **File ownership.** The target runs the container as your user (`--user $(id -u):$(id -g)`).
  The image's own user (uid 10001) could not write to a Linux bind mount owned by
  your uid, so without this the run would fail when it writes `reports/runs/`. On macOS,
  Docker Desktop maps ownership either way.

Exit codes: 0 done, 1 the command ran and failed, 2 refused before any work.

## Rules for Phase 5 (final evaluations)

- **One index build.** Candidate lists are deterministic per index build, not across
  builds (ADR 0009): Qdrant orders tied BM25 scores inside its own prefetch. Every
  final evaluation runs on a single build, and each run records its build id
  (`index.build_id` in `summary.json` and the summary header). Results from different
  builds are not compared as equals.
- **Held-out split.** It was never scored in Phase 2 and is scored only for the final
  evaluation, once, after the configuration is fixed.
- **Masked ablations.** The BGE prefix, RRF k, avg_len and the product filter were
  measured behind the reranker. If reranking is turned off, re-measure them.

## Running the answer benchmark

The stack must be up (`make up`) and `.env` must hold `ANTHROPIC_API_KEY` for the billed
targets. A host-side driver restarts the backend with each run's settings, checks them,
asks every item over the chat API and restores the defaults afterwards; the harness
container never holds the key.

```
make eval-estimate                                          # budget plan and spend, no key, no stack
make eval EVAL_PROVIDER=stub                                # free pipeline check, no quality claims
make eval                                                   # billed: reference-1 on dev, report, summary
make eval-answers ANSWER_ARGS="--run dense"                 # billed: one matrix row (--resume, --retry-errors)
make eval-ablation                                          # billed: every unreserved row in priority order
make eval-judge JUDGE_ARGS="--run reference-1"              # billed: judge pass via the Batch API
make eval-report REPORT_ARGS="--run dense"                  # one run's report.md / report.json (files only)
make eval-summary                                           # reports/answers-summary.md (files only)
make eval-label-sheet LABEL_RUN=reference-1                 # blind label sheet for the human labels
make eval-agreement LABEL_RUN=reference-1                   # judge vs human: raw agreement and kappa
```

- **Budget.** The ledger `answer-runs/ledger.jsonl` is append-only and a hard stop: an
  item starts only if the spend plus its worst case stays under `cap_usd`, and a run
  starts only if what is left covers its estimate plus 25%. Rewrite calls are charged at
  their token bound (their usage is not stored), so `make eval-estimate` shows measured
  and estimated spend apart.
- **Held-out items.** `--allow-heldout` is required by `eval-answers`, `eval-judge` and
  `eval-report`, and only the `heldout` row accepts it. The held-out run was made once,
  on the final default, after every decision.
- **Raw output.** Answers, excerpts and label sheets stay under `answer-runs/`. The
  committed summary holds ids, counts and intervals only.
- **Regenerate the summary** after a run with `make eval-report REPORT_ARGS="--run NAME"`
  and `make eval-summary`, and commit it alone so its git SHA is clean.

Exit codes as for the retrieval benchmark; `make eval` without a key exits 2.

## Phase 5 results

Rule 7 kept every default; the held-out run is reported in the summary. The decision
record, the known limitations and the next steps are in
[ADR 0012](../docs/adr/0012-answer-benchmark.md).
