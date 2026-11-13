# Evaluation

Golden dataset, benchmark runner and reports. The retrieval benchmark is Phase 2
([ADR 0009](../docs/adr/0009-retrieval-strategy-and-evaluation.md)); generation metrics
follow in Phase 5.

| Path | What it is |
|---|---|
| `golden/retrieval-v1.toml` | The retrieval golden set: questions, splits and anchors (short facts, never HP text) |
| `retrieval-matrix.toml` | Named retrieval configurations, the bootstrap seed and resamples |
| `reports/runs/<utc>-<sha>/` | Raw output of one run (git-ignored): `results.jsonl`, `summary.json`, `summary.md` |
| `reports/retrieval-summary.md` | The committed summary of the Phase 2 benchmark |

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
