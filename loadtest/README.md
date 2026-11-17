# Load testing

Locust load tests for the chat API, the capacity reports, and the SLO they are judged
against. Decisions and findings: [ADR 0013](../docs/adr/0013-load-testing.md).

**Headline (Profile A, stub LLM):** 16 users, **32.9 completed questions per minute**
(87.9 requests per minute), SLO met, on one backend replica (4 CPUs, 2 GiB) with the
default inference split of 1 worker × 4 ONNX threads. Measured on an Apple M2 laptop
with 8 GB memory, with Locust on the same machine.

## Profiles

- **Profile A (stub LLM):** system capacity. The stub streams 19 pieces at 160 ms each
  (about 3.0 s), so time to first token excludes provider latency. All capacity numbers
  come from this profile.
- **Profile B (real LLM):** one short run end to end on the Anthropic models, under a
  spend guard. Its results are bounded by provider limits and are never compared with
  Profile A.

## How to run

All commands run from the repository root and need a `.env` (see the main README).

```sh
make load-up                      # start the load stack: project hpdocs-load, API on :8100
make load-measured LOAD_RUN=ramp-a LOAD_STEPS=2,4,8,16
make load-down                    # stop it (volumes are kept)
```

- `make load-up` starts only the backend and its dependencies, with
  `docker-compose.load.yml` (resource limits: backend 4 CPUs / 2 GiB, Postgres and
  Qdrant 1 CPU each) and [`load.env`](load.env) (rate limits raised to 10,000 per
  minute, stub provider, `LLM_MAX_RETRIES=0`).
- `make load-reset` gives every run the same starting state: it clears conversations
  (the index is kept), recreates the backend and waits for `/ready`.
- `make load LOAD_RUN=<name> ...` runs Locust headless and prints the run's table.
- `make load-measured` is the run that counts: it runs `load-reset`, records host swap,
  memory and paging at the start and end, runs `make load`, and voids the run when the
  host slept or woke inside its window. It re-runs itself under `caffeinate -i -s`
  ([`measured_run.sh`](measured_run.sh)), because the test laptop idle-sleeps after one
  minute. Exit codes: 0 counted, 1 reset failed, 2 usage error or log exists, 3 the
  host slept (void); its log is `results/<run>.runner.log`.

Variables (empty means the default):

| Variable | Default | Meaning |
|---|---|---|
| `LOAD_RUN` | required | Run name; records go to `results/<run>/` (git-ignored) |
| `LOAD_STEPS` | `1` | Users per step, comma separated, e.g. `2,4,8,16` |
| `LOAD_STEP_SECONDS` | `300` | Length of each step |
| `LOAD_WARMUP_SECONDS` | `60` | Start of each step left out of its measured window |
| `LOAD_SPAWN_RATE` | `1` | Users started per second |
| `LOAD_SEED` | `1` | Seed of the workload's random choices |
| `LOAD_SAMPLE_SECONDS` | `5` | Interval of the container and host samples |
| `LOAD_BURST` | `0` | `1` adds the burst client of the rate-limits run |
| `LOAD_BUDGET` | `0` | `1` turns on the spend guard |
| `LOAD_PROVIDER` | `stub` | `anthropic` runs Profile B and always turns the spend guard on |

**Profile B** sends questions and retrieved excerpts to the Anthropic API and is billed.
It needs `ANTHROPIC_API_KEY` in `.env`:

```sh
make load-up LOAD_PROVIDER=anthropic
make load-measured LOAD_PROVIDER=anthropic LOAD_RUN=profile-b LOAD_STEPS=4 \
  LOAD_STEP_SECONDS=600 LOAD_WARMUP_SECONDS=60
make load-reset                   # back to the stub
```

The guard charges each question its worst case before sending it and stops the run at
$3.00 or 90 questions, whichever comes first.

**Rate-limits run:** the compose default limits set in the shell, plus `LOAD_BURST=1`;
the exact commands are at the end of [`reports/rate-limits.md`](reports/rate-limits.md).

## Workload

Dev-set questions from the retrieval golden set, 1-3 follow-ups per conversation
(0.5 / 0.3 / 0.2), 10-30 s of think time before each question, and the UI's
conversation actions around them: select (30%), reload (10%), delete (10%).

## How to read a report

`make load` prints one row per step; the same table is written to
`results/<run>/report.md`. Rerun it with
`uv run --frozen python -m hpdocs_load.report results/<run>` from `loadtest/`.

- **Q/min** is completed questions per minute (streams that reached `done`), the
  capacity unit. **Req/min** counts every request the users sent.
- **First** and **Follow-up** are the two SLO classes: a follow-up is rewritten into a
  standalone query before retrieval, so it has its own bounds.
- **TTFT** is time to first token, **done** the time to the end of the stream; both
  are measured by the client. **Rerank** and **TTFT - retrieval** come from the
  backend's stage timings.
- **429** is counted apart from errors; capacity runs raise the limits so none occur.
- **Counts** says whether the step is evidence: `yes` when the measured window is at
  least 3 minutes, Locust stayed under 90% CPU, no host sample gap exceeded 15 s and
  page-outs stayed at or under 3,000 per minute. A run whose swap grew by more than
  500 MiB does not count at all.
- **SLO** is `pass` when both classes meet their p95 bounds for TTFT and done, and
  errors stay under 1%, over the whole measured window ([`SLO.md`](SLO.md)). The
  headline is the counting step with the most questions per minute that passes.

## Hardware and limitations

Apple M2 (4 performance + 4 efficiency cores), 8 GB memory, macOS 26.6.2, Docker
Desktop 29.1.2 (VM: 8 CPUs, 3.8 GiB). The machine is shared: Locust, Docker Desktop and
the host's services compete for memory and the scheduler, and runs started with
residual swap (restated per run in each report). Locust's CPU is recorded per step; it
never limited a capacity step. Numbers from another machine are not comparable.

## Reports

- [`SLO.md`](SLO.md): the SLO, fixed from the single-user baseline before the ramp.
- [`reports/baseline.md`](reports/baseline.md): capacity before tuning (8 users,
  15.8 questions per minute) and the saturation plateau.
- [`reports/bottleneck.md`](reports/bottleneck.md): reranking on the inference pool.
- [`reports/tuning.md`](reports/tuning.md): 1 × 4 replaces 2 × 2; the headline run.
- [`reports/rate-limits.md`](reports/rate-limits.md): compose limits on, with a burst
  client.
- [`reports/profile-b.md`](reports/profile-b.md): the capped run on the real LLM.
