# Baseline capacity, Profile A (stub LLM)

Runs of 2026-11-16 (CET), before any tuning. Judged against [`loadtest/SLO.md`](../SLO.md)
(commit 64148ad), which was written from `baseline-1u` before the ramp ran. Every table
below is printed by `python -m hpdocs_load.report results/<run>` (the last step of
`make load`); raw run directories are git-ignored.

## Result: two numbers

| | Users | Completed questions / min | Requests / min | SLO | Bound by |
|---|---|---|---|---|---|
| **Headline** (highest passing step, confirmed 10 min) | 8 | **15.8** | 42.7 | pass | the SLO's TTFT bound |
| Saturation plateau (ramp, 24 and 32 users) | 24-32 | ~39 | ~103 | fail | backend CPU (~305% of 4 CPUs) |

The headline is the capacity at the SLO on one backend replica. The plateau is the most
the system completes when latency is allowed to grow without bound; it is not a capacity
at the SLO and is never quoted as one.

What separates them: queueing in front of the reranker raises time to first token past
2x the single-user value well before throughput stops growing. From 2 to 20 users each
user still completes about 1.9-2.1 questions per minute, as at 1 user (2.1), so the
system keeps up; but rerank p95 grows from 2.1 s (4 users) to 3.9 s (12 users), and at
12 users first-question TTFT p95 is 4.45 s against a bound of 4.14 s. Throughput levels
off only at 24 users, where backend CPU reaches about 305% and rerank p50 grows from
1.7 s to 6-12 s. The bottleneck analysis follows in `bottleneck.md`.

The grid between 8 (pass) and 12 (fail) is coarse; the tuning round re-measures 8 and 10
users and then 9 or 11 to find the highest passing step on a finer grid.

## Setup

- **Host:** Apple M2 (4 performance + 4 efficiency cores), 8 GB memory, macOS 26.6.2,
  Docker Desktop 29.1.2 (VM: 8 CPUs, 3.8 GiB). Shared machine: Locust runs on the same
  host as the system under test (see Limitations).
- **Load stack:** `make load-up` (project `hpdocs-load`, `docker-compose.load.yml`):
  backend 4 CPUs / 2 GiB, Postgres and Qdrant 1 CPU each; rate limits raised to 10,000
  per minute; stub provider at 160 ms per streamed piece (19 pieces, about 3.0 s, the
  Phase 5 p50 of real TTFT plus streaming). **Profile A TTFT excludes provider latency.**
- **Workload:** dev-set questions, 1-3 follow-ups per conversation (0.5 / 0.3 / 0.2),
  10-30 s think time before each question; around the questions, conversation actions
  as in the UI: select (30%), reload (10%), delete (10%).
- **Same starting state:** every run starts from `make load-reset` (conversations
  cleared, index kept, fresh backend, `/ready` waited for).
- **Measured window:** each step's first 60 s are warm-up and excluded. A ramp step of
  5 minutes therefore measures **4 minutes**, the 10-minute confirmation **9 minutes**,
  and the 16-minute baseline **15 minutes**. A question counts in the step where its
  stream ends. The `Min` column of each table is that measured window.
- **Generator:** Locust process CPU peaked at 2% in every step, far from the 90% limit,
  so no step was generator bound.

## Runs

| Run | Window (CET) | Steps | Start swap | Counted | Notes |
|---|---|---|---|---|---|
| `baseline-1u` | 11:48-12:04 | 1 user, 16 min | 1,065 MiB | yes | source of the SLO |
| `ramp-1` | 12:11-12:51 | 2, 4, 8, 12, 16, 20, 24, 32 | 1,144 MiB | **no** | swap grew 802 MiB, 762 of it in the 4-user step (rule: 500 MiB) |
| `ramp-2` | 13:28-14:08 | 2, 4, 8, 12, 16, 20, 24, 32 | 1,348 MiB | yes | rerun of `ramp-1` after a Docker Desktop restart |
| `diag-norerank-24` | 14:09-14:14 | 24 users, reranker off | 1,570 MiB | diagnostic | never in a capacity table |
| `confirm-8u` | 14:14-14:24 | 8 users, 10 min | 1,562 MiB | yes | confirms the headline |

`ramp-1` is excluded under the swap rule, not for its numbers, which match `ramp-2`
closely: the same steps pass (2, 4, 8) and fail (12 and up), 8 users completed 15.8
questions per minute (15.5 in `ramp-2`), and the plateau was 39.5 / 39.0 at 24 / 32 users
(39.5 / 39.0). Docker Desktop was restarted between the two ramps.

Every run was taken under `caffeinate -i -s`, with no Sleep, Wake or DarkWake in pmset's
log inside its window, a largest sampler gap of 7 s (limit 15 s) and page-outs under
300 per minute (limit 3,000). These runs used an uncommitted wrapper script; `make
load-measured` (fbad981) now holds the same checks for every later run.

## Capacity table (`ramp-2`)

Compact view; the full printed tables are below.

| Users | Q/min | Req/min | First TTFT p95 s | First done p95 s | Follow-up TTFT p95 s | Follow-up done p95 s | Rerank p50 / p95 s | Backend CPU mean % | SLO |
|---|---|---|---|---|---|---|---|---|---|
| bound | | | 4.14 | 10.10 | 9.12 | 15.16 | | | |
| 2 | 3.8 | 10.2 | 2.84 | 5.85 | 5.31 | 8.31 | 1.71 / 2.63 | 31 | pass |
| 4 | 8.2 | 20.5 | 1.98 | 4.96 | 4.81 | 7.79 | 1.68 / 2.11 | 42 | pass |
| 8 | 15.5 | 43.2 | 3.16 | 6.16 | 7.13 | 10.11 | 1.96 / 2.72 | 81 | pass |
| 12 | 24.8 | 66.8 | **4.45** | 7.44 | 7.66 | 10.63 | 2.25 / 3.86 | 150 | fail |
| 16 | 31.0 | 78.5 | 5.49 | 8.48 | 8.80 | 11.78 | 2.58 / 4.64 | 215 | fail |
| 20 | 37.5 | 106.5 | 7.00 | 9.73 | 10.57 | 13.56 | 2.90 / 5.84 | 254 | fail |
| 24 | 39.5 | 103.8 | 18.46 | 21.43 | 21.09 | 24.06 | 5.99 / 12.24 | 306 | fail |
| 32 | 39.0 | 102.8 | 35.07 | 38.06 | 35.68 | 38.65 | 12.19 / 21.40 | 305 | fail |

No step had an error or a 429. Every step counts (4-minute window, generator not
saturated). The first failing bound at 12 users is first-question TTFT; from 16 users
follow-ups fail too.

### Confirmation (`confirm-8u`, 9 minutes measured)

15.8 completed questions per minute (42.7 requests per minute), 142 questions, no
errors. First questions: TTFT p95 3.47 s, done p95 6.47 s; follow-ups: TTFT p95 5.46 s,
done p95 8.46 s. Swap did not grow.

### Single-user baseline (`baseline-1u`, 15 minutes measured)

| Class | TTFT p50 / p95 s | Done p50 / p95 s |
|---|---|---|
| First questions | 1.94 / 2.07 | 4.95 / 5.05 |
| Follow-ups | 4.49 / 4.56 | 7.47 / 7.58 |

At one user, rerank dominates first-question TTFT: rerank p50 1.76 s of 1.94 s.

**Why two SLO classes.** The stub streams the follow-up rewrite at the same 160 ms per
piece, so a follow-up waits about 2.55 s (TTFT p50 4.49 s against 1.94 s) before
retrieval starts. The real rewrite model took p50 about 970 ms and p95 about 1,260 ms in
Phase 5 (10 multi-turn rewrites in the reference runs), so the stub overstates this wait
by about 1.6 s. An earlier note said the stub rewrite does not matter for capacity
because it is a sleep, not CPU work. That holds for CPU, but not for the capacity
number: with a single bound, the stub's long rewrite would have widened the bound and
the share of follow-ups at a step would have moved the result. Each class therefore has
its own baseline and bound.

**Rerank time against ADR 0008.** Single-user rerank is about 1.7-1.9 s, against 1.4 s in
ADR 0008's warm-up shape. An isolated probe of `/v1/search` (52 dev first questions,
second call kept) gave p50 1,664 ms and p95 1,717 ms. The real candidates average 1,139
characters against 930 in ADR 0008's shape, and 1,444 ms x 1,139 / 930 = 1,770 ms, which
matches. The backend's cgroup was throttled in 1 of 6,103 periods, and the narrow p95/p50
ratio rules out a split between performance and efficiency cores. The cause is input
length, not the CPU limit.

### Diagnostic: reranker off at 24 users (not a capacity result)

53.0 completed questions per minute, backend CPU mean 8%, first-question TTFT p50
0.20 s. Without the reranker the same 24 users are bound by their own think time, not by
the server. This run changes answer quality and is never a capacity number.

## Limitations

- **Shared machine.** Locust, Docker Desktop and the host's own services share one
  8 GB laptop. Generator CPU was recorded per step and never limited a step, but the
  containers and the generator compete for memory and the host scheduler.
- **Residual swap.** Runs started with about 1.1-1.6 GB of residual swap on an 8 GB host
  (start values in the Runs table). A run counts only if swap grew by less than 500 MiB.
  The start-of-run swap rule was relaxed twice on 2026-11-16, before the baseline:
  first "stable" was redefined as at most 2.5 GB with growth under 100 MB per 5-minute
  sample while idle; then that growth was taken as the average over the whole check
  (1,056 -> 1,153 MB in about 7 minutes, about 70 MB per 5 minutes; samples -13 and
  +110 MB), which passed.
- **Idle sleep.** The host idle-sleeps after 1 minute on AC power. A swap check on
  2026-11-15 slept from 23:17 to 23:35 and swap grew from 1,560 to 2,126 MB across it.
  Every run therefore runs under `caffeinate -i -s` and is void if pmset logs a sleep or
  wake inside its window. No power setting was changed.
- **Stub provider.** Profile A measures the system without provider latency or provider
  limits; the follow-up rewrite is about 1.6 s slower than the real model's.

## Full printed tables

Columns: Locust CPU max % is the generator; swap growth is within the step; Counts is
the window and generator rule.

<details>
<summary><code>ramp-2</code></summary>

| Users | Min | Q/min | Req/min | Err % | 429 | First TTFT p50 / p95 s | First done p50 / p95 s | Follow-up TTFT p50 / p95 s | Follow-up done p50 / p95 s | Rerank p50 / p95 s | TTFT - retrieval p50 / p95 s | Backend CPU mean / max % | PG CPU % | Qdrant CPU % | Locust CPU max % | Host mem max % | Swap max GiB | Swap growth MiB | Page-outs/min | Counts | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.0 | 3.8 | 10.2 | 0.00 | 0 | 1.88 / 2.84 | 4.86 / 5.85 | 4.39 / 5.31 | 7.40 / 8.31 | 1.71 / 2.63 | 2.50 / 2.70 | 31 / 201 | 0 | 0 | 0 | 86 | 1.7 | 342 | 155 | yes | pass |
| 4 | 4.0 | 8.2 | 20.5 | 0.00 | 0 | 1.90 / 1.98 | 4.86 / 4.96 | 4.36 / 4.81 | 7.32 / 7.79 | 1.68 / 2.11 | 2.51 / 2.66 | 42 / 200 | 0 | 0 | 1 | 80 | 1.7 | 0 | 122 | yes | pass |
| 8 | 4.0 | 15.5 | 43.2 | 0.00 | 0 | 2.17 / 3.16 | 5.13 / 6.16 | 4.70 / 7.13 | 7.65 / 10.11 | 1.96 / 2.72 | 2.50 / 2.67 | 81 / 310 | 0 | 0 | 1 | 80 | 1.7 | -24 | 143 | yes | pass |
| 12 | 4.0 | 24.8 | 66.8 | 0.00 | 0 | 2.48 / 4.45 | 5.35 / 7.44 | 5.02 / 7.66 | 8.02 / 10.63 | 2.25 / 3.86 | 2.48 / 2.67 | 150 / 304 | 0 | 0 | 1 | 78 | 1.6 | 0 | 122 | yes | fail |
| 16 | 4.0 | 31.0 | 78.5 | 0.00 | 0 | 3.12 / 5.49 | 6.06 / 8.48 | 5.54 / 8.80 | 8.51 / 11.78 | 2.58 / 4.64 | 2.48 / 2.67 | 215 / 334 | 1 | 1 | 1 | 80 | 1.6 | -16 | 293 | yes | fail |
| 20 | 4.0 | 37.5 | 106.5 | 0.00 | 0 | 3.96 / 7.00 | 6.68 / 9.73 | 6.34 / 10.57 | 9.32 / 13.56 | 2.90 / 5.84 | 2.47 / 2.66 | 254 / 326 | 1 | 0 | 1 | 78 | 1.6 | -8 | 148 | yes | fail |
| 24 | 4.0 | 39.5 | 103.8 | 0.00 | 0 | 9.77 / 18.46 | 11.44 / 21.43 | 12.84 / 21.09 | 15.81 / 24.06 | 5.99 / 12.24 | 2.48 / 2.65 | 306 / 328 | 1 | 0 | 1 | 79 | 1.6 | 0 | 128 | yes | fail |
| 32 | 4.0 | 39.0 | 102.8 | 0.00 | 0 | 20.67 / 35.07 | 22.95 / 38.06 | 25.30 / 35.68 | 28.27 / 38.65 | 12.19 / 21.40 | 2.48 / 2.66 | 305 / 324 | 1 | 0 | 1 | 79 | 1.5 | 0 | 155 | yes | fail |

Host: swap 1348 -> 1586 MiB (peak growth 358 MiB), memory used 74% -> 78%, page-ins
+279634, page-outs +6617, largest sampler gap 7 s.

</details>

<details>
<summary><code>confirm-8u</code> and <code>baseline-1u</code></summary>

| Run | Users | Min | Q/min | Req/min | Err % | 429 | First TTFT p50 / p95 s | First done p50 / p95 s | Follow-up TTFT p50 / p95 s | Follow-up done p50 / p95 s | Rerank p50 / p95 s | TTFT - retrieval p50 / p95 s | Backend CPU mean / max % | PG CPU % | Qdrant CPU % | Locust CPU max % | Host mem max % | Swap max GiB | Swap growth MiB | Page-outs/min | Counts | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `confirm-8u` | 8 | 9.0 | 15.8 | 42.7 | 0.00 | 0 | 2.06 / 3.47 | 4.99 / 6.47 | 4.43 / 5.46 | 7.43 / 8.46 | 1.86 / 2.52 | 2.50 / 2.68 | 112 / 315 | 0 | 0 | 1 | 79 | 1.5 | -24 | 138 | yes | pass |
| `baseline-1u` | 1 | 15.0 | 2.1 | 5.3 | 0.00 | 0 | 1.94 / 2.07 | 4.95 / 5.05 | 4.49 / 4.56 | 7.47 / 7.58 | 1.76 / 1.92 | 2.51 / 2.70 | 12 / 210 | 1 | 0 | 0 | 84 | 1.2 | 83 | 254 | yes | pass |

Host (`confirm-8u`): swap 1562 -> 1538 MiB, peak growth 0 MiB. Host (`baseline-1u`):
swap 1065 -> 1190 MiB, peak growth 167 MiB.

</details>
