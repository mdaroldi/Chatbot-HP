# ADR 0013: Load testing: unit, SLO, harness, tuning and the capacity result

- **Status:** accepted (Phase 6)
- **Date:** 2026-11-17

## Context

The challenge asks for scalability shown with load tests, as requests per minute
supported. A request count alone says little for a chat service: one question is a
message stream plus the conversation calls the UI makes around it, and the stream's
cost depends on retrieval, reranking and the model. The constraints were one laptop for
both the system and the load generator, no paid capacity runs on the real model, and
answers that must not change for the sake of speed.

The reports with every table are in [`loadtest/reports/`](../../loadtest/reports/):
[`baseline.md`](../../loadtest/reports/baseline.md),
[`bottleneck.md`](../../loadtest/reports/bottleneck.md),
[`tuning.md`](../../loadtest/reports/tuning.md),
[`rate-limits.md`](../../loadtest/reports/rate-limits.md) and
[`profile-b.md`](../../loadtest/reports/profile-b.md). The SLO is
[`loadtest/SLO.md`](../../loadtest/SLO.md). This ADR quotes their numbers; it does not
recompute them.

## Result

**16 users, 32.9 completed questions per minute (87.9 requests per minute), SLO met**,
on one backend replica with the shipped default of 1 inference worker × 4 ONNX threads
(Profile A, stub LLM). Run `w7c-confirm-1x4-16`, a 10-minute step with 9 minutes
measured: 0 errors, 0 HTTP 429, first-question TTFT p95 2.38 s (bound 4.14 s), follow-up
TTFT p95 4.93 s (bound 9.12 s), backend memory peak 1,432 MiB of 2,048.

- **Unit:** completed questions per minute at the SLO, on one replica. A question is
  completed when its stream reaches `done`. Requests per minute counts every HTTP
  request the simulated users send (messages, plus the calls the UI makes around them:
  create, list, open and delete conversations, and the readiness check), and is given
  beside it because the challenge asks for it.
- **Hardware:** Apple M2 (4 performance + 4 efficiency cores), 8 GB memory, macOS
  26.6.2, Docker Desktop 29.1.2 (VM: 8 CPUs, 3.8 GiB). Backend limited to 4 CPUs and
  2 GiB, Postgres and Qdrant to 1 CPU each. Locust ran on the same host.
- **History:** before tuning, on the previous default of 2 × 2, the 10-minute headline
  was 8 users and 15.8 questions per minute (`baseline.md`). The new headline is 2× the
  users and 2.1× the questions per minute, with no change to answers.

## Decision

### Load stack and harness

- **Own compose project.** `make load-up` starts `hpdocs-load` from
  `docker-compose.load.yml`, which changes only environment and resource limits (a unit
  test pins this). Only the backend and its dependencies run; the API is on port 8100.
  [`loadtest/load.env`](../../loadtest/load.env) raises every rate limit to 10,000 per
  minute for capacity runs, sets `LLM_MAX_RETRIES=0` and keeps the stub provider.
- **Locust on the host**, as its own uv project in `loadtest/`, with tested helpers in
  `hpdocs_load` (workload, SSE timing, samplers, report, burst client, spend guard).
  Coverage gate 90% with branches; the only file omitted is `locustfile.py`, which is
  glue with no branching of its own, and a test pins the omit list to that one path.
- **Workload:** dev-set questions, 1-3 follow-ups per conversation (0.5 / 0.3 / 0.2),
  10-30 s of think time before each question, and the UI's conversation actions around
  them: select (30%), reload (10%), delete (10%).
- **Stub delay:** 160 ms per streamed piece, 19 pieces, about 3.0 s: the Phase 5 p50 of
  the real model's time to first token plus streaming. **Profile A TTFT excludes
  provider latency.** The stub streams the follow-up rewrite at the same pace, about
  1.6 s slower than the real rewrite model.
- **Same starting state.** `make load-reset` clears conversations (the index is kept),
  recreates the backend and waits for `/ready`. A failed reset voids the run.
- **Measured runs** go through `make load-measured`: the run is held under
  `caffeinate -i -s`, reset first, host swap, memory and paging are recorded at the
  start and end, and the run is void when `pmset` logged a sleep or wake inside its
  window.
- **What counts.** Each step's first 60 s are warm-up and excluded. A step counts when
  its measured window is at least 3 minutes, Locust's CPU stayed under 90%, no host
  sample gap exceeded 15 s and page-outs stayed at or under 3,000 per minute. A run
  counts only when swap grew by at most 500 MiB.

### SLO

Fixed once in [`loadtest/SLO.md`](../../loadtest/SLO.md) (commit 64148ad) from the
single-user baseline, before the ramp ran, and never edited. Two classes, because a
follow-up waits for its rewrite before retrieval starts, and with one bound the share of
follow-ups at a step would move the capacity number:

| Class | TTFT p95 bound | Done p95 bound |
|---|---|---|
| First questions | 4.14 s | 10.10 s |
| Follow-ups | 9.12 s | 15.16 s |

Each bound is 2× that class's single-user p95. Errors must stay under 1% of all
questions; 429s are counted apart. A step is judged on its whole measured window.

### Bottleneck and tuning

- **Bottleneck:** reranking on the inference pool (`bottleneck.md`). Postgres and Qdrant
  stayed at about 1% CPU, Locust at 2% at most; with the reranker off (a diagnostic run,
  never a capacity number) 24 users completed 53 questions per minute at 8% backend CPU.
  On 2 × 2, queueing for the pool lifted first-question TTFT past its bound while
  throughput still grew, and two concurrent reranks finished only about 12% more per
  minute than one at a time while each cost about 40% more CPU. The plateau of about 39
  questions per minute is not a capacity at the SLO and is never quoted as one.
- **Default changed to 1 × 4** (`INFERENCE_WORKERS=1`, `INFERENCE_ONNX_THREADS=4`,
  `tuning.md`). One rerank on 4 threads takes 1,024 ms p50 in isolation against 1,636 ms
  on 2 threads. On 5-minute steps the highest passing step rose from 10 users
  (20.5 questions per minute) to 16 users (32.8); at the same load first-question TTFT
  p95 fell by 35-42%. A drift control between measurement windows moved at most 3.1%.
- **Parity:** the 2 × 2 and 1 × 4 pipelines scored 56 dev questions × 20 candidates =
  1,120 pairs with max |d| = 0: bit-identical scores, 0 refusal-gate flips, 0 top-5
  order changes. The thread split changes speed only.
- **4 × 1 is out:** at the 2 GiB limit the backend was OOM-killed (6 memory-cgroup kills
  of uvicorn in one container over 446 s), and the restarts are the errors in its runs.
  Each concurrent rerank session costs about 480 MiB. *Extrapolation, not measured:*
  4 × 1 would need about 955 + 4 × 480 ≈ 2,880 MiB (about 2.9 GiB). Measured are only
  the OOM kills at 2 GiB and the peaks of the other arms: 1 × 4 at 1,424-1,442 MiB,
  2 × 2 at 1,915 MiB, which left the previous default 133 MiB of headroom at 12 users.
- **Not adopted:** a separate query-embedding pool (`INFERENCE_EMBED_WORKERS`) passed
  the same 16 users and was worse above it; it stays off. No cache was measured, and
  quality-affecting changes (fewer rerank candidates, reranker off) were not applied.

### Rate limits on (outside the SLO)

One 5-minute step at the compose defaults (messages 10 per minute with a burst of 5),
with the 16 normal users plus a burst client: 3 Locust users sharing one client ID,
sending without think time or backing off (`rate-limits.md`).

- The burst client was held to its quota: 40 answers in the 4-minute window (10.0 per
  minute) and 110,537 `429 rate_limited` responses. A probe after the run showed the
  429 carries `Retry-After` and the unified error envelope.
- The normal users had **0 429s and 0 5xx**.
- Their latency still rose past the SLO bounds: 30.5 questions per minute, first TTFT
  p95 6.00 s and follow-up 9.30 s, against 2.96 s and 5.17 s at 16 users without the
  burst client. Two hypotheses, neither tested: the burst client's 10 answered
  questions per minute brought the load near the 20-user step, which already failed;
  and about 460 refused requests per second on the single event loop cost something on
  top.
- The in-process limiter protects the model and inference capacity, not the event loop
  from a flood. The next step is a limit in front of the API (reverse proxy or gateway,
  per IP and per client ID).

### Profile B: real LLM, bounded by provider limits

A separate capped run end to end on the real Anthropic models (`profile-b.md`). Its
throughput and latency depend on the provider's latency and on the account's rate
limits, which this deployment does not control. It is not a capacity table and is never
compared with Profile A, whose SLO does not apply to it.

| Users | Min | Q/min | Req/min | Err % | 429 | First TTFT p50 / p95 s | Follow-up TTFT p50 / p95 s |
|---|---|---|---|---|---|---|---|
| 4 | 6.5 | 9.2 | 23.7 | 0.00 | 0 | 1.71 / 2.23 | 2.65 / 3.51 |

*Bounded by provider limits.* 67 questions, all ending in `done`, 0 provider 429s,
5xx or timeouts. 9.2 questions per minute is the pace of 4 users with think time, not a
ceiling.

- **One attempt per call:** `LLM_MAX_RETRIES=0`, so a provider error would show as an
  error rather than a hidden retry.
- **Spend guard:** `make load` turns it on whenever `LOAD_PROVIDER=anthropic`. It
  charges every question its worst case before sending it (one answer call, plus one
  rewrite call for a follow-up, at the Phase 5 token bounds and prices), with a $3.00
  cap and a hard stop at 90 questions. It stopped the run at 67 questions with $2.99906
  charged; measured spend was at most $0.53.

## Limitations

- **Shared 8 GB laptop.** Locust, Docker Desktop and the host's own services share one
  machine. Generator CPU was recorded for every step and never limited one (at most 2%
  in capacity runs), but they compete for memory and the scheduler.
- **Residual swap.** Runs started with residual swap on the 8 GB host, restated per
  run in the reports: 1,065-1,570 MiB in the baseline runs, 992-2,311 MiB in the tuning
  runs, 1,084 MiB for the headline run, 1,856 MiB for the rate-limits run and 2,075 MiB
  for Profile B. Each measurement window started only after a memory check (at most
  2.5 GB of swap, average growth under 100 MB per 5 minutes).
- **Idle sleep.** The host idle-sleeps after 1 minute on AC power. A swap check on
  2026-11-15 slept from 23:17 to 23:35 and swap grew from 1,560 to 2,126 MB across it.
  That is why every measured run is held under `caffeinate` and voided on a sleep or
  wake. No power setting was changed.
- **Overnight swap.** After the second tuning window the Mac sat overnight without
  `caffeinate`, and swap grew from 1,615 to 2,470 MB, next to the 2.5 GB limit. The
  headline run was taken after a reboot.
- **TeamViewer** is a login item. It was open after one reboot and was quit before the
  window; later windows checked that it was closed.
- **Stub provider.** Profile A has no provider latency or provider limits, and its
  follow-up rewrite is about 1.6 s slower than the real model's.
- **Coverage exclusion:** `loadtest/locustfile.py` is outside the coverage gate (see
  above).
- **The API key is an environment variable of the backend container.** Compose passes
  `ANTHROPIC_API_KEY` from `.env`, so anyone with access to the Docker daemon can read it
  with `docker inspect` or `docker exec … printenv`. For this reason the Profile B run
  printed only `LLM_PROVIDER` and `LLM_MAX_RETRIES`. A deployment would mount it as a
  secret file or read it from a secrets manager.

## Consequences

- The capacity claim is one number with its unit, hardware and SLO, measured on one
  replica. It is repeatable with `make load-measured` and the commands in the reports.
- The default inference split is 1 × 4, with more memory headroom than before. Hosts
  with more CPUs can revisit the split and the embedding pool through environment
  variables alone.
- **Next steps for replicas** (out of scope here): more backend replicas need a reverse
  proxy in front that balances requests and keeps SSE streams open; migrations stay a
  single one-shot job and never run per replica (ADR 0007); and the rate limiter needs
  a shared store, since each replica now keeps its own buckets and N replicas allow N
  times the rate (ADR 0010). The same proxy is where the flood limit above belongs.
