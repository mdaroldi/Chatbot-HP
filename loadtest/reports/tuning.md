# Tuning, Profile A (stub LLM)

Follows [`bottleneck.md`](bottleneck.md), which named reranking on the inference pool as
the bottleneck. This report tests the non-quality candidates it listed: how the
backend's 4 CPUs are split between inference workers and ONNX threads per call, and a
separate pool for query embeddings. Every arm ran on one prebuilt backend image
(`sha256:ae823a25906c…`), selected by environment only; nothing was built during the
measurements. The SLO ([`../SLO.md`](../SLO.md)) is unchanged.

**Result: 1 worker × 4 ONNX threads replaces 2 × 2 as the default.** Scores are
bit-identical, and the highest passing step on 5-minute steps rises from 10 to 16 users
(20.5 to 32.8 questions per minute). The 10-minute confirmation at 16 users passes with
32.9 Q/min: the new headline (see the last section). The separate embedding pool does
not win. 4 × 1 is unusable at the 2 GiB memory limit.

## Arms

`INFERENCE_WORKERS` × `INFERENCE_ONNX_THREADS` (`INFERENCE_EMBED_WORKERS`):

| Arm | Workers | ONNX threads per call | Embedding pool | Threads in total |
|---|---|---|---|---|
| 2x2 (default before) | 2 | 2 | 0 (shared) | 4 |
| 1x4 | 1 | 4 | 0 (shared) | 4 |
| 4x1 | 4 | 1 | 0 (shared) | 4 |
| 1x4-e | 1 | 4 | 1 (own pool) | 8: the embedding model also runs 4 ONNX threads, which trips the backend's CPU warning |

## How it was measured

- Workload, report and SLO as in [`baseline.md`](baseline.md). Steps are 5 minutes with
  a 60 s warm-up, so each step has **4 minutes measured**. The 10-minute confirmation
  has 9 minutes measured. Only steps of the same length are compared below.
- Two windows on the same image: 2026-11-16 16:38–17:59 (probes, 2x2, 1x4 at 8–12,
  4x1) and 2026-11-16 23:35 – 2026-11-17 00:56 (1x4 at 14–24, 1x4-e, control,
  1x4-e confirmation), with a reboot in between, and a third, 2026-11-17 08:07–08:43
  after another reboot, for the 10-minute confirmation of the shipped default (1x4
  at 16 users). Each window started only after a 20-minute
  memory check: every sample at most 2.5 GB of swap, average growth under 100 MB per
  5 minutes. The second window's first check failed on growth because its first sample
  came before the stack's memory settled (+670 MB in the first interval, then −96, −16,
  −8 MB). It was repeated with the stack already up and passed (+48 MB per 5 minutes).
  The rule was not changed.
- Every swap check and run ran under `caffeinate -i -s`. No sleep or wake events
  occurred inside either window (`pmset -g log`).
- A run counts only if swap grew by at most 500 MiB, no step was stalled (sampler gap
  over 15 s) and the load generator stayed under 90% CPU. Locust peaked at 1%.
- Residual swap at the start of each run, on an 8 GB host: before-2x2 992 MiB, 1x4
  1,231, 4x1 1,280, 4x1-r2 2,311, 1x4-hi 1,719, 1x4-e 1,679, control 1,663,
  1x4-e confirmation 1,631, 1x4 confirmation 1,084.

## Isolated rerank probes

`/v1/search` on the 56 dev first questions, one call per question after a warm-up call,
no other load:

| Arm | p50 ms | p95 ms | max ms |
|---|---|---|---|
| 2x2 | 1,636 | 1,689 | 1,822 |
| 1x4 | 1,024 | 1,099 | 1,153 |
| 4x1 | 3,097 | 3,161 | 3,319 |

One rerank of 20 candidates uses 4 threads better than 2: 1x4 is 37% faster per call.
Under load, 2x2 runs two such calls at once, each at the slower speed.

## Before and after (5-minute steps, 4 minutes measured)

SLO bounds: first-question TTFT p95 ≤ 4.14 s, follow-up TTFT p95 ≤ 9.12 s (done bounds
10.1 s and 15.16 s; no step failed on them first). Every step below counts: 0 errors,
0 HTTP 429.

### 2x2, before (`w7-before-2x2`)

| Users | Q/min | First TTFT p50 / p95 s | Follow-up TTFT p50 / p95 s | Rerank p50 / p95 s | Backend CPU mean / max % | SLO |
|---|---|---|---|---|---|---|
| 8 | 15.2 | 1.94 / 2.62 | 4.60 / 5.28 | 1.95 / 2.54 | 109 / 319 | pass |
| 10 | 20.5 | 1.96 / 3.10 | 4.72 / 5.51 | 1.85 / 2.61 | 127 / 304 | pass |
| 12 | 24.2 | 2.25 / 4.46 | 4.87 / 6.39 | 2.07 / 2.67 | 138 / 306 | fail |

Highest passing step: **10 users, 20.5 Q/min** (5-minute step). The earlier headline of
8 users ([`baseline.md`](baseline.md)) came from a coarser grid (8, then 12). 10 users
was not measured then.

### 1x4, after (`w7-1x4`, 8–12; `w7b-1x4-hi`, 14–24)

| Users | Q/min | First TTFT p50 / p95 s | Follow-up TTFT p50 / p95 s | Rerank p50 / p95 s | Backend CPU mean / max % | SLO |
|---|---|---|---|---|---|---|
| 8 | 16.0 | 1.15 / 1.65 | 3.63 / 4.35 | 0.94 / 1.04 | 129 / 400 | pass |
| 10 | 21.5 | 1.15 / 2.00 | 3.65 / 4.65 | 0.95 / 1.05 | 141 / 398 | pass |
| 12 | 25.0 | 1.16 / 2.59 | 3.69 / 4.78 | 0.96 / 1.94 | 161 / 405 | pass |
| 14 | 28.8 | 1.16 / 2.27 | 3.67 / 4.62 | 0.94 / 1.80 | 164 / 407 | pass |
| 16 | 32.8 | 1.23 / 2.96 | 3.67 / 5.17 | 0.95 / 1.90 | 194 / 402 | pass |
| 20 | 42.8 | 1.78 / 4.29 | 4.10 / 6.75 | 1.01 / 3.03 | 275 / 402 | fail |
| 24 | 47.5 | 3.62 / 9.38 | 6.13 / 10.47 | 2.22 / 5.79 | 367 / 403 | fail |

Highest passing step: **16 users, 32.8 Q/min** (5-minute step). The extension (28, 32,
40 users) was not run: it was planned only if 1x4 still passed at 20.

### Same users, same step length

| Users | 2x2 Q/min | 1x4 Q/min | 2x2 first TTFT p95 s | 1x4 first TTFT p95 s | 2x2 follow-up TTFT p95 s | 1x4 follow-up TTFT p95 s |
|---|---|---|---|---|---|---|
| 8 | 15.2 | 16.0 | 2.62 | 1.65 (−37%) | 5.28 | 4.35 (−18%) |
| 10 | 20.5 | 21.5 | 3.10 | 2.00 (−35%) | 5.51 | 4.65 (−16%) |
| 12 | 24.2 (fail) | 25.0 (pass) | 4.46 | 2.59 (−42%) | 6.39 | 4.78 (−25%) |

Below saturation, Q/min follows the number of users (think time paces the workload),
so the gain shows in latency at the same load, and in the higher step that still passes.
Rerank p50 halves (about 1.9 s to 0.95 s): a question waits for one fast call, not two
slow ones sharing the CPUs.

### 4x1 (`w7-4x1`, void; `w7-4x1-r2`)

| Run | Users | Q/min | Err % | First TTFT p50 / p95 s | Rerank p50 / p95 s | SLO |
|---|---|---|---|---|---|---|
| w7-4x1 (void: swap +857 MiB) | 8 | 10.8 | 16.05 | 3.39 / 3.91 | 3.28 / 3.72 | fail |
| w7-4x1 (void) | 10 | 15.5 | 16.67 | 3.42 / 3.78 | 3.28 / 3.59 | fail |
| w7-4x1-r2 | 8 | 11.2 | 15.53 | 3.38 / 3.57 | 3.22 / 3.32 | fail |
| w7-4x1-r2 | 10 | 12.8 | 20.44 | 3.34 / 3.62 | 3.23 / 3.53 | fail |

Fails at every step on errors, not latency. See the next section.

### Embedding pool, 1x4-e (`w7b-1x4-e`): no win

| Users | Q/min | First TTFT p50 / p95 s | Follow-up TTFT p50 / p95 s | Rerank p50 / p95 s | Backend CPU mean / max % | SLO |
|---|---|---|---|---|---|---|
| 12 | 24.0 | 1.17 / 1.75 | 3.68 / 4.58 | 1.00 / 1.90 | 152 / 399 | pass |
| 14 | 29.8 | 1.23 / 2.45 | 3.68 / 5.13 | 1.02 / 2.42 | 215 / 403 | pass |
| 16 | 34.5 | 1.21 / 2.57 | 3.67 / 5.28 | 1.03 / 2.81 | 189 / 406 | pass |
| 20 | 41.0 | 2.43 / 5.07 | 4.78 / 8.53 | 2.19 / 5.43 | 316 / 408 | fail |
| 24 | 45.5 | 5.23 / 8.94 | 7.64 / 10.90 | 5.04 / 8.24 | 398 / 406 | fail |

Same highest passing step as 1x4 (16 users). At 16 users first-question TTFT p95 is
lower (2.57 s vs 2.96 s), but that is one 4-minute sample. Above the knee it is worse:
at 20 users first TTFT p95 is 5.07 s vs 4.29 s and rerank p50 2.19 s vs 1.01 s. The
embedding pool's 4 ONNX threads compete with the reranker's for the same 4 CPUs.
The 10-minute run on 1x4-e at 16 users (`w7b-confirm-1x4-e-16`: pass, 32.8 Q/min,
first TTFT p95 3.13 s, follow-up 5.08 s, 9 minutes measured) was taken before this
ruling. It is supporting evidence for 16 users, not the headline. Ruling: the pool stays
off (`INFERENCE_EMBED_WORKERS=0`). The setting remains for hosts with more CPUs.

## Memory: 4x1 is out, 2x2 had little headroom

Backend container, 2,048 MiB limit, from the 5-second samples:

| Arm | Idle MiB | Peak MiB | Headroom MiB | Concurrent rerank sessions |
|---|---|---|---|---|
| 1x4 (all runs up to 24 users) | about 955 | 1,424–1,442 (1,430 in `w7-1x4`) | about 610 | 1 |
| 2x2 (`w7-before-2x2`, at 12 users) | about 955 | 1,915 | 133 | 2 |
| 4x1 | about 955 | killed at the limit | none | 4 |

- **4x1 is OOM-killed.** The Docker VM kernel log shows memory-cgroup OOM kills of
  uvicorn at the 2 GiB limit (anon-rss about 2,083,000 kB): 6 kills in one container
  over 446 s, matching the 6 memory drops in `w7-4x1-r2`'s samples over 448 s
  (`w7-4x1` shows 7 drops in the same pattern). `restart: unless-stopped` brings it
  back. The errors in the table (connection reset or refused, 503 while restarting,
  409 on a conversation whose turn was lost) are those restarts.
- **Each concurrent rerank session costs about 480 MiB** (20 candidates of about
  1,139 characters): 1x4 peaks about 475 MiB above idle with one session, 2x2 about
  960 MiB with two.
- **Extrapolation, not measured:** 4x1 would need about 955 + 4 × 480 ≈ 2,880 MiB
  (≈ 2.9 GiB), over the limit. 4x1 never ran past the limit, so only the OOM kills at
  2 GiB and the 1x4 and 2x2 peaks are measurements.
- 1x4 also removes the previous default's thin margin: 2x2 came within 133 MiB of the
  limit at 12 users.

## Control: drift between the windows

`w7b-control-2x2` (second window) repeated 2x2 at 8 users, against `w7-before-2x2`
(first window):

| Metric | Before | Control | Difference |
|---|---|---|---|
| Q/min | 15.20 | 15.20 | +0.0% |
| First TTFT p95 s | 2.62 | 2.70 | +3.1% |
| Follow-up TTFT p95 s | 5.28 | 5.33 | +0.9% |
| Rerank p95 s | 2.54 | 2.56 | +0.8% |
| SLO | pass | pass | |

Drift is at most 3.1%, far below the 1x4 gains (16–42% on the same p95s), so comparing
runs across the two windows is sound.

## Parity: the change does not affect answers

Checked before the default was changed. The 2x2 and 1x4 pipelines were served on the
load stack (`SEARCH_ENDPOINT_ENABLED=true`, container environment printed before each
collection) on 56 dev questions × 20 candidates = **1,120 pairs**:

- same candidates 56/56, **max |d| = 0** (scores bit-identical);
- **0 refusal-gate flips** (threshold −6.54 on the top score), 0 top-5 order changes;
  6/56 refused under both; closest top score 0.0009 from the gate;
- offline, the reranker at 1, 2 and 4 ONNX threads and the dense query vectors at 1, 2
  and 4 threads are also bit-identical (max |d| = 0).

The thread split changes speed only; retrieval results, citations and refusals are the
same.

## Not tried

- **Cache:** no cache was measured. Dense embedding is small next to reranking
  (`bottleneck.md`), and a cache result would be reported separately with its hit rate,
  never as the headline.
- **Quality-affecting changes** (fewer rerank candidates, reranker off): proposals only,
  not applied. Each would move refusal-gate layer 1 and needs the Phase 5 measurement.

## Headline

**16 users, 32.9 completed questions per minute (87.9 requests per minute), SLO met,**
on the shipped default (1x4), confirmed over a 10-minute step (9 minutes measured):

| Run | Users | Q/min | Req/min | Err % | First TTFT p50 / p95 s | First done p50 / p95 s | Follow-up TTFT p50 / p95 s | Follow-up done p50 / p95 s | Rerank p50 / p95 s | Backend CPU mean / max % | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `w7c-confirm-1x4-16` | 16 | 32.9 | 87.9 | 0.00 | 1.36 / 2.38 | 4.22 / 5.31 | 3.67 / 4.93 | 6.64 / 7.91 | 0.96 / 1.92 | 222 / 404 | pass |

Window 2026-11-17 08:07–08:43, after a reboot: same image, 5 minutes of settling after
`up`, then the memory check (swap 1,116, 1,108, 1,108, 1,108, 1,108 MB, −2 MB per
5 minutes). Swap 1,084 → 1,076 MiB during the run (residual swap at start 1,084 MiB on
an 8 GB host), no sleep or wake events, largest sampler gap 7 s, backend memory peak
1,432 MiB of 2,048. 0 HTTP 429.

Against the previous headline (8 users, 15.8 Q/min, 10-minute confirmation in
[`baseline.md`](baseline.md)) this is 2× the users and 2.1× the questions per minute,
with no change to answers. Above it, the SLO breaks at 20 users on first-question TTFT
p95 (4.29 s against the 4.14 s bound), with throughput still rising (42.8 Q/min).
That is the same pattern as before: queueing for the reranker lifts TTFT past the bound
before throughput saturates.
