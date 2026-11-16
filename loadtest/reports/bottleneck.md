# Bottleneck, Profile A (stub LLM)

Analysis of the baseline runs in [`baseline.md`](baseline.md) (`baseline-1u`, `ramp-2`,
`confirm-8u`, `diag-norerank-24`), before any tuning. No new measurement: every figure
comes from those runs' backend stage timings (`stages.jsonl`, one row per retrieval),
container CPU samples (every 5 s) and Locust records.

## Finding

**The bottleneck is reranking on the inference pool**: the cross-encoder that scores the
search candidates for every question, on a pool of 2 threads (`INFERENCE_WORKERS=2`)
with 2 ONNX threads per call (`INFERENCE_ONNX_THREADS=2`), inside the backend's 4-CPU
limit. Two things follow from it:

1. **The headline (8 users) is set by queueing in that pool**, which lifts time to first
   token past the SLO bound at 12 users while throughput still grows in line with users.
2. **The plateau (about 39 questions per minute) is set by the pool's throughput under
   concurrency**, which is about half of what the isolated rerank time predicts, while
   backend CPU stays at about 305% of 400%.

What it is not:

| Candidate | Evidence against it |
|---|---|
| Load generator | Locust process CPU at most 2% in every step (limit 90%) |
| Postgres, Qdrant | at most 1% CPU each at every step; search p50 4-5 ms up to 20 users, 14-26 ms at 24-32 (backend busy) |
| Event loop, HTTP, SSE, chat store | reranker off at 24 users: 53 questions per minute at 8% backend CPU, first-question TTFT p50 0.20 s; the rest of a question costs about 0.09 CPU-seconds |
| Stub LLM | a fixed sleep per piece; adds wall time to every question, no CPU |
| Embedding | dense query service time 15-35 ms at low load; BM25 under 1 ms |
| Host memory | swap growth at most 358 MiB in the counted ramp, page-outs under 300 per minute |

## Per-step evidence (`ramp-2`, `baseline-1u`)

| Users | Q/min | Dense embed p50 / p95 ms | Rerank p50 / p95 ms | Backend CPU mean % | CPU-s per question |
|---|---|---|---|---|---|
| 1 | 2.1 | 35 / 53 | 1,762 / 1,830 | 12 | 3.5 |
| 4 | 8.2 | 22 / 31 | 1,680 / 1,953 | 42 | 3.0 |
| 8 | 15.0 | 20 / 897 | 1,970 / 2,720 | 81 | 3.3 |
| 12 | 24.2 | 21 / 1,432 | 2,246 / 3,498 | 150 | 3.7 |
| 16 | 30.8 | 37 / 2,354 | 2,579 / 4,643 | 215 | 4.2 |
| 20 | 37.5 | 641 / 2,563 | 2,903 / 5,841 | 254 | 4.1 |
| 24 | 39.5 | 3,161 / 8,982 | 5,986 / 12,176 | 306 | 4.6 |
| 32 | 39.0 | 10,442 / 18,762 | 12,210 / 21,277 | 305 | 4.7 |

Q/min here counts retrievals that ended in the step's measured window, so it differs
slightly from the report's count of streams that reached `done`. CPU-s per question is
mean backend CPU x 60 s / questions per minute.

**How to read the timings.** Every stage timer wraps the call to the inference pool, so
it includes the wait for a free pool thread. Each question submits three tasks to the
same first-in, first-out pool, in order: dense query embedding, BM25 query embedding,
rerank. A new question's dense task waits behind the reranks already queued, so dense
embed time is a direct probe of the pool's queue: about 20 ms of service, the rest is
waiting. Its p95 is already 0.9 s at 8 users and 1.4 s at 12, which is how the queue
reaches first-question TTFT before throughput saturates. The BM25 task right after
waits only behind other questions' cheap dense tasks (p50 under 70 ms at every step).
The rerank then waits behind reranks again, so its time is queue plus service.

**Reranks per question.** Exactly one: no request ID has more than one retrieval in any
run (996 retrievals, 996 IDs in `ramp-2`). A follow-up runs the rewrite first and then
one retrieval; there is no second retrieval or rerank.

## Reconciliation: about 72 expected, about 39 measured

The isolated rerank takes about 1.66 s (p50 of an isolated `/v1/search` probe, 52
first questions; see `baseline.md`). Two pool threads each running one rerank at a time
predict 2 x 60 / 1.66 = **about 72 reranks per minute**, with 2 x 2 ONNX threads filling
the 4 CPUs. The plateau is **39.5 per minute** at 24 users and 39.0 at 32, at **about 305%**
backend CPU.

The prediction assumes that one rerank per question holds (it does, above) and that a
rerank costs the same with two running side by side. It does not:

| | Isolated (1-8 users) | Plateau (24-32 users) | Ratio |
|---|---|---|---|
| Wall time per rerank | 1.66-1.76 s | 3.04 s | x 1.8 |
| CPU-seconds per rerank | about 3.0-3.5 | about 4.6 | x 1.4 |
| Cores busy per rerank in progress | about 1.9 of 2 | about 1.5 of 2 | x 0.8 |

- **Wall time at the plateau** comes from throughput: with a standing queue (dense wait
  p50 3.2 s at 24 users) both pool threads are always busy, so each rerank holds a
  thread for 2 x 60 / 39.5 = 3.04 s. Rerank time minus dense wait gives a similar
  estimate (p50 2.7 s at 24 users).
- **CPU-seconds** at the plateau: 305% x 60 s / 39.5 = 4.6 CPU-seconds per question, of
  which about 0.09 is the rest of the question (reranker-off run). At 1-8 users the
  same sum gives 3.0-3.5.
- **Cores busy:** 4.6 CPU-s / 3.04 s = 1.5 cores per rerank, and two at a time give
  2 x 1.5 = 3.0 cores, which matches the 305% measured. At one user, 3.3 CPU-s over
  1.76 s is about 1.9 cores: the 2 ONNX threads are used almost fully when a rerank runs
  alone.

So two concurrent reranks complete only about 12% more per minute than one at a time
would (60 / 1.7 = 35 per minute), each costs about 40% more CPU, and the four ONNX
threads keep only about three of the four CPUs busy. The 4-CPU cgroup limit is not what
binds: CPU stays below 400% on average and, in 5-second samples, never passes 334%.

**Why concurrent reranks cost more is not settled by these runs.** The candidates that
fit the evidence:

- The Docker VM's 8 vCPUs run on 4 performance and 4 efficiency cores shared with
  Locust, Postgres, Qdrant and the host. One 2-thread rerank fits on performance cores;
  four busy ONNX threads plus everything else may not, and work moved to efficiency
  cores costs more CPU time for the same result. The narrow p95/p50 of the isolated
  probe rules this out for a rerank running alone, not for two at a time.
- Two ONNX sessions share caches and memory bandwidth; a cross-encoder over 20 pairs of
  up to 512 tokens is memory heavy.
- onnxruntime's intra-op threads spin while they wait at synchronization points, so a
  contended call burns more CPU-seconds without finishing sooner.
- cgroup throttling in short bursts: 1 throttled period in 6,103 in the isolated probe,
  not measured under load.

Telling them apart needs a measurement, which the tuning round makes anyway: changing
how the 4 CPUs are split between pool threads and ONNX threads changes the amount of
concurrency, and the before and after runs show which way capacity moves.

## Tuning candidates (no effect on answers)

Ordered by the evidence above. Each is a separate commit with a before and after run
from the same starting state; the SLO stays as committed.

1. **Workers x ONNX threads split.** Concurrency buys little (x 1.12) and costs CPU
   (x 1.4), and the SLO is broken by latency before throughput. One pool thread with 4
   ONNX threads (`1 x 4`) removes the side-by-side contention and, if a rerank scales
   to 4 threads, shortens each rerank, which lowers both queueing and TTFT. `4 x 1`
   maximizes concurrency and makes each rerank slower, which raises TTFT at every load;
   it ranks last. `2 x 2` is the baseline.
2. **Embedding off the rerank queue.** A question's dense embedding (about 20 ms of work)
   waits behind other questions' reranks: p95 0.9 s at 8 users, 1.4 s at 12, where
   first-question TTFT p95 misses its bound by 0.3 s. A separate small pool for the
   query embeddings, or a priority for them, removes that wait from TTFT. It is a code
   change in the backend's inference wiring and needs a test first.
3. **Embedding cache for repeated queries.** Saves about 20 ms of CPU per hit, so little
   throughput; it would skip the dense queue wait on a hit. The load questions come from
   a small fixed list, so the hit rate in a load run would overstate real traffic.
   Results with a cache are reported apart, with the hit rate, and never as the
   headline.
4. **Not candidates by this evidence:** database pool sizes and Postgres settings
   (Postgres at 1% CPU), Qdrant settings (search p50 at most 26 ms), more uvicorn workers (each
   process would add its own pool and model copies inside the same 4 CPUs and 2 GiB,
   with more of the concurrency that already costs CPU).

Changes that would affect answers, such as fewer rerank candidates or the reranker off,
are proposals for Ana Paula only, each with its consequence for the refusal gate's first
layer (the reranker's relevance score), and are not part of the tuning round.
