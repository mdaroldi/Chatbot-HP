# Rate limits on, Profile A (stub LLM)

Capacity runs set every rate limit to 10,000 per minute
([`../load.env`](../load.env)), so a 429 there would be an error. This run puts the
compose defaults back for one 5-minute step and adds a client that exceeds them. It
answers two questions: does the limiter refuse the abusive client with the documented
response, and do normal users next to it go through untouched? **This run is outside
the SLO.** It is not a capacity result and appears in no capacity table.
[`../SLO.md`](../SLO.md) is unchanged.

**Result:** the burst client's messages were held to the configured rate: 10 per minute
after a burst of 5. Everything above that got `429 rate_limited` with `Retry-After`.
The 16 normal users had 0 429s and 0 5xx. Refusals are cheap per request but not free:
at ~460 refused requests per second, normal users' latency rose well past the SLO
bounds (see "Cost of refusals").

## Setup

- **Limiter:** chat routes are keyed by the `X-Client-ID` header they require (a
  missing ID is 400). Only `/v1/search` falls back to the peer IP (ADR 0010, "Rate
  limiting"). So all Locust users, which share 127.0.0.1, are still limited one client
  ID at a time.
- **Limits (compose defaults):** messages 10/min with a burst of 5, conversations
  120/min with a burst of 30, search 30/min with a burst of 10. They were set in the
  shell for this run only; shell values take precedence over `--env-file` in compose
  interpolation.
  - *During the run:* `docker exec … printenv`, on the same backend container right after
    it, showed `RATE_LIMIT_MESSAGES_PER_MINUTE=10`, `…_MESSAGES_BURST=5`,
    `…_CONVERSATIONS_PER_MINUTE=120`, `…_CONVERSATIONS_BURST=30`,
    `…_SEARCH_PER_MINUTE=30`, `…_SEARCH_BURST=10`.
  - *Afterwards:* a plain `make load-reset` restored the load profile, and every value
    read 10000 again.
- **Normal users:** 16, the tuned headline ([`tuning.md`](tuning.md)), with the
  workload from [`baseline.md`](baseline.md): one client ID each and 10–30 s of think
  time.
- **Burst client** (`LOAD_BURST=1`, `hpdocs_load.burst`):
  - 3 Locust users share one client ID, each in its own conversation. They ask the fixed
    dev first questions back to back with no think time and no backing off on a 429.
    They come on top of the 16 users, and their requests go to their own file
    (`burst.jsonl`), so the normal users' numbers never include them.
  - Why three: a single sequential user never exceeds the limit. Each later question in
    its conversation is a follow-up (stub rewrite plus answer, ~6 s), so it sends about
    10 messages per minute, the refill rate. A 150 s smoke showed 25 messages and no 429.
- **Step and timing:** one 5-minute step: 60 s warm-up, 4 minutes measured, spawn rate 4.
  Run through `make load-measured` on 2026-11-17, 10:01–10:07 (window 09:36–10:07).
  Backend 1 worker × 4 ONNX threads, image `sha256:ae823a25906c…`.
- **Host:**
  - *Memory check before the run:* swap 1,665 → 1,856 MB over 20 minutes, an average of
    +48 MB per 5 minutes.
  - *During the run:* swap 1,856 → 1,848 MiB, peak growth 0, page-outs 459/min, largest
    sampler gap 7 s.
  - *Sleep:* no sleep or wake events (pmset) during the window.

## Burst client

| | Measured window (4 min) | Whole run (incl. warm-up) |
|---|---|---|
| Requests | 110,577 (27,644 / min) | 138,376 |
| 200, answered to `done` | 40 (10.0 / min) | 53 |
| 429 `rate_limited` | 110,537 | 138,320 |
| Other statuses | none | 3 × 201 (one conversation per user) |

- **Limiter accuracy:** 53 answers in ~5.1 minutes matches the configured rate: 5 from
  the full bucket plus 10 per minute.
- **When 429s began:** the first 429 came 10.9 s after the start, once the burst of 5
  was spent.
- **Speed of a refusal:** 2.6 ms at p50, 8.4 ms at p95, 22 ms at p99 (128 ms max),
  measured from the client.

## Response shape

The burst records hold only times, statuses and the error code, which the harness reads
from the body's `error.code`. Headers were checked with a probe on the same container
right after the run: a fresh client ID sending 7 messages to an unknown conversation.
The probe got `404 404 404 404 404 429 429`. The limiter runs before the conversation
lookup, so a 404 also spends a token. The next 429, verbatim (server text, no user
content):

```
HTTP/1.1 429 Too Many Requests
retry-after: 6
content-type: application/json
x-request-id: 5a905358-622c-4aad-8683-54f6269bc930

{"error":{"code":"rate_limited","message":"Too many requests; retry after the time in Retry-After.","request_id":"5a905358-622c-4aad-8683-54f6269bc930","details":[]}}
```

`Retry-After` is sent, in whole seconds rounded up: 6 s is one token at 10 per minute.
The body uses the API's unified error envelope.

## Normal users

| Users | Min | Q/min | Req/min | Err % | 429 | 5xx | First TTFT p50 / p95 s | Follow-up TTFT p50 / p95 s | Rerank p50 / p95 s | Backend CPU mean / max % | Locust CPU max % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4.0 | 30.5 | 78.8 | 0.00 | 0 | 0 | 3.25 / 6.00 | 6.10 / 9.30 | 1.39 / 4.90 | 346 / 415 | 56 |

The whole run, warm-up included, also had no 429s and no 5xx: 354 × 200, 59 × 201 and
3 × 204.

## Cost of refusals

Compared with the same-length 16-user step without the burst client (w7b-1x4-hi,
[`tuning.md`](tuning.md)):

| | Q/min | First TTFT p95 s | Follow-up TTFT p95 s |
|---|---|---|---|
| Without burst client | 32.8 | 2.96 | 5.17 |
| This run | 30.5 | 6.00 | 9.30 |

In this run, backend CPU averaged 346% of 4 cores.

The burst client added two kinds of load, and this run cannot separate them:

- 10 answered questions per minute, real work through retrieval and reranking. Together
  with the users' own ~31 Q/min, that is about the 20-user load, which already failed
  the SLO on its own (tuning.md: first TTFT p95 4.29 s, follow-up 6.75 s at 20 users).
- ~460 refused requests per second on the single uvicorn event loop.

The latency rise is larger than the 20-user step's. So the refusals likely cost
something on top, but that is an inference, not a measurement.

What it means for deployment:

- The in-process limiter protects the model and inference capacity: the burst client
  never got more than its quota.
- It does not protect the event loop from a request flood. That belongs in front of
  the API (a reverse proxy or gateway limit per IP and per client ID), together with
  the shared limiter store that replicas would need anyway (ADR 0010).
- The client ID is chosen by the client, so a determined client can also rotate IDs
  (ADR 0010, "Limitation").

## Reproduce

```sh
RATE_LIMIT_CONVERSATIONS_PER_MINUTE=120 RATE_LIMIT_CONVERSATIONS_BURST=30 \
RATE_LIMIT_SEARCH_PER_MINUTE=30 RATE_LIMIT_SEARCH_BURST=10 \
RATE_LIMIT_MESSAGES_PER_MINUTE=10 RATE_LIMIT_MESSAGES_BURST=5 \
  make load-measured LOAD_RUN=rate-limits LOAD_STEPS=16 LOAD_STEP_SECONDS=300 \
  LOAD_WARMUP_SECONDS=60 LOAD_SPAWN_RATE=4 LOAD_BURST=1
make load-reset   # back to the load profile's limits
```
