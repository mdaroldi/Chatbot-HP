# Load test SLO

The service level objective that every capacity step is judged against. It was fixed
once, from the single-user baseline, before any ramp ran, and is not edited afterwards.
`python -m hpdocs_load.report` reads the TOML block below.

## Classes

Questions fall into two classes, each with its own bound:

- **First questions**: the first question of a conversation. No rewrite before retrieval.
- **Follow-ups**: a later question. The backend first rewrites it into a standalone
  query, and the stub provider streams that rewrite at 160 ms a piece (about 2.7 s), so
  a follow-up waits for the rewrite before retrieval starts. With one bound for both
  classes, the share of follow-ups at a step would move the capacity number.

## Rule

A step meets the SLO when all of these hold over its whole measured window (warm-up
excluded):

- for each class, p95 time to first token (TTFT) <= 2 x that class's single-user p95;
- for each class, p95 time to `done` <= 2 x that class's single-user p95;
- errors are under 1% of all questions, counting both classes together. A 429 is not an
  error here: rate-limit responses are reported apart, since capacity runs raise the
  limits to 10,000 per minute.

Per-minute values are evidence of stability only; they do not decide a step. A step
also needs to count (window of at least 180 s, Locust process CPU under 90%) and to
stay valid on the host (no sleep or wake, no sampler gap over 15 s, swap growth under
500 MiB, at most 3,000 page-outs per minute).

## Baseline

Run `baseline-1u`, 2026-11-16 11:48-12:04 CET: one user, 15 minutes measured after
60 s of warm-up, stub provider (Profile A), load stack (backend 4 CPUs / 2 GiB). No
errors, no 429s.

| Class | Questions | TTFT p50 / p95 s | Done p50 / p95 s |
|---|---|---|---|
| First questions | 13 | 1.94 / 2.07 | 4.95 / 5.05 |
| Follow-ups | 20 | 4.49 / 4.56 | 7.47 / 7.58 |

The question counts include the warm-up. With so few questions, each p95 is close to the
slowest one; the single-user times are steady (p95 within 7% of p50), so 2x leaves room
for queueing alone. Profile A TTFT excludes provider latency: the stub replaces the
real LLM's time to first token with fixed per-piece delays.

```toml
max_error_rate = 0.01

[first]
ttft_p95_ms = 4140
done_p95_ms = 10100

[follow_up]
ttft_p95_ms = 9120
done_p95_ms = 15160
```
