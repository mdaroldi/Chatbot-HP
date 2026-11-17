# Profile B: capped run on the real LLM

Profile B measures the whole path end to end: retrieval, reranking and the real
Anthropic models, with the provider's own latency included. A spend cap keeps it short,
so it is one small run. **It is bounded by provider limits:** its throughput and
latency depend on the provider's latency and on the account's rate limits, which this
deployment does not control. It is not a capacity table, and it must not be extrapolated
from or to Profile A (stub LLM; see [`tuning.md`](tuning.md)). The SLO in
[`../SLO.md`](../SLO.md) is Profile A's: its bounds come from the stub baseline and do
not apply here.

**Result:** 67 questions at 4 users, all answered (0 errors, 0 provider 429s). First
questions had a p50 / p95 time to first token of 1.71 / 2.23 s; follow-ups, which add a
real rewrite call, had 2.65 / 3.51 s. Measured spend was at most **$0.53**, against a
$3.00 cap and a $0.55 estimate.

## Setup

- **Run:** `profile-b`, 2026-11-17 10:41–10:48. One step of 4 users with a 60 s warm-up,
  the step planned at 600 s:
  `make load-measured LOAD_PROVIDER=anthropic LOAD_RUN=profile-b LOAD_STEPS=4
  LOAD_STEP_SECONDS=600 LOAD_WARMUP_SECONDS=60`.
  The workload is the same as Profile A ([`baseline.md`](baseline.md)): dev questions,
  1–3 follow-ups, 10–30 s of think time.
- **Models:** answers `claude-sonnet-5-5`, rewrites `claude-haiku-5-5` (compose
  defaults). `docker exec … printenv` on the backend showed `LLM_PROVIDER=anthropic` and
  `LLM_MAX_RETRIES=0`.
- **One attempt per call:** [`../load.env`](../load.env) sets `LLM_MAX_RETRIES=0`, so
  each provider call is exactly one attempt. A provider 429 or 5xx would show as an
  error and would be reported, not retried.
- **Spend guard** (`hpdocs_load.budget`; `make load` turns it on whenever
  `LOAD_PROVIDER=anthropic`):
  - The cap is $3.00, with a hard stop at 90 questions.
  - Each question is charged its worst case before it is sent: one answer call
    (12,000 input + 2,048 output tokens = $0.04448), plus one rewrite call for a
    follow-up (4,000 + 100 tokens = $0.00045). These are the token bounds and prices
    from the Phase 5 answer matrix (prices as of 2026-11-03).
  - At worst case the dollar cap binds first, at about 67 questions. The guard stopped
    the run at 67 questions, 7.5 min in, so the measured window is 6.5 min. That is
    longer than the 3 minutes a step needs to count.
- **Rate limits:** the load profile's 10,000 per minute, as in every capacity run.
- **Host:** the run started after a 5-minute memory check: swap 2,107 → 2,075 MB. Only
  Docker Desktop, VS Code and Finder were open, on AC power (checked with `osascript`,
  `pgrep` and `pmset`). The run was under `caffeinate`, with no sleep or wake events in
  the run window.

## Result

| Users | Min | Q/min | Req/min | Err % | 429 | First TTFT p50 / p95 s | First done p50 / p95 s | Follow-up TTFT p50 / p95 s | Follow-up done p50 / p95 s | Rerank p50 / p95 s | TTFT - retrieval p50 / p95 s | Backend CPU mean / max % | Locust CPU max % | Swap growth MiB | Counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 6.5 | 9.2 | 23.7 | 0.00 | 0 | 1.71 / 2.23 | 2.58 / 4.90 | 2.65 / 3.51 | 3.27 / 4.74 | 0.96 / 1.08 | 1.53 / 2.20 | 80 / 400 | 1 | -56 | yes |

*Bounded by provider limits. One measured window of 6.5 min, cut by the spend guard.*

- **Throughput:** 9.2 questions per minute is the pace of 4 users who think for 10–30 s
  between questions. It is not a ceiling of the system.
- **Where the time goes:** "TTFT − retrieval" is the provider's part: the rewrite call
  for a follow-up, then the answer model's time to first token. Its p50 of 1.53 s sits
  next to Phase 5's 980 ms single-request first-token p50. The difference is the
  follow-up rewrites, plus whatever provider variance there was on the day. Retrieval
  stays local and small at 4 users (rerank p50 0.96 s).
- **Whole run, client side (warm-up included):** 181 requests, all 2xx; 67 messages,
  all ending with `done` (42 of them follow-ups). No error codes.
- **Outcomes, from the backend logs:** 62 answers completed and 5 were refused: 3 below
  the retrieval threshold (no model call) and 2 by the model. All 42 follow-up rewrites
  succeeded, with no fallback.
- **Host:** swap 2,075 → 2,019 MiB (peak growth 0), at most 106 page-outs per minute,
  largest sampler gap 7 s.

## Provider limits and errors

There were no provider 429s, 5xx responses or timeouts. The backend logged no warnings
and no errors (517 INFO lines), and the client saw no error codes. Because retries were
off, none was hidden. At this pace the account's limits were never reached, so this run
does not measure them. A higher-rate Profile B run would hit the account's request and
token limits before it hit the backend's ceiling. That is why the table is labelled
"bounded by provider limits".

## Spend

| | USD |
|---|---|
| Cap | 3.00 |
| Charged ceiling (guard, worst case: 67 answer calls + 42 rewrite calls) | 2.99906 |
| Measured: answer calls (64 calls; 219,590 input and 6,964 output tokens at $2 / $10 per Mtok) | 0.5088 |
| Measured: rewrite calls, upper bound (42 calls at the $0.00045 worst case) | 0.0189 |
| **Measured total (upper bound)** | **0.528** |
| Estimate before the run (67 × Phase 5 mean of ~$0.0083) | 0.55 |

- **Answer cost** comes from the token counts the backend logs with each answer (the
  "chat answer" log line). Each count is priced at the answer model's dated price.
- **Rewrite token counts are not logged.** Each rewrite call is therefore charged its
  worst case, which makes the total an upper bound.
- **Provider calls:** 106 in all: 64 answer calls and 42 rewrite calls. The guard
  charged 109, because it also charged the 3 refusals below the threshold, which never
  reach the model.
- **Biggest single answer call:** 4,293 input tokens (bound: 12,000) and 642 output
  tokens (bound: 2,048).
- **Against the estimate:** the measured total came in about 4% under it, because the
  3 threshold refusals cost nothing.

## Data handling

- **The key:** it lives only in the git-ignored `.env`. A search for its value after the
  run found it in no run file, report, log copy, tracked or untracked file, git history,
  or backend container log.
- **What was sent to Anthropic:** dev questions from the golden set and the retrieved
  excerpts of the two HP manuals. No personal data.
- **After the run:** a plain `make load-reset` put the load stack back on the stub
  (`LLM_PROVIDER=stub`).
