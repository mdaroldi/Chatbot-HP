# ADR 0012: Answer benchmark: harness, judge, budget and the dev ablation

- **Status:** accepted (Phase 5)
- **Date:** 2026-11-15

## Context

Phase 2 measured retrieval alone (ADR 0009). Phase 5 measures the answer end to end on
the real API, and asks which design choices matter: search mode, reranker, chunker, the
number of excerpts given to the model, and the query rewrite. The constraints were a hard
budget of $10 for every paid call of the phase (answers, rewrites and judge), answers that
cannot be made deterministic (no sampling parameters), two small hand-written datasets,
and the rule that HP text never lands in the repository.

The datasets are `eval/golden/answers-v1.toml` (key facts for each retrieval-v1 question)
and `eval/golden/slices-v1.toml` (multi-turn, long-query, out-of-scope and adversarial
items), both 1.0.0, written by chat Claude from the PDFs, approved by Ana Paula and frozen
at `c66ccd4`. 135 items: dev 105, held-out 30.

## Decision

### Harness: it reads the API the UI reads

- The harness lives in `backend/app/evaluation/answers/` and runs in the backend image,
  so it shares the domain code and the 90% gate (kept at 100%). It sends each item over
  the streaming `POST /v1/conversations/{id}/messages`, one item at a time, with a fresh
  UUID4 `X-Client-ID`; earlier turns of a multi-turn item are replayed as real turns.
- Flags come from the `done` event. A field the harness needs that is missing or of the
  wrong type fails loudly. A failed item is a result with an error code; more than 5%
  errors marks the run invalid. Only 429 and 503 before the first token are retried, at
  most twice, honouring `Retry-After`; retries are counted.
- **Excerpts the model saw** come from the stored rows, with no backend change: assistant
  rows read from Postgres in a READ ONLY transaction (chunk ids in rank order, model,
  tokens, rewritten query, build id, relevance signal, timings), chunk payloads from
  Qdrant by id, excerpts re-rendered with `app/domain/prompt.py`. The conversation is
  deleted through the API after its rows are saved. Raw outputs stay in the git-ignored
  `eval/answer-runs/`.
- **Config per run.** A host-side driver restarts the backend with the run's overrides
  (`RETRIEVAL_MODE`, `RETRIEVAL_RERANK_ENABLED`, `RETRIEVAL_TOP_K`,
  `CHAT_REWRITE_HISTORY_TURNS`, `QDRANT_COLLECTION_ALIAS`), checks them with `printenv`
  (an allowlist, never the key), `/ready` and every stored row (model, build id, number
  of chunks, relevance signal present iff rerank, no rewrite iff rewrite off), and
  restores the defaults in `finally`. The harness container never holds the API key.

### Metrics and statistics

- **Primary:** strict key-fact pass on the 48 single-turn in-scope dev questions of
  answers-v1. A refusal of an in-scope question is a fail. Slices are reported apart
  and never pooled.
- Proportions carry k/n and a Wilson 95% interval. Differences are paired bootstrap 95%
  intervals (2000 resamples, seed in `eval/answer-matrix.toml`). The noise floor is the
  largest minus the smallest value over three reference runs, never less than 1/n.
  Reference run 1 is the comparator.
- **Rule 7, a pure function** (`answers/decision.py`): a default changes only if the
  alternative's primary interval excludes zero (above it), the gap exceeds the noise
  floor, and false refusals and near-domain refusal recall do not worsen by more than
  their noise floor. A reranker drop is deferred to Phase 6 whatever the result.

### Judge

- `claude-opus-5-5` (never the answer model) through the Batch API, structured output,
  effort medium, no temperature. Rubric `judge-v1`: each claim supported, partial or
  unsupported against the excerpts the model saw; correctness against the anchor gold
  chunks plus the answers-v1 key facts. The cache key is the SHA-256 of model, rubric,
  question, excerpts, reference and answer.
- Two faithfulness figures: **lenient** (no unsupported claim, judge-v1's definition) and
  **strict** (every claim supported).
- Judged only: reference run 1, the held-out run and a 20-item repeatability sample
  judged twice with the cache bypassed.
- **Data flow:** the judge receives HP excerpts and model answers; with the judge on,
  they are sent to the Anthropic API, as user questions and excerpts already are by the
  answer path.

### Budget

- `eval/answer-matrix.toml` holds the cap ($10), dated prices and token bounds. An
  append-only ledger (git-ignored) is a hard stop: before each item, spent plus that
  item's worst case must stay under the cap, and before each run the remaining budget
  must cover the run's estimate plus 25%. Answer calls are charged from stored token
  counts (measured); rewrite calls, whose usage the backend does not store, are charged
  at their token bound (estimated). `make eval-estimate` shows both.
- Cuts made to fit: ablation rows on the 56 answers-v1 dev items only; slices only on the
  reference runs and the final default; no Opus answer row; no paid fresh-clone check
  (`make eval EVAL_PROVIDER=stub` checks the pipeline instead); no neighbor expansion; no
  naive recalibration; no judge on ablation rows.

### Naive chunker baseline

`CHUNKER=naive` builds fixed windows of 256 tokens with 64 overlap, prefixed by the
document title only (`naive-v1`), into the alias `hpdocs_chunks_naive`. It keeps the
structure-aware threshold (-6.54), so its refusals are reported apart, and its primary
metric is also given on the items neither run refused at layer 1. Its section path is the
document title, so anchor-based measures (gold cited) do not apply to it.

## Results

Spend: $5.41 of $10 before the held-out run. Every dev run was valid with 0 errors (k3
had one counted retry). All runs used index build `20261113t173108z`, except `naive`
(`naive-v1_20261115t100901z`). Full tables:
[`eval/reports/answers-summary.md`](../../eval/reports/answers-summary.md).

**Reference (hybrid + rerank, k 5, rewrite on).** Primary 45/48 in each of the three runs,
Wilson [0.832, 0.979]; false refusals 2/48 each; near-domain recall 4/4 each. Noise floor
1/48 (0.021) on both primary and false refusals.

| row | primary | difference vs reference-1 [95%] | false refusals | layer 1 refused | verdict |
|---|---|---|---|---|---|
| hybrid without rerank | 44/48 | -0.021 [-0.083, +0.042] | 2/48 | 0 | keep |
| dense (no rerank) | 42/48 | -0.062 [-0.125, +0.000] | 1/48 | 0 | keep |
| sparse (no rerank) | 43/48 | -0.042 [-0.125, +0.042] | 2/48 | 0 | keep |
| naive chunker | 41/48 | -0.083 [-0.167, -0.021] | 4/48 | 7 | keep |
| k = 3 | 41/48 | -0.083 [-0.167, -0.021] | 5/48 | 6 | keep |
| k = 8 | 44/48 | -0.021 [-0.062, +0.000] | 2/48 | 6 | keep |

- **Rule 7 keeps every default.** No row beats the reference. Naive and k = 3 are the
  only rows whose interval excludes zero, below it, and both raise false refusals: that
  is evidence they are worse. For the others there is no evidence of a difference.
- **Rule 7 cannot pass at this ceiling.** With 45/48, an alternative that fixed exactly
  the three failures would still have a lower bound of zero: a resample misses all three
  with probability (45/48)^48 ≈ 0.045, above the 0.025 tail. So the primary metric
  cannot separate configurations here, and the weight of the findings is in the slices
  and the refusals below.
- **Reranker.** Dropping it costs one question (not distinguishable) and saves about 2 s
  of server time to first token (p50 992 ms without it against 2977 to 3380 ms on the
  answers-dev reference runs; same laptop, not a load test). Without rerank there is no
  layer 1 gate. The decision is Phase 6's; the masked Phase 2 ablations are owed only if
  a no-rerank default is proposed, and none is.
- **Rewrite off** (multi-turn slice, descriptive): 7/10 against 10/10.
- **Judge (reference-1).** Lenient faithfulness 67/67 [0.946, 1.000]; strict 65/67
  [0.898, 0.992]; correctness 64 correct, 2 partial, 1 incorrect. Repeatability 20/20 on
  both. Ana Paula's 20 blind labels: faithful 20/20 (kappa n/a: both raters said yes on
  every item, so chance agreement is 1), correctness 20/20 (kappa 1.00). Five corrupted
  answers (negative controls): correctness flagged 5, strict faithfulness 5, lenient 4.
  **The judge graded a wrong value ("10 -> 3 seconds", nc-omen-procedure-001) as partial,
  not unsupported, so lenient faithfulness missed it.** Both figures are reported; strict
  catches it. The faithful agreement is weak evidence, since no label sheet item was
  unfaithful.

**Held-out (rule 8: once, on the final default, after every decision).** The reference
config, 30 items, 0 errors, judged. Primary 11/12, Wilson [0.646, 0.985]; the one fail is
envy-factual-004, refused by the model. False refusals 2/19 (envy-factual-004, lq-010,
both model refusals; none at layer 1). Out-of-scope recall 9/9; adversarial 4/4; gold
cited 15/15. Judge: lenient faithfulness 17/17, strict 16/17, correctness 17 of 17
correct. The one held-out rewrite trap, mt-013, is not clean, which repeats the dev
finding below. No held-out item expects a safety note. The held-out numbers agree with
dev within their wide intervals; n = 12 cannot show more.

Spend for the phase: $5.85 of $10 (measured $5.84, estimated $0.01).

## Known limitations and next steps

The owner skipped the improvement round (rule 9): the rewrite prompt, the answer system
prompt and the refusal threshold are unchanged, and the summary's `improvement` row reads
"not run" for that reason (its reserved $0.57 was not spent). These findings from reference-1 are the
next steps, by item id:

1. **Safety warnings carried 4/6.** envy-warning-002 and lq-008 answer without the
   source's WARNING or CAUTION line. Fixing this means changing the answer system prompt,
   which goes to Ana Paula first (the frozen prompt_leak patterns adv-006, 007, 008 and
   015 test it, and a change needs a new dataset version).
2. **Rewrite traps clean 1/3.** On mt-002 and mt-009 the rewrite adds a product name the
   user did not give (and on held-out mt-013). The rewrite prompt is separate from the answer prompt; a fix there
   is the cheapest next change, measured on the multi-turn slice.
3. **Layer 1 refusals of answerable questions:** envy-table-003, both-ambiguous-001,
   lq-001 and lq-004 fall below the relevance threshold (-6.54). Recalibration was out of
   scope (report only); it rests on 8 out-of-scope questions (ADR 0009) and should be
   redone on the larger out-of-scope set now available.
4. **Long queries 3/8.** Besides lq-001 and lq-004 (layer 1), lq-003 is refused by the
   model and lq-006 and lq-007 miss key facts. Padding a question to 1,000 characters
   dilutes both the reranker score and the answer.
5. **adv-011 (format hijack)** answers but hits a forbidden pattern: the one adversarial
   failure (13/14).

## Alternatives considered

- **A debug field on the API exposing chunk ids**, or re-running retrieval in the
  harness: the first is a backend change, the second can differ from what the model saw.
  Reading the stored rows needs neither.
- **Judging every ablation row:** about 48 more judge requests per row (around $0.60 a
  row at the measured $0.0123 per request). It was cut when the budget was planned, before
  the judge was measured; the deterministic key facts separate the rows as well as
  n = 48 allows.
- **A Sonnet judge:** cheaper, but the judge must never equal the answer model.

## Consequences

- Every default from Phases 2 and 3 stands. The ablation shows which choices matter in
  the measurable direction: structure-aware chunking and k = 5 over k = 3.
- 48 primary questions put the reference near the ceiling; a future dataset version needs
  more, harder in-scope questions before rule 7 can detect an improvement.
- `make eval-answers`, `eval-judge`, `eval-report` and `eval-summary` reproduce any row
  within the budget; `make eval` runs the reference once. The answers differ run to run,
  so a rerun matches within the noise floor, not exactly.
