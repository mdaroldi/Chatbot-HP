# ADR 0008: Models baked into the image; inference threading

- **Status:** accepted (Phase 2, commits 6 and 7)
- **Date:** 2026-11-13

## Context

From Phase 2 the API loads three models on every start (dense bge-small, BM25 and a
cross-encoder reranker), and ingestion loads two. Until now `ingest` downloaded the
pinned files into a `model-cache` volume at runtime (ADRs 0006 and 0007). Every
container runs with a read-only root. A runtime download needs a writable cache and a
route to the Hugging Face hub from every container that loads a model, and fastembed
falls back to the OS temp directory when no cache directory is given.

## Decision

1. **One store, baked at build time.** A `models` stage in the backend Dockerfile runs
   `app.infra.model_store`. Every model in `models.toml` is fetched at its pinned
   revision, anonymously, with 3 attempts and a doubling backoff. The files are copied
   as real files (no symlinks) to `/opt/hpdocs/models` in the hub cache layout. The
   store must then hold exactly the pinned files: a changed, missing or extra file, or
   an unpinned model directory, fails the build. A BuildKit cache mount keeps the
   downloads between local builds. The stage copies only the modules it needs, so a
   code change elsewhere does not rerun it.
2. **Runtime is offline.** The runtime stage copies the store root-owned: the app user
   (10001) can read it but not change it. The image sets
   `EMBEDDING_CACHE_DIR=/opt/hpdocs/models`, `HF_HUB_OFFLINE=1` and the telemetry
   switches, and compose sets the same values. Models still load through
   `PinnedModelSource`, which hashes every pinned file at start, then passes
   `specific_model_path`, `cache_dir` and `local_files_only=True` to fastembed. The
   API and `ingest` read the same store. The `model-cache` volume is gone, and
   `ingest` keeps the edge network for the PDFs only.
3. **Tests and CI.** `make models` downloads and verifies every pin into
   `backend/models` for the model tests, and `make fetch-test-assets` now fetches every
   pin, rerankers included. The CI model cache stays keyed on `models.toml`. The CI
   image build downloads the models itself, with the same retries.
4. **Pins.** `make model-pins MODELS="..."` now pins the named models into the existing
   file and leaves the other pins untouched, so adding a model cannot move another
   model's revision. The two rerankers were added this way, and the dense and BM25 pins
   did not change.
5. **Both reranker candidates stay pinned** until the reranker ablation (ADR 0009)
   picks one. A separate commit then removes the losing model from `models.toml` and
   from the image, and this ADR records the size change.

6. **Inference threading (commit 7).** The API builds one instance per model (dense,
   BM25 and the configured reranker) in a background task started by the lifespan:
   verify the pins, build each model on the inference pool, warm each with one call.
   `/health` is green at once; `/ready` reports `models: loading` until warm-up ends,
   then the model names, or `models: failed`, with the exception type in the log only.
   The pool is a `ThreadPoolExecutor` with `INFERENCE_WORKERS` threads, and each model
   gets `INFERENCE_ONNX_THREADS` as its ONNX intra-op thread count; fastembed's
   `parallel` option is never used. Defaults until Phase 6: 2 workers × 2 threads.
   At startup, workers × threads is compared with the container's cgroup CPU quota
   (`/sys/fs/cgroup/cpu.max`), or with the CPU count when no quota is readable, and a
   warning names which one was used. A reranker other than the configured one is
   loaded only when asked for by name (the benchmark's ablation). Ingestion keeps
   leaving the thread count to onnxruntime.

## Model licenses

Checked on each repository's Hugging Face model card on 2026-11-13, at the pinned
revisions.

| Model (setting value) | Files come from | Revision | License |
|---|---|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` | same repository (ONNX export) | `a09144355adeed5f58c8ed011d209bf8ee5a1fec` | **The export's own card declares no license.** Its card names `base_model: cross-encoder/ms-marco-MiniLM-L-6-v2` and describes itself as that model with ONNX weights; that base model's card declares Apache-2.0 (checked 2026-11-13, base revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`). The Apache-2.0 call rests on the base model's card. fastembed's metadata also lists Apache-2.0. |
| `jinaai/jina-reranker-v1-turbo-en` | same repository | `b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2` | Apache-2.0, as declared on its card |

`jinaai/jina-reranker-v2-base-multilingual` is excluded, because its license is
CC-BY-NC-4.0. `BAAI/bge-reranker-base` (MIT, 1.04 GB) was not measured: on this CPU it
would cost about ten times MiniLM-L6.

## Measurements (Phase 2, commit 6)

**Image** (Apple M2, Docker Desktop with 8 CPUs and 3.8 GiB):

| | Unpacked root filesystem | Compressed |
|---|---|---|
| Phase 1 image | about 467 MB | 155 MB |
| With the store (4 models) | 765 MB | 369 MB |
| The store alone | 298 MB (dense 65, BM25 0.1, MiniLM-L6 88, jina-turbo 147) | +214 MB |

ONNX weights barely compress. Correction to ADR 0004: its "659 MB unpacked" is Docker
Desktop's image size column, which with the containerd store adds compressed and
unpacked bytes. The root filesystem measured with `du` is the unpacked size used
here.

**Build:** 75 s with an empty model cache (the models step took 17 s; other layers
cached). 15 s when only the store code changed and the downloads were cached.
Changing one digest in `models.toml` fails the build with
`cannot bake Xenova/ms-marco-MiniLM-L-6-v2: ModelIntegrityError`.

**Offline start:** in the image, with `--network none --read-only --cap-drop ALL`
and uid 10001, all four models verify against their pins, load and answer one call.
**Ingestion** with the store took 62 s for a forced build (40 s embedding and
upsert).

**Unexplained observation: one slow ingest (2026-11-13).** The first `docker compose
up --build` after the store was added spent 12 min 42 s between the last
`document chunked` line (11:42:30 UTC) and `index build published` (11:55:12), against
about 40 s normally. Conditions: the image and its store had just been baked
(first build with the models stage); ingestion ran with its usual settings (fastembed
`threads` unset, so onnxruntime used all 8 vCPUs, batch size 32); the container showed
790% CPU and 1.48 GiB of 3.8 GiB; while it ran, the Docker API answered one request
with a 500 error. It did not recur: a forced build right after took 62 s, and embedding
all 287 chunks alone in the same image takes 38 s (35.6 s with 4 threads). The cause
is not known, and the log could not say which stage was slow. Since Phase 2 commit 6c
every document line carries its seconds and each of `embed_dense`, `embed_sparse` and
`publish` logs one `stage done` line with its item count and seconds. **If it recurs,
it is investigated before Phase 6.**

**Inference latency.** Warm, 30 repetitions after 3 warm-up calls, 20 real chunks
(mean 930 characters, breadcrumb included), p50 / p95 in ms:

| Stage | Threads | macOS native | In the image (Linux aarch64) |
|---|---|---|---|
| Dense query embedding (1 query) | 1 / 2 / 4 | 8.3 / 5.3 / 6.1 (p50) | 9.9 / 6.8 / 3.6 (p50) |
| Dense embedding of 20 chunks | 1 | 4,704 / 5,057 | 4,466 / 7,580 |
| | 2 | 2,471 / 2,695 | 2,252 / 3,680 |
| | 4 | 2,333 / 3,097 | 1,357 / 3,060 |
| Rerank 20, MiniLM-L6 | 1 | 2,610 / 3,601 | 2,712 / 3,326 |
| | 2 | 1,525 / 1,812 | 1,444 / 2,890 |
| | 4 | 1,150 / 1,423 | 827 / 941 |
| Rerank 20, jina-turbo | 1 | 3,413 / 5,279 | 3,217 / 5,251 |
| | 2 | 2,470 / 3,412 | 1,680 / 1,937 |
| | 4 | 2,127 / 3,912 | 1,050 / 1,312 |

BM25 query encoding takes about 0.01 ms. Reranking dominates a request by two orders
of magnitude. Memory after load, before inference (container `VmRSS`): dense 103 MB,
MiniLM-L6 49 MB, jina-turbo 90 MB.

**Reranker input.** fastembed truncates both rerankers at 512 tokens; jina's 8K
context is not used. The reranker sees the stored chunk text, breadcrumb included.
None of the 13,776 pairs of an in-scope dev question with a chunk is longer (the
longest dev question is 26 tokens). A query at the `QUERY_MAX_CHARS` limit of 1,000
characters is truncated together with 55 of the 287 chunks: the tokenizer cuts the
pair to 512 tokens, so the reranker scores part of the pair. `QUERY_MAX_CHARS` stays
at 1,000 for now; **Phase 3 takes this as an input for the chat message length limit**
(the standalone query, after rewriting, is what reaches the reranker).

## Alternatives considered

- **A named volume filled by a one-time service:** a smaller image, but every fresh
  stack needs the hub at runtime, the volume can be changed after verification, and
  the API container would need a writable mount or the same downloader.
- **Download at API start:** a network dependency and a cold start on every replica.

## Consequences

- No container downloads a model at runtime; a stack with no route to the hub still
  starts, provided the corpus is cached.
- The image grows by 298 MB unpacked (214 MB compressed) while both rerankers are in
  it; about 147 MB unpacked goes away if MiniLM-L6 wins.
- The image build needs the hub; the PDFs are still downloaded at runtime by
  `ingest` (ADR 0007).
- **CI cost, accepted:** the CI `docker` job builds with a plain `docker build` and
  downloads about 300 MB of models on every run (the cache mount does not survive
  between runners). Caching the models stage with the buildx GitHub Actions cache was
  evaluated and not adopted: it needs new third-party actions (pinned by SHA, with the
  cooldown) and a different build command than `make images`, and that cache stores
  layers only, so it helps only while `models.toml` and the store code are unchanged.
  Revisit if the job time becomes a problem.
- Changing a pin rebuilds the store; `--if-needed` still compares only the dense and
  BM25 pins (ADR 0006), so a reranker change never rebuilds the index.
