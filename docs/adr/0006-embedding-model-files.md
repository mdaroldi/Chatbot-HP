# ADR 0006: Pinned and verified embedding model files

- **Status:** accepted (Phase 1; schema in commit 9, implementation in commit 9b)
- **Date:** 2026-11-12

## Context

A build depends on the model files as much as on the PDFs: the dense model's
tokenizer sets chunk boundaries (ADR 0005), and the ONNX weights and BM25 files set
the vectors. A file that changes without anyone noticing breaks traceability and the
`--if-needed` rule, and it could make query vectors differ from document vectors.

What fastembed 0.9.1 does today, measured on the local cache:

- **No pinned revision.** On download it asks the hub for the repository's current
  HEAD (`model_info(...).sha`) and fetches that, so the same settings can produce
  different files on different days. Cached revisions: dense
  `Qdrant/bge-small-en-v1.5-onnx-Q` at `aa8f8b06…` (5 files, about 66 MB), sparse
  `Qdrant/bm25` at `22b8d2af…` (18 small files).
- **Weak integrity check.** It records each file's size and the hub's git blob ID.
  Online, it compares both with the hub. Offline, it compares sizes only. For LFS
  files, such as the ONNX weights, the blob ID belongs to the pointer, not the content.
- **Loading from a directory works.** `TextEmbedding` and `SparseTextEmbedding` both
  accept `specific_model_path` and load from it offline, without network and without
  writing to the cache (verified for both models).

## Decision

1. **Pins file.** `backend/models.toml` (committed, like `corpus.toml`) lists each
   allowed model: the fastembed name (`BAAI/bge-small-en-v1.5`, `Qdrant/bm25`), the
   hub source repository, the revision (full commit SHA), and the SHA-256 of every file
   fastembed loads. `EMBEDDING_DENSE_MODEL` and `EMBEDDING_SPARSE_MODEL` still choose
   the model. A name missing from the pins file fails ingestion
   (`ModelNotAllowedError`).
2. **Fetch the pinned revision.** A new adapter `infra/model_files.py` (in `infra/`,
   not `ingestion/`, so the backend can reuse it in Phase 2) looks for
   `EMBEDDING_CACHE_DIR/<repo>/snapshots/<revision>/`. If every pinned file is there
   with the right hash, it uses that directory without any network call. Otherwise it
   calls `huggingface_hub.snapshot_download(repo, revision=<pin>,
   allow_patterns=<pinned files>)` and hashes again. huggingface-hub becomes a direct
   runtime dependency in `pyproject.toml`, with fastembed's cap made explicit
   (`<2.0`). It is already locked (1.33.0, Apache-2.0), so no new package is added.
3. **Hash after download, and on every build.** Every file is hashed with SHA-256,
   streamed in a worker thread, on each build, including from cache (about 66 MB, well
   under a second). fastembed then loads the models with
   `specific_model_path=<verified directory>`, so the files it loads are the files that
   were hashed. The token counter loads from the same directory.
4. **Mismatch.**
   - A cached file that does not match its pin: that revision's snapshot is deleted and
     downloaded once more, as the corpus downloader does for a bad cached PDF.
   - Still no match after the download: `ModelChecksumMismatchError`, which names the
     model, the file and both digests. A download failure raises
     `ModelDownloadError` instead, with a clearly different message.
   - Both errors happen before the build starts: there is no build ID, no collection
     and no `index_builds` row. Ingest writes one structured log line (model name, file
     path inside the revision, error type) and exits non-zero. The alias and
     the live index stay untouched, and the backend does not start (it waits for
     `ingest`).
   - `index_builds.status` therefore covers only builds that started: `succeeded`, or
     `failed` for a build that got a build ID and then stopped.
5. **What each build stores.** `index_builds` gets one JSON column, `models`:
   `{"dense": {"name", "source", "revision", "files": {path: sha256}}, "sparse": {…}}`.
   It holds the digests computed in step 3, not a copy of the pins. The `dense_model`
   and `sparse_model` string columns stay, for simple queries.
6. **`--if-needed`.** The skip check also compares the pins of the configured models
   (revision and every file hash) with the live build's `models`. It reads only
   `models.toml`, so the skip path still never hashes, downloads or loads a model. Any
   difference triggers a rebuild, and a unit test covers it.
7. **Changing a pin** is a reviewed commit to `models.toml`. A `model`-marked test
   downloads each pinned revision into a temporary cache and checks every digest. CI
   runs it from commit 11.
8. **Regenerating the pins file.** `make model-pins` (commit 9b) runs a small CLI in
   the backend image's code. For each configured model it asks the hub for the
   current revision of the model's source repository, downloads that revision into a
   temporary directory (never into `EMBEDDING_CACHE_DIR`), hashes every file fastembed
   loads, and rewrites `backend/models.toml`. It never commits. The maintainer then:
   - reviews the diff (revision, file list, digests);
   - re-checks the license on each repository's model card, as in the table below;
   - runs `make test-int`, because a new tokenizer can move chunk boundaries, which
     the chunk-ID digest test reports (and which needs a `chunking_version` bump);
   - commits `models.toml` on its own, with the old and new revisions in the message.

## Model licenses

Taken from each repository's Hugging Face model card (license field and card text),
checked on 2026-11-12. The revisions are the ones pinned first.

| Model (setting value) | Files come from | Revision | License |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | `Qdrant/bge-small-en-v1.5-onnx-Q` (quantized ONNX export) | `aa8f8b060edb00e03bfdd08813a2949946c8ba55` | MIT (export); the original `BAAI/bge-small-en-v1.5` is MIT too |
| `Qdrant/bm25` | `Qdrant/bm25` (stop-word lists and config, no weights) | `22b8d2af71a76161e18dd432d2cee0eefa66e412` | Apache-2.0 |

## Alternatives considered

- **Record hashes after fastembed's own download, without pins:** traceable after the
  fact, but an upstream change still slips into a build silently and is only seen
  when someone compares builds.
- **Trust fastembed's metadata check:** sizes and pointer IDs; offline, sizes only.
- **One combined digest per model:** shorter, but a mismatch would not say which
  file changed. It can be derived from the per-file hashes if needed.

## Consequences

- Builds are reproducible down to the model bytes, and every manifest row says
  exactly which files produced it.
- The first `docker compose up` still needs network. Later runs need none, because
  a verified cache short-circuits the hub.
- Upgrading fastembed can change which files a model needs. The pinned file list
  then no longer matches, and ingestion fails loudly instead of loading something
  unverified.
- **Phase 2:** the backend embeds queries with the dense model. It reuses the
  `infra/model_files.py` loader, loads from the same verified directory, and reports
  not ready when its digests differ from the live build's `models`. This is recorded
  here and in the Phase 1 plan, and not built in Phase 1.

## Corrections after implementation

### Pinned model files (Phase 1, commit 9b)

- **Files pinned per model:** fastembed's model file and its additional files are
  required; fastembed's common config and tokenizer files are pinned when the
  repository has them (the list is copied from fastembed 0.9.1 in
  `app/infra/model_pins.py`). First pins: 5 files for the dense model (about 66 MB),
  18 for BM25, all equal to the digests of the cache measured before this ADR.
- **Hub access:** always anonymous (`token=False`), always at an explicit revision;
  metadata requests time out after 30 s. A cache that does not match is downloaded
  again with `force_download`, because a plain download keeps a corrupted file that is
  already in the snapshot.
- **Logging:** the source logs one warning (model and file) when it downloads a bad
  cache again. `ModelIntegrityError` and `ModelDownloadError` carry the model, file and
  digests; the ingest CLI (commit 10) logs the one structured line and exits non-zero.
- **`make model-pins`** regenerates the models already pinned, or the ones named in
  `MODELS="name ..."`; it reads no settings and needs no database.
- **Offline load:** the models get `specific_model_path` and `local_files_only=True`.
  A `model` test makes every fastembed hub call fail and still embeds and counts
  tokens from the verified folder.
- **Secret scanning:** gitleaks' `generic-api-key` rule reads three digests in
  `models.toml` as tokens, because their keys are file names such as `tokenizer.json`.
  The root `.gitleaks.toml` keeps every default rule and exempts only that rule, only
  in `backend/models.toml`, and only for a value that is exactly a lowercase SHA-256
  digest (`condition = "AND"`). `targetRules` is required: in gitleaks 8.30.1 a global
  allowlist with `paths` skips the whole file, so `AND` would not apply (measured).
  `minVersion` makes an older gitleaks refuse the file instead of ignoring the
  exemption's scope. Proved by running gitleaks: the pins pass; fake keys added to the
  pins are still reported (two `generic-api-key`, one `aws-access-token`); digests on
  `token` and `key` lines in another file are still reported. The pre-commit hook,
  `make security` and CI (which runs `make security`) all read the root file; a unit
  test pins its content.

### Failed builds (Phase 1, commit 9c)

- **`index_builds.error_type`** (migration 0003, nullable): the exception type name of a
  failed build, never its message, as in a build warning. `IndexBuild` requires it for
  `failed` and rejects it for `succeeded`; the check `ck_index_builds_error_type_status`,
  `(status = 'failed') = (error_type IS NOT NULL)`, enforces the same rule in the
  database. The allowed status values stay with `ck_index_builds_status`, so each check
  tests one rule.
- `alembic check` compares columns, not check constraints; the Postgres integration
  tests prove the check, its upgrade and its downgrade.
