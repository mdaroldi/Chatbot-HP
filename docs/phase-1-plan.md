# Phase 1 plan: ingestion

Approved plan for Phase 1 (download, verify, parse, clean, chunk, embed, upsert,
manifest). No retrieval, chat or LLM calls. Read [CLAUDE.md](../CLAUDE.md) and the
ADRs in [docs/adr](adr/) first; [ADR 0004](adr/0004-pdf-parsing.md) records the
parsing spike. Page numbers below are **PDF page numbers** unless marked "printed".

## Working rules for this phase

- Strict TDD. For each commit, run the new tests against a behavior-free skeleton
  and report how many passed and why (expected: zero); tighten vacuous tests.
- Report to the user after every commit and wait for approval before the next one.
- Every fix is its own `fix:` commit. Ask before testing an uncovered branch whether
  the branch is needed. No coverage exclusions.
- No HP text in ADRs, README, tests, commit messages or this file beyond short
  identifiers (part numbers, model numbers, labels). Full chunks appear only in
  reports to the user. Never commit PDFs or extracted text.
- Check the license of every new dependency; do not adopt a release in its first
  5 days except for security fixes. Pin through the lock files.
- Do not push; the user pushes.

## Commits

| # | Commit | Status |
|---|---|---|
| 1 | `docs: ADR 0004 PDF parsing` | done (4e3bfef) |
| 2 | `feat(ingestion): domain IR and protocols` | done (`app/domain/ingestion.py`) |
| 3 | `feat(ingestion): corpus manifest and verified downloader` | done (`backend/corpus.toml`, `app/infra/corpus_http.py`); `fix: anchor the corpus cache ignore pattern` (da7cf4b) |
| 4 | `feat(ingestion): pdfplumber parser with footer, bookmark and row-band tables` | done (`app/infra/pdfplumber_parser.py`, `app/ingestion/{lines,footer,outline,pages,tables}.py`) |
| 4b | `feat(ingestion): [icon] placeholders for inline images` (image boxes in the parser output, `[icon]` inserted during line grouping, lost-icon count in the stats: measured 87 ENVY, 3 OMEN; the spike's 43 was not reproducible, see ADR 0004 corrections) | done |
| 5 | `feat(ingestion): cleaning stage with drop statistics` | next |
| 6 | `feat(ingestion): sections and chunking v1` (with ADR 0005 and the CLAUDE.md chunking row) | pending |
| 7 | `feat(ingestion): fastembed embedder and token counter` | pending |
| 8 | `feat(ingestion): Qdrant versioned collection and alias swap` | pending |
| 9 | `feat(db): index_builds manifest (migration 0002)` | pending |
| 10 | `feat(ingestion): pipeline, CLI, migrate and ingest services, make ingest/inspect-chunks` (with the startup-order ADR and CLAUDE.md update) | pending |
| 11 | `ci: corpus job with cached download` | pending |
| 12 | `docs: README ingestion section` | pending |

Corpus tests land in the commit of the stage they test (item 10), not all at once. They carry
the `corpus` marker; `make test-int` runs them, and the CI integration job excludes them
(`-m "integration and not corpus"`) until commit 11 adds the cached corpus job.

**Phase 3 note:** chunk text contains `[icon]` where the manual shows a button glyph
(commit 4b). The Phase 3 system prompt must tell the model what `[icon]` means (an
unrecognised button or symbol) and that it must not guess which button it is.

## Dependencies (licenses checked)

| Package | Version | License | Notes |
|---|---|---|---|
| pdfplumber | 0.11.10 | MIT | with pdfminer.six 20260107 (MIT), Pillow 13.0.0 (MIT-CMU) |
| pypdfium2 | 5.15.0 | BSD-3-Clause / Apache-2.0 | installed with pdfplumber, unused for text; 5.15.1 and 5.15.2 too new, pinned by a commented `constraint-dependencies` entry in `backend/pyproject.toml` |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | via pdfminer.six (encrypted PDFs) |
| cffi | 2.1.1 | MIT-0 | via cryptography |
| pycparser | 3.12 | BSD-3-Clause | via cffi |
| fastembed | 0.9.1 | Apache-2.0 | onnxruntime (MIT), tokenizers (Apache-2.0), huggingface-hub (Apache-2.0), numpy (BSD-3), py-rust-stemmers (MIT), mmh3 (MIT) |
| loguru | 0.7.3 | MIT | pulled by fastembed; 0.8.0 too new |
| tqdm | via fastembed | MPL-2.0 AND MIT | file-level copyleft, used unmodified; flagged to the user |
| qdrant-client | 1.20.0 | Apache-2.0 | grpcio (Apache-2.0), protobuf (BSD-3), portalocker (BSD-3) |
| aiosqlite | dev group only | MIT | manifest repository unit tests |

Re-check ages and licenses when adding each one; pick the newest release older
than 5 days.

## Corpus

`backend/corpus.toml`: doc_id, title, product, URL, pinned SHA-256 (computed from the
local PDFs, verified against the download). Both URLs answered 200 without redirects
and with sizes equal to the local files during the spike.

| doc_id | URL | SHA-256 |
|---|---|---|
| ENVY 6000 User Guide | https://h10032.www1.hp.com/ctg/Manual/c06584210.pdf | d7f346897f0660b78dea88719a76267c604cbb2ee81de686ceabce56ab7a34d1 |
| OMEN 17.3 Service Guide | https://kaas.hpcloud.hp.com/pdf-public/pdf_10301542_en-US-1.pdf | e19b003d287283dd409271f72da5ccfa28fd34247a616628a65e198e469b5356 |

Downloader: allowlisted URLs only, HTTPS, same-host redirects only, size cap,
`%PDF-` magic check, timeouts, `CORPUS_DIR` cache volume for offline runs. A checksum
mismatch fails with a clear message, distinct from a download failure.

## Design

The user's design, with accepted changes 1 to 7:

1. Tables use a **custom row-band extractor** (horizontal rules split rows, header-word
   gaps give column starts, a row is never split from its cells), not pdfplumber's
   table finders.
2. Drop reasons: `toc`, `index`, `chapter_toc` (chapter-opener section lists),
   `footer`, `non_latin`, `non_english_latin`, `empty_after_cleaning`. Non-Latin is
   per line: mostly CJK, Hangul, Thai, Cyrillic, kana, Arabic or Hebrew. Greek is kept
   (unit symbols).
3. The **printed page** comes from each page's footer number (null for unnumbered
   front matter), not from a fixed offset.
4. **Regulatory** content is identified by bookmark path and tagged
   `content_type=regulatory`, not dropped.
5. Ingestion lives in the backend project (`app/ingestion`) and the **backend image**
   (`python -m app.ingestion`). Trade-off and way out recorded in ADR 0004.
6. Sparse vectors use fastembed `Qdrant/bm25`; the collection's sparse vector uses
   `modifier=IDF`.
7. Corpus tests run under `make test-int` and in a **required** CI corpus job with a
   cached download; never in `make check`. A download failure and a failed assertion
   produce clearly different messages. If HP blocks the CI runner, tell the user and
   propose a documented local-only job.

Layers (dependency rule unchanged):

- `domain/ingestion.py`: IR dataclasses (page lines with font info → blocks →
  sections → chunks), the `ContentType` and drop-reason enums, and the Protocols
  `CorpusSource`, `DocumentParser`, `Chunker`, `TokenCounter`, `Embedder` (dense and
  sparse), `VectorIndex`, `ManifestStore`. No I/O.
- `ingestion/`: pure stages over plain data: footer detection, bookmark-to-section
  mapping, heading matching, row-band tables, cleaning with counters, sections,
  chunking v1, chunk ids, the pipeline with stats, and `__main__.py` (CLI).
- `infra/`: `corpus_http.py`, `pdfplumber_parser.py` (thin adapter),
  `fastembed_embedder.py` and the token counter (CPU work in a thread pool),
  `qdrant_index.py`, `manifest_repository.py`.

## The user's items 1 to 10, as revised

1. **`docker compose up` ends with a searchable index.**
   - Startup order: `postgres` and `qdrant` healthy → **`migrate`** (one-shot, backend
     image, `alembic upgrade head`) → **`ingest`** (one-shot, `python -m
     app.ingestion --if-needed`) → **`backend`** (`service_completed_successfully` on
     both; its entrypoint no longer runs migrations) → `frontend`.
   - One-shot services set `restart: "no"` (the shared hardening uses
     `unless-stopped`) and disable the image healthcheck. `ingest` is on both
     networks, non-root, read_only, cap_drop, tmpfs, with `corpus-cache` and
     `model-cache` volumes. The `tools` profile is gone.
   - `--if-needed` skips the build (exit 0, no download, no model load) only when the
     live alias's latest successful manifest matches **all** of: document hashes
     (vs `corpus.toml`), chunking version, dense model, sparse model, and the
     collection's point count equals the manifest's chunk count. **Each condition is
     unit-tested separately; each mismatch triggers a rebuild.**
   - `make ingest` stays as the forced rebuild (`--force`).
   - Trade-off accepted: the first `up` needs network and takes a few minutes; if a
     download fails, the backend does not start. Later runs need no network.
     Document both in the README with the **measured** first-up time.
   - Write an ADR for the startup order and the `--if-needed` rule; update CLAUDE.md
     (no `tools` profile; migrations not in the backend entrypoint) in the same commit.
2. **Coverage without exclusions.** Footer, bookmark, heading and row-band logic are
   pure functions in `ingestion/`, unit-tested with synthetic input. Per adapter:
   - `corpus_http.py`: httpx `MockTransport` + `tmp_path`.
   - `pdfplumber_parser.py`: a synthetic PDF built by a **test helper under
     `tests/` only** that computes xref offsets (two pages, chosen font sizes, a
     horizontal rule, a two-entry outline, a small inline image). No HP text.
   - Embedder and token counter: model classes injected (fastembed classes as
     defaults); fakes in unit tests; real models in integration/corpus tests.
   - `qdrant_index.py`: qdrant-client local mode (`location=":memory:"`). **If local
     mode lacks aliases or the IDF modifier, stop and tell the user before choosing
     a fallback.** Integration tests repeat on the real server.
   - `manifest_repository.py`: unit tests on SQLite via **aiosqlite (dev group
     only)**; the table uses SQLAlchemy's generic `JSON` type. The Postgres
     integration test remains the source of truth for migration 0002.
3. **Non-English text in Latin script** (`non_english_latin`).
   - Spike evidence, PDF pages 121-132 of ENVY: 35 lines counted non-English by a
     loose classifier (French 19, Portuguese 7, Spanish 4, German 3, 2 more French
     first counted as Dutch). PDF page 133 holds about 9 Portuguese lines. A strict
     rule hit 33 lines, only on PDF pages 122-124 and 133, and **0 lines on every other
     page of both manuals**.
   - Rule: remove URLs, then score each line with built-in function-word lists
     (English, German, French, Spanish, Portuguese, Italian, Dutch). Drop when the line
     has **≥ 3 words**, its best non-English score is **≥ 2**, and that score is
     **greater than the English score**.
   - Continuation rule: a **short** line in the **same block**, between two dropped
     lines and scoring below 2, is dropped too. A unit test keeps an English line that
     sits between two dropped lines.
   - Lines the strict rule kept on PDF pages 121-132 (10): 5 are `hp.com` URLs
     (correctly kept); 3 are short continuation lines of German/French paragraphs
     (PDF pages 122, 123, 124), which the continuation rule must catch; 2 are real
     non-English lines with too few listed words (a French line on PDF page 124 with
     elisions such as `d'`, a Portuguese line without accents on PDF page 132).
   - Word-list fix to implement test-first: count French elisions (`l'`, `d'`, `qu'`,
     `n'`) as French; add Portuguese `deste`, `desta`, `foi`, `nao`, `pelo`, `pela`.
     Re-run the precision check on both manuals: still zero hits outside the
     back-matter pages, or report it instead of loosening the threshold.
   - **At the cleaning commit, show the user the 10 lines, the word-list fix and the
     precision re-check.** The spike scripts were not kept; recreate the check.
   - After cleaning, a section with no substantive text left (for example only a
     heading) is dropped as `empty_after_cleaning` and counted.
   - **`[icon]` is not text content** (commit 4b): the cleaning rules strip it before
     scoring a line's language (it never counts as a word for `non_english_latin` or
     as a script for `non_latin`) and before the `empty_after_cleaning` check, so a
     section whose only content is a heading and `[icon]` placeholders is still
     dropped as empty. Commit 5 adds unit tests for both.
   - The stats table reports, per document, how many lines each rule dropped.
4. Every chunk stores **both** the PDF page range and the printed page range; every
   report says which numbering it uses.
5. Before the alias swap, verify the new collection's point count equals the chunk
   count. On any failure, leave the alias untouched and delete the partial
   collection. The manifest has a `status` column (or records only successes).
6. In the same commit as ADR 0005, change the CLAUDE.md chunking row to 450 tokens,
   with bge-small's 512-token limit as the reason. ADR 0005 also states how long
   prose sections are split and which section path a merged chunk carries.
7. Confirm the pinned Qdrant server image (v1.20.0) is compatible with
   qdrant-client 1.20.0.
8. Icons: if the parser sees the inline image position, insert an `[icon]`
   placeholder at the gap; if that costs more than a few lines, keep the documented
   limitation and say so.
9. No HP text in ADRs, README, tests or commit messages beyond short identifiers.
10. Corpus tests land in the commit of the stage they test.

## Chunking v1 (ADR 0005)

- Section-based; at most **450 tokens including** the "doc > chapter > section" prefix,
  counted with the bge-small tokenizer.
- Procedures split at step boundaries, repeating the section title with "(continued)".
- One table per chunk; when split, the header row is repeated and rows are never cut.
- Sections under `CHUNK_MIN_TOKENS` merge into the next section under the same parent.
- WARNING/CAUTION stays with its step and sets `has_warning`.
- Payload: doc_id, product, section_path, chapter, pdf_page_start/end,
  printed_page_start/end, content_type (procedure|table|reference|specs|parts|
  regulatory), has_warning, chunking_version, token_count, text.
- `chunk_id = uuid5(namespace, doc_id | section_path | index | sha256(text))`:
  re-running yields the same ids.
- The `Chunker` Protocol leaves room for a naive baseline in Phase 5.

## Index and manifest

- Qdrant collection `hpdocs_chunks_v1_<build>`: named vectors `dense`
  (BAAI/bge-small-en-v1.5, CPU) and `sparse` (BM25, IDF modifier), payload indexes on
  `doc_id` and `content_type`; build, verify, switch the alias, keep the previous
  collection for rollback (`INDEX_KEEP_PREVIOUS=1`), prune older ones.
- Migration 0002: `index_builds` (build_id, created_at, status, document hashes,
  chunking_version, dense and sparse model, counts, drop stats, collection name).
- New settings, each documented in `.env.example` (the parity test must pass):
  `CORPUS_DIR`, `CORPUS_MAX_BYTES`, `CORPUS_DOWNLOAD_TIMEOUT_SECONDS`,
  `EMBEDDING_DENSE_MODEL`, `EMBEDDING_SPARSE_MODEL`, `EMBEDDING_CACHE_DIR`,
  `EMBEDDING_BATCH_SIZE`, `CHUNK_MAX_TOKENS=450`, `CHUNK_MIN_TOKENS`,
  `QDRANT_COLLECTION_ALIAS`, `INDEX_KEEP_PREVIOUS=1`.
- `make inspect-chunks` writes a git-ignored `ingest-review/chunks.md`; `.gitignore`
  also covers `corpus/`, `ingest-review/` and the model cache.

## Corpus tests (marked, on the real PDFs)

No chunk over the limit; every chunk has a section path and both page ranges; no TOC
or index chunks; no "ENWW"; no non-Latin characters; spare part N96208-001 stays next
to "Bottom cover"; the 8 error-code rows (E0 to E6 and E9) on PDF page 105 keep code and description in the same row; regulatory model number
SNPRC-1807-01 present; the Wi-Fi Direct five-device limit present; a tampered file
fails the checksum check.

## Open points to report

- Qdrant local mode: aliases and IDF modifier (stop if missing).
- Qdrant server/client compatibility (item 7).
- Icon placeholder feasibility (item 8) and the lost-icon passage count.
- Measured first `docker compose up` time, and the backend image size increase.
- Whether HP blocks the GitHub runners (CI corpus job).

## Acceptance evidence

- `make check`; `make test-int` including corpus tests.
- From a fresh clone: `docker compose up` ends with a searchable index; a second
  `docker compose up` skips the build; `make ingest` rebuilds without duplicates
  (same ids, new collection, alias moved).
- Stats table per document: pages parsed and dropped by reason, lines dropped per
  rule, chunk count, tokens min/median/max, content types.
- Points per document in Qdrant.
- 15 representative chunks printed in full **in the report only** (a procedure, a
  table, the error codes, a parts table, a warning, a regulatory chunk, a
  button-glyph passage, and others).
- The count of lost-icon passages.
- The CLAUDE.md phase table update, in the final report.
