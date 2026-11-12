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
| 5 | `feat(ingestion): cleaning stage with drop statistics` | done (`app/ingestion/{cleaning,language}.py`; `pages.chapter_toc_lines` replaces `chapter_opener_pages`; continuation rule and opener lead-in as approved, see ADR 0004 corrections) |
| 6 | `feat(ingestion): sections and chunking v1` (with ADR 0005 and the CLAUDE.md chunking row) | done (`app/ingestion/{spans,blocks,sections,chunking}.py`; unit tests use a word-counting fake tokenizer) |
| 6b | `feat(ingestion): drop navigation and table residue` (section mini-TOCs, non-Latin table residue, URL-only sections; see "Commits 6b to 6d") | done |
| 6c | `feat(ingestion): index front-matter safety and notice content` (with the ADR 0005 update) | done |
| 6d | `feat(ingestion): treat question lead-ins of mini-TOCs as part of the list` (see "Commits 6b to 6d") | done |
| 7 | `feat(ingestion): fastembed embedder and token counter` (with the corpus test of the real 450-token limit, and `CHUNK_MIN_TOKENS` re-measured with the real tokenizer) | done (`app/infra/fastembed_{embedder,tokens}.py`; `CHUNK_MIN_TOKENS` 65, see ADR 0005 corrections) |
| 8 | `feat(ingestion): Qdrant versioned collection and alias swap` | done (`app/infra/qdrant_index.py`, `app/ingestion/publish.py`; `fix: anchor the data, model and cache ignore patterns` before it) |
| 9 | `feat(db): index_builds manifest (migration 0002)` (with ADR 0006) | done (`app/infra/{orm,manifest_repository}.py`, migration 0002; `ModelFiles`, `BuildWarning` and the new `IndexBuild` in `app/domain/ingestion.py`) |
| 9b | `feat(ingestion): pinned and verified model files` (ADR 0006: `backend/models.toml`, download of the pinned revision, SHA-256 of every file on every build, loading from the verified folder, the embedder and token counter switched to it, `make model-pins`, huggingface-hub as a direct dependency `<2.0`; the loader lives in `app/infra/` for Phase 2 reuse) | done (`backend/models.toml`, `app/infra/{model_files,model_pins}.py`; root `.gitleaks.toml` with one narrow exemption for the pinned digests; ADR 0006 corrections) |
| 9c | `feat(db): error_type for failed builds (migration 0003)` (nullable `index_builds.error_type`: the exception type name only, never its message; required for `failed`, absent for `succeeded`) | pending |
| 10 | `feat(ingestion): pipeline, CLI, migrate and ingest services, make ingest/inspect-chunks` (with the startup-order ADR and CLAUDE.md update, and the chunk-ID digest corpus test; the embedder is built only after the `--if-needed` check, and a unit test shows the skip path never constructs it; see "Decisions before commit 9") | pending |
| 11 | `ci: corpus job with cached download` (also runs the `model` tests, with the model cache restored) | pending |
| 12 | `docs: README ingestion section` (with a third-party licenses section that names tqdm, MPL-2.0, and a model section stating that the dense model files come from Qdrant's quantized ONNX build of bge-small, `Qdrant/bge-small-en-v1.5-onnx-Q`, ADR 0006) | pending |

Corpus tests land in the commit of the stage they test (item 10), not all at once. They carry
the `corpus` marker; `make test-int` runs them, and the CI integration job excludes them
(`-m "integration and not corpus"`) until commit 11 adds the cached corpus job. Tests of
the real embedding models (`tests/embeddings/`) carry the `model` marker, and so do corpus
tests that need a model. The CI expression already leaves them out, because they are not
marked `integration`; commit 11 caches `EMBEDDING_CACHE_DIR` and runs them.

## Decisions before commit 9

- **`INDEX_KEEP_PREVIOUS`** is a yes/no setting (`true` or `false`), default `true`.
- **Pruning failures (commit 10):** pruning runs only after a successful alias swap.
  If deleting an old build fails, the new build stays live: the failure is a
  structured warning (collection name and error type, no message text) recorded in
  the build's manifest row, and ingest still exits 0, so the backend, which waits for
  `ingest` to complete successfully, is never blocked by leftover collections. A unit
  test covers it.
- **Build ID:** the lowercase UTC timestamp of the build start, `%Y%m%dt%H%M%Sz`
  (for example `20261112t143005z`), which matches the existing build-id pattern.
- **Collection version:** the `v1` in `hpdocs_chunks_v1_<build id>` follows
  `chunking_version`, so a new chunking version gets a new prefix. Commit 10 updates
  `publish.py` and corrects ADR 0002, which still calls it the collection schema.
- **Local-mode warning filter:** the unit-test filter for qdrant-client's local-mode
  payload-index warning matches that one message only (category `UserWarning`), never
  a broader pattern.
- **Embedding model files:** ADR 0006 (accepted): pinned revision, SHA-256 of every
  file on every build, one `models` JSON column. A failed model check writes no
  `index_builds` row (it exits non-zero with its own error and one structured log
  line); `status` covers builds that started.

**Phase 2 note:** the backend embeds queries with the dense model. It reuses the
verified-model loader from `app/infra/` (commit 9b), loads from the same verified
folder, and reports not ready when its digests differ from the live build's `models`
(ADR 0006).

**Phase 5 note:** when reading retrieval quality, remember that the dense vectors come
from Qdrant's quantized ONNX build of bge-small (`Qdrant/bge-small-en-v1.5-onnx-Q`),
not the original weights; published bge-small scores are not a direct baseline.

**Phase 3 note:** chunk text contains `[icon]` where the manual shows a button glyph
(commit 4b). The Phase 3 system prompt must tell the model what `[icon]` means (an
unrecognised button or symbol) and that it must not guess which button it is.

## Commits 6b to 6d (decided after commit 6)

Inserted before commit 7. Each rule is test-first, with its own drop reason and
counter, and the corpus tests pin which PDF pages it touches; a rule that hits
anything beyond these pages is reported, not loosened.

**6b, three cleaning rules:**

1. **Section mini-TOCs** (`section_toc`): the `chapter_toc` idea for any section. A
   bullet list (consecutive bullet items in the section's content) is dropped when it
   has at least 2 items and **every** item's text equals the title of one of the
   section's own child bookmarks (compared like headings: case and whitespace
   ignored). As for chapter openers, a lead-in line just before the list that ends
   with a colon goes with it. A list with any non-matching item (a URL, a title with
   more words, a title wrapped onto a second line) is kept whole; no corpus item
   wraps. Measured: 19 lists with 94 items and 3 lead-ins, all in ENVY (the strict
   rule hits 20 sections; the 1-item list on PDF 40 is kept by the minimum of 2); 0
   in OMEN. Chapter-opener lists stay `chapter_toc`, which comes first.
2. **Non-Latin table residue** (`non_latin_table`): a ruled table that lost at least
   one line as `non_latin` is dropped when none of its kept lines holds a Latin word,
   a word being at least two consecutive lowercase Latin letters (so element symbols
   such as Pb, acronyms and marks do not count). Tables without `non_latin` lines are
   never candidates, so tables of numbers or part numbers are untouched. Measured:
   ENVY PDF 130 only (78 lines); the PDF 129 table keeps English cells and stays.
3. **URL-only sections:** URLs are not text for the `empty_after_cleaning` check, so a
   section whose only content is a URL is dropped and counted there. Measured: ENVY
   PDF 133 only.

Sections emptied by these rules are dropped as `empty_after_cleaning` (ENVY 21 → 32:
8 left with only a mini-TOC, the PDF 130 table section, the PDF 133 URL section);
their subsections keep the titles in their paths. OMEN is unchanged.

**6c, front matter:** the headed safety and notice parts of the front matter (before
the first bookmark) are indexed under `Front matter > <heading>`, with `has_warning`
when they hold a WARNING or CAUTION. Headings are found by font size, not by a keyword
list, with one pinned exception: ENVY sets its safety heading in body size, so a
short line (at most 5 words, no end punctuation) with the word `safety` is a heading
too. The cover, copyright and trademark text stays out of the index. New content type
`front_matter`, checked before the title rules. Measured: ENVY 1 chunk (PDF 2), OMEN
2 (PDF 3, 4). ADR 0005 is updated in the same commit.

**6d, question lead-ins:** the `section_toc` lead-in may also end with `?`, so a
question just before a dropped mini-TOC goes with it (a colon lead-in stays as it is;
chapter openers keep the colon-only rule). Measured: ENVY PDF 95 and 100 only, one line
each; both sections are left empty and dropped as `empty_after_cleaning` (ENVY 32 →
34), so ENVY loses two chunks that held only the question. OMEN is unchanged.

**Plan addition for commit 10, chunk-ID digest:** a corpus test pins a digest of the
sorted chunk IDs per document, counted with the real tokenizer (commit 7). Any change
in cleaning or chunking output fails it and forces a `chunking_version` bump, because
`--if-needed` compares the version, not the code.

**Deferred to the Phase 5 ablation:** merging a parent's one-line introduction into
its first child section.

## Dependencies (licenses checked)

| Package | Version | License | Notes |
|---|---|---|---|
| pdfplumber | 0.11.10 | MIT | with pdfminer.six 20260107 (MIT), Pillow 13.0.0 (MIT-CMU) |
| pypdfium2 | 5.15.0 | BSD-3-Clause / Apache-2.0 | installed with pdfplumber, unused for text; 5.15.1 and 5.15.2 too new, pinned by a commented `constraint-dependencies` entry in `backend/pyproject.toml` |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | via pdfminer.six (encrypted PDFs) |
| cffi | 2.1.1 | MIT-0 | via cryptography |
| pycparser | 3.12 | BSD-3-Clause | via cffi |
| fastembed | 0.9.1 | Apache-2.0 | adopted 2026-11-12 (released 2026-10-24); new at runtime with it: onnxruntime 1.31.1 (MIT), tokenizers 0.23.3 (Apache-2.0), huggingface-hub 1.33.0 (Apache-2.0; fastembed caps it below 2.0), hf-xet 1.8.0 (Apache-2.0), numpy 2.5.4 (BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0), py-rust-stemmers 0.1.8 (MIT, from the repository; no license in its metadata), mmh3 5.3.1 (MIT), flatbuffers 25.12.19 (Apache-2.0), protobuf 7.36.2 (BSD-3), fsspec 2026.9.0 (BSD-3), requests 2.34.2 (Apache-2.0), urllib3 2.8.0 (MIT), packaging 26.3 (Apache-2.0 OR BSD-2), pyyaml 6.0.3 (MIT), colorama 0.4.6 (BSD-3), win32-setctime 1.2.1 (MIT, Windows only) |
| loguru | 0.7.3 | MIT | pulled by fastembed; 0.8.0 too new, pinned by a commented `constraint-dependencies` entry (fastembed also caps it below 0.8) |
| filelock | 4.2.9 | MIT | pulled by huggingface-hub (already a dev dependency); 4.2.10 (2026-11-09, bug fix, no advisory) too new, pinned by a commented `constraint-dependencies` entry |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | via fastembed; file-level copyleft, used unmodified; flagged to the user |
| qdrant-client | 1.20.0 | Apache-2.0 | adopted 2026-11-12 (released 2026-10-22); new at runtime with it: grpcio 1.85.1 (Apache-2.0), portalocker 4.4.2 (BSD-3), h2 4.4.1, hpack 4.2.0, hyperframe 6.1.0 (MIT); protobuf came with fastembed. Later grpcio uploads (1.78.2, 1.80.1, 1.81.2) are backports to older lines; the client talks REST, not gRPC |
| huggingface-hub | 1.33.0, declared directly (`>=1.33.0,<2.0`) | Apache-2.0 | already locked through fastembed; no new package. Pinned model downloads (commit 9b) |
| aiosqlite | 0.22.1 (dev group only) | MIT | manifest repository unit tests; released 2025-12-23, no runtime dependencies |

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
   - Continuation rule (as approved at commit 5): a line the main rule kept is dropped
     when the line above it in the **same block** (same page, column and font size, at
     most 1.5 em apart) was dropped as non-English and its English score is below 2.
     The tails end their paragraphs, often before a heading, so "between two dropped
     lines" missed two of the three. A unit test keeps an English line that follows a
     dropped line in the same block.
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
- Sections under `CHUNK_MIN_TOKENS` (65 tokens, re-measured at commit 7) merge into the next section under the same parent;
  the merged chunk carries the parent's path and each member's title (ADR 0005).
- A chunk holds at most one table; a word or row that fits no chunk raises
  `ChunkTooLargeError` instead of overflowing.
- The chunker takes the limits as arguments; the `CHUNK_*` settings are wired in commit 10.
- The corpus test of the real 450-token limit landed in commit 7 with the tokenizer.
- WARNING/CAUTION stays with its step and sets `has_warning`.
- Payload: doc_id, product, section_path, chapter, pdf_page_start/end,
  printed_page_start/end, content_type (procedure|table|reference|specs|parts|front_matter|
  regulatory), has_warning, chunking_version, token_count, text.
- `chunk_id = uuid5(namespace, doc_id | section_path | index | sha256(text))`:
  re-running yields the same ids.
- The `Chunker` Protocol leaves room for a naive baseline in Phase 5.

## Index and manifest

- Qdrant collection `hpdocs_chunks_v1_<build>`: named vectors `dense`
  (BAAI/bge-small-en-v1.5, CPU) and `sparse` (BM25, IDF modifier), payload indexes on
  `doc_id` and `content_type`; build, verify, switch the alias, keep the previous
  collection for rollback (`INDEX_KEEP_PREVIOUS=true`), prune older ones.
- Migration 0002: `index_builds` (build_id, created_at, status, collection name
  (unique), document hashes, chunking_version, dense and sparse model names, `models`
  with revisions and file digests, total `chunk_count` and per-document `chunk_counts`,
  drop stats, `warnings`), all JSON columns with SQLAlchemy's generic `JSON` type.
- New settings, each documented in `.env.example` (the parity test must pass):
  `CORPUS_DIR`, `CORPUS_MAX_BYTES`, `CORPUS_DOWNLOAD_TIMEOUT_SECONDS`,
  `EMBEDDING_DENSE_MODEL`, `EMBEDDING_SPARSE_MODEL`, `EMBEDDING_CACHE_DIR`,
  `EMBEDDING_BATCH_SIZE`, `CHUNK_MAX_TOKENS=450`, `CHUNK_MIN_TOKENS`,
  `QDRANT_COLLECTION_ALIAS`, `INDEX_KEEP_PREVIOUS=true` (yes/no).
- `make inspect-chunks` writes a git-ignored `ingest-review/chunks.md`; `.gitignore`
  also covers `corpus/`, `ingest-review/` and the model cache.

## Corpus tests (marked, on the real PDFs)

No chunk over the limit; every chunk has a section path and both page ranges; no TOC
or index chunks; no "ENWW"; no non-Latin characters; spare part N96208-001 stays next
to "Bottom cover"; the 8 error-code rows (E0 to E6 and E9) on PDF page 105 keep code and description in the same row; regulatory model number
SNPRC-1807-01 present; the Wi-Fi Direct five-device limit present; a tampered file
fails the checksum check.

## Open points to report

- Qdrant local mode: aliases and the IDF modifier work (commit 8). Payload indexes have no
  effect there and warn; unit tests filter that one warning, integration tests check the
  payload schema on the server.
- Qdrant server/client compatibility (item 7): server image v1.20.0, client 1.20.0; an
  integration test pins both, and the client's compatibility check raises no warning.
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
