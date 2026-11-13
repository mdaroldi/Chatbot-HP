# ADR 0005: Structure-aware chunking v1

- **Status:** accepted (Phase 1, commit 6)
- **Date:** 2026-11-12

## Context

Retrieval quality depends on chunks that keep one procedure, table or topic
together and say where they come from. The dense model, BAAI/bge-small-en-v1.5,
truncates input at 512 tokens, so a chunk that exceeds the limit loses its end without
any error. Both manuals have a complete bookmark tree (ADR 0004), so structure is
available for every page after the front matter.

## Decision

**Sections.** One section per bookmark whose heading survived cleaning, holding the
lines up to the next heading. The section path comes from the bookmark tree, not
from heading lines: a section keeps the titles of ancestors whose heading line was
dropped as `empty_after_cleaning`. Heading lines are not content; the chunk prefix
carries the path. Lines before the first bookmarked heading form **front-matter
sections** when they have a heading of their own; the rest (cover, copyright,
trademarks) is counted, not indexed (see Corrections after implementation).

**Blocks.** Lines become paragraphs, steps (`1.`, `a.` hanging markers), bullets
(`●`, `•`, en dash), notes (`WARNING!`, `CAUTION:`, `IMPORTANT:`, `NOTE:`, `TIP:`) and
tables (one header block plus one block per row, from the row-band extractor). Lines
join a block when they are on the same page, at most 1.5 em apart, overlapping
horizontally, and in the same font size. A block records the x position of its
first line (`indent`).

**Chunks.** At most **450 tokens including the prefix**, counted with the bge-small
tokenizer, which leaves margin under the 512-token limit for the tokenizer's special
tokens and for counting differences.

- Text: `document title > chapter > ... > section`, a blank line, then one block per
  line. Chunks after the first of a section append ` (continued)` to the prefix.
- Units that are never split while they fit: a step with the blocks indented under
  it; a hazard note at the step's own level, which travels with the next unit;
  a paragraph, bullet or note; a whole table with its caption (`Table N-N ...`).
  WARNING and CAUTION therefore stay with their step and set `has_warning`.
- A chunk holds **at most one table**, next to prose when the whole table fits. A
  longer table is split between rows, repeating its caption and header row; a row is
  never cut.
- **Long prose** that does not fit one chunk is split between blocks, then between
  sentences, then between words, always filling the chunk greedily.
- A word or a table row that fits no chunk raises `ChunkTooLargeError`, naming the
  document, section path and token count: nothing is silently cut or left over the
  limit.
- **Merging:** a section under `CHUNK_MIN_TOKENS` (65 tokens for v1; see Corrections
  after implementation) absorbs the next section with the same parent while it stays under
  the minimum and the result fits the limit. A merged chunk carries the **parent's
  path**, the deepest path true for all of its text, and each member's title as a
  heading line, so the citation never names a section that only part of the chunk
  belongs to. Top-level sections never merge.
- **Content type:** `front_matter` for chunks under `Front matter`; else
  `regulatory` when a title in the path is about regulations,
  notices, environmental programs, recycling or power cords; else `specs` when a
  title names specifications; else `parts` for a table with spare-part numbers
  (`N12345-001` pattern), `table` for another table; else `procedure` when it has
  steps; else `reference`.
- **Identity:** `chunk_id = uuid5(namespace, "doc_id|section path|index|sha256(text)")`,
  with a fixed namespace and the chunk's index within its section, so a rebuild of
  the same input yields the same ids.
- Every chunk stores the PDF and the printed page ranges; printed pages are null
  for unnumbered and roman-numbered pages.

The limits come from `CHUNK_MAX_TOKENS` and `CHUNK_MIN_TOKENS` when the pipeline is
wired (Phase 1 plan, commit 10); the chunker takes them as arguments.

## Alternatives considered

- **Fixed-size or recursive character splitting:** simple, but cuts steps from
  their warnings and tables from their headers, and has no section to cite.
- **Semantic (embedding or LLM) chunking:** slower, non-deterministic, and still
  needs the structure for citations.
- **Merged chunks citing the first or the next section:** shorter citations, but
  wrong for part of the text.

## Consequences

- Chunks are deterministic and traceable to a bookmark path and both page ranges.
- The `Chunker` protocol leaves room for a naive baseline in the Phase 5 ablation.
- Front-matter safety and notice text is indexed; the cover, copyright and trademark
  text before the first bookmark is not.
- Unit tests count tokens with a word-counting fake; the real 450-token limit is
  checked by corpus tests once the tokenizer is available (Phase 1, commit 7).
- Table quality follows the row-band extractor: tables with spanning title rows yield
  fewer columns (ADR 0004).

## Corrections after implementation

### Front matter (Phase 1, commit 6c)

Both manuals print safety and notice text before their first bookmark (ENVY PDF 2,
OMEN PDF 3-4), which v1 first left out of the index. PDF page numbers.

- **Sections:** a front-matter heading starts a section that runs to the next one or
  to the first bookmarked heading. The path is `Front matter > <heading>`. Kept lines
  before the first front-matter heading are not indexed, and neither is a heading
  with no content. Front-matter sections come before the bookmarked ones and merge
  under the usual rule (their parent is `Front matter`).
- **Headings by layout:** a line at heading size (12 pt or more, as for bookmarked
  headings) on a page after the cover (PDF 1). OMEN sets its two notice headings at
  14 pt over a 10 pt body.
- **Keyword exception (deliberate):** ENVY sets its safety heading in the same size,
  font and colour as its body (8 pt), so no layout rule finds it. A line with at most
  5 words, no sentence punctuation at its end, and the word `safety` is therefore a
  heading too. The keyword list is exactly `{safety}`, pinned by unit tests; it does
  not match the copyright heading, which stays out with the trademark text. A new
  keyword needs a new decision.
- **Content type `front_matter`**, checked before the title rules: the notice
  headings would otherwise read as `regulatory`. `has_warning` keeps its meaning (a
  WARNING or CAUTION note): OMEN's heat notice sets it; ENVY's safety list, which
  has no hazard label, does not.
- **Pages:** front-matter pages are unnumbered or roman-numbered, so their printed
  page is null and citations use the PDF page.
- Measured: ENVY 1 chunk (PDF 2), OMEN 2 chunks (PDF 3 and 4); not indexed: ENVY 15
  and OMEN 83 lines on PDF 1-2.

### Real tokenizer (Phase 1, commit 7)

- **Counter:** `FastembedTokenCounter` counts with the dense model's own tokenizer
  (fastembed loads it without the ONNX session). Counts include `[CLS]` and `[SEP]`, so
  the limit is a little stricter than the text alone. They saturate at the model's
  512-token input: a longer text counts as 512, which is still over the limit.
  `[icon]` costs 3 tokens.
- **Model build:** fastembed serves `BAAI/bge-small-en-v1.5` from Qdrant's quantized
  ONNX export (`Qdrant/bge-small-en-v1.5-onnx-Q`), with the original tokenizer.
- **Limit holds:** with 450 and the real counts, no chunk of either manual exceeds the
  limit and `ChunkTooLargeError` never fires (largest: ENVY 442, OMEN 450). The
  word-counting stand-in had let chunks reach the 512 cap.
- **Merge floor re-measured:** the earlier floor of 50 counted words. Text runs at
  about 1.27 tokens per word (median over all sections). 65 tokens is the floor that
  best reproduces the 50-word merges: 25 mergeable sections fall under each floor,
  and the two sets differ by one section each way.
- Measured with 450 and 65: ENVY 149 chunks, OMEN 141.

### Settings wired (Phase 1, commit 10)

- `CHUNK_MAX_TOKENS` (default 450, at most 512) and `CHUNK_MIN_TOKENS` (default 65,
  below the maximum) reach the chunker through the ingest CLI. A corpus test pins a
  digest of the sorted chunk IDs per document with these values; any change in
  cleaning or chunking output fails it and needs a `chunking_version` bump, because
  `--if-needed` compares the version, not the code.

### Tables across a heading (Phase 1, fix after commit 10)

- **Defect:** a table band counted as a row when its dominant font size (by
  characters) was table size. On OMEN PDF 16, 18 and 27 two tables of the same width
  sit on one page with a figure between them, and the band from the first table's
  last rule to the second's header holds the second's heading or intro (10 pt and up)
  and its caption (9 pt). The caption is longer, so the band was a row and the two
  tables became one across the section boundary. Blocks emit a table when any of its
  lines is in the section, so the whole table went into both sections: 3 duplicated
  table chunks, and one `[icon]` counted twice in chunk text (4 instead of 3).
- **Fix:** a band is a row only if **every** line in it is at table size (ADR 0004).
  Measured: only those three OMEN bands change; ENVY's tables are identical.
- **`chunking_version` is now `v2`**: the chunk output changed, so `--if-needed`
  rebuilds an index made with `v1`, and new collections are named
  `hpdocs_chunks_v2_<build id>`. The design is still the v1 design of this ADR.
  Corpus tests pin that each table row lands in one section and each kept `[icon]` in
  one chunk.
- Measured with 450 and 65: ENVY 149 chunks (digest unchanged), OMEN 138.
