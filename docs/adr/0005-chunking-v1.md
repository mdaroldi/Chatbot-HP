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
carries the path. Kept lines before the first heading (cover and legal front matter)
belong to no section and are counted, not indexed (see Consequences).

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
- **Merging:** a section under `CHUNK_MIN_TOKENS` (50 for v1, re-measured with the
  real tokenizer) absorbs the next section with the same parent while it stays under
  the minimum and the result fits the limit. A merged chunk carries the **parent's
  path**, the deepest path true for all of its text, and each member's title as a
  heading line, so the citation never names a section that only part of the chunk
  belongs to. Top-level sections never merge.
- **Content type:** `regulatory` when a title in the path is about regulations,
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
- Front matter (cover, legal notices, and safety notices printed before the first
  bookmark) is not indexed in v1; whether to index it is an open decision.
- Unit tests count tokens with a word-counting fake; the real 450-token limit is
  checked by corpus tests once the tokenizer is available (Phase 1, commit 7).
- Table quality follows the row-band extractor: tables with spanning title rows yield
  fewer columns (ADR 0004).
