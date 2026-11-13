# ADR 0004: PDF parsing with pdfplumber

- **Status:** accepted (Phase 1, after a spike on 2026-11-12)
- **Date:** 2026-11-12

## Context

Ingestion must turn two HP manuals into clean sections. A pdfplumber 0.11.10 spike
found (PDF page numbers):

| Aspect | ENVY 6000 User Guide (135 pages) | OMEN 17.3 Service Guide (91 pages) |
|---|---|---|
| Bookmarks | 169, levels 1-4, all resolve to a page | 113, levels 1-5, all resolve |
| Headings | one font family; 18/14/12 pt by level, body 10 pt | same scheme |
| Footer | bottom ≈40 pt: page number (alternating sides), chapter, "ENWW"; no header | same, without "ENWW" |
| Printed page | PDF − 6 on numbered pages; front matter unnumbered | PDF − 7 |
| TOC / index | 3-5 / 134; chapter openers list their sections | 5-7 / 89 |
| Tables | horizontal rules only | ruled; spare-part rows |
| Other scripts | CJK, Hangul, Thai, Cyrillic in whole lines on 121-132; French, German, Spanish, Portuguese lines on 122-124 and 133 | 4 Greek unit symbols only |
| Steps, notes | numbered and lettered steps; WARNING!, CAUTION:, NOTE:, TIP: | same, plus IMPORTANT: |
| Icons | inline images; the text keeps a gap where the button was (43 passages) | 3 passages |

No private-use characters or `(cid:N)` markers. pdfplumber's table finders fail on
both table styles: no table on rule-only ENVY pages, words split by the "text"
strategy, and a single column for OMEN's parts table.

## Decision

- **Parser:** pdfplumber (MIT) on pdfminer.six (MIT); pypdfium2 (BSD-3/Apache-2.0)
  is installed with it but not used for text. PyMuPDF is excluded (AGPL).
- **Structure from bookmarks:** the outline gives the section tree; a heading is the
  first line at or after its target page that matches the title at 12 pt or more.
- **Footer:** the bottom zone is dropped; its number becomes the printed page
  (none for unnumbered front matter).
- **Drops, each counted by reason:** TOC and index pages, chapter-opener section
  lists, footers, lines mostly in non-Latin scripts (Greek kept), non-English lines
  in Latin script (function-word scores), and sections left empty by cleaning.
- **Tables:** a row-band extractor: horizontal rules split rows, header-word gaps give
  column starts, and a row is never separated from its cells.
- **Pure logic, thin adapter:** footer, bookmark, heading and table logic are pure
  functions over plain data in `app/ingestion/`; `infra/pdfplumber_parser.py` only
  converts pdfplumber objects into that data.
- **Packaging:** ingestion ships in the backend image; pdfplumber is a runtime
  dependency of the backend project.

## Alternatives considered

- **PyMuPDF:** accurate and fast, but AGPL.
- **pypdf:** permissive, but no positions or fonts for layout decisions.
- **Docling / unstructured:** strong layout models, but heavy ML dependencies.
- **pdfplumber's table finders:** tested; they fail on both table styles.

## Consequences

- Only the adapter sees PDFs; every later stage is unit-tested with synthetic data.
- **Lost icons are a known limitation:** no vision pass; affected passages are counted.
- The row-band extractor depends on HP's ruled-row layout; new document styles need
  new corpus tests.
- **Single image trade-off:** the API image carries pdfplumber, pdfminer.six,
  pypdfium2 and Pillow, which it never imports: a larger image and attack surface for
  one build and one dependency set. Measured at Phase 1 commit 10b, the whole
  ingestion stack (these plus fastembed, onnxruntime, numpy and qdrant-client) grows
  the backend image from the Phase 0 baseline by about 92 MB compressed (63.6 to
  155.4 MB) and 362 MB unpacked (297 to 659 MB). Way out: a uv dependency group `ingest` for
  pdfplumber and a second Dockerfile target (`--target ingest`) that syncs it, while
  the API target syncs without the group.

## Corrections after implementation

Measured with the implemented parser (Phase 1, commit 4); corpus tests keep these true.
PDF page numbers.

- **Heading sizes:** bookmark levels 1-4 are set at 24/18/14/12 pt in both manuals (not
  18/14/12). OMEN's two level-5 headings are set at the 10 pt body size. A heading is
  the first line matching the title at 12 pt or more on the bookmark's page or the
  next; failing that, a match at any size is accepted **on the bookmark's own page
  only**. A leading chapter number or appendix letter ("A") is ignored. All 169 and
  113 bookmarks find their heading.
- **Index spans:** the index runs from its bookmark to the last page: ENVY 134-135,
  OMEN 89-91 (the table above lists only the first page).
- **Front matter:** pages 1-2 carry no footer; the following front-matter pages (ENVY
  3-6, OMEN 3-7) are numbered in roman numerals. Roman numerals are not printed page
  numbers: such pages have `printed_page = null`, and citations fall back to the PDF
  page. Numbered pages follow PDF − 6 (ENVY, 129 pages) and PDF − 7 (OMEN, 84 pages).
- **Vertical separators:** some ENVY tables (e.g. the error-code table, page 105) also
  draw vertical column separators. The extractor does not use them.
- **Table-row rule:** a table is a run of at least three rules of the same horizontal
  extent in which every band is a row, and a band is a row only if its text is mostly
  set at 9 pt or less (cells are 8 pt, captions 9 pt, body 10 pt). Empty bands and
  bands of body text, headings or boxed notes end a table, which also separates two
  tables of the same width on one page. Without a top border, the header is taken from
  the table-size lines just above the first rule, unless another rule lies in between.
  Tables with spanning title rows yield fewer columns, but rows are never split.
- **Row bands, corrected after commit 10:** a band is a row only if every line in it
  is set at 9 pt or less. The first rule used the dominant size, which let a band with
  a heading and intro over the next table's caption count as a row (OMEN PDF 16, 18,
  27; ADR 0005 corrections).
- **Font sizes** are rounded to 0.01 pt: pdfminer derives them from glyph boxes and
  reports values such as 8.000000000000057.
- **Icons:** an image becomes an `[icon]` placeholder only when it sits in a gap
  between two words of one line and is at most 3 ems tall. Images at the start or end
  of a line (OMEN's 72 margin symbols beside notes, row-start glyphs) and images
  outside lines (logos, figures) get none. Measured: ENVY 87 placeholders (57 in body
  text on 45 line segments, 30 in table cells), OMEN 3 (table cells). The spike's 43
  could not be reproduced (its scripts were not kept); 45 body-text segments is the
  closest match. Chunk text holds the same counts: commit 10's report showed 4 for
  OMEN because one table was duplicated across two sections (ADR 0005 corrections),
  not because of a fourth placeholder.

### Cleaning (Phase 1, commit 5)

Measured on both manuals with the implemented cleaning stage; corpus tests pin the
counts. PDF page numbers.

- **Order:** each line is dropped once, under the first matching rule: footer, TOC
  page, index page, chapter-opener list, non-Latin, non-English Latin, empty section.
- **Chapter-opener lists** are dropped by line position, not by page: the bullets that
  name the chapter's sections, plus the lead-in line just before them when it ends
  with a colon. The chapter's own introduction on that page is kept.
- **Function-word lists hold no English words:** a word that is also English ("in",
  "is", "do", "as", "a", "per", "am") is on the English list only. With shared words, a
  first draft left six English lines tied with a Dutch or Portuguese score.
- **Continuation rule, as implemented** (approved change to plan item 3): a line the
  main rule kept is dropped when the line above it in the same column and block was
  dropped as non-English, and its English score is below 2. Same block: same page and
  font size, top-to-top distance of at most 1.5 em. The spike's tails end their
  paragraphs and are mostly followed by a heading, so "between two dropped lines"
  missed two of the three. The rule drops 6 tails (PDF 122, 123 ×3, 124, 133).
- **Empty sections:** a section runs from its heading to the next matched heading; a
  chapter number or appendix letter on the heading's visual line belongs to the
  heading. A section with no kept line containing a letter or digit (after removing
  `[icon]`) is dropped with its heading. ENVY: 21 (8 chapter headings whose only
  content was the opener list, 12 notices whose text was in another language, and one
  heading directly followed by its first subsection); OMEN: 0.
- **Navigation and table residue (commit 6b):** three rules run after the language
  rules. A bullet list of at least two items in which every item is the title of a
  child bookmark of its section is dropped as `section_toc`, with a colon lead-in
  (ENVY: 19 lists; chapter openers stay `chapter_toc`). A ruled table that lost a
  line as `non_latin` and keeps no Latin word (two lowercase Latin letters in a row)
  is dropped as `non_latin_table` (ENVY PDF 130). URLs are not text for the empty
  check (ENVY PDF 133). Empty sections: ENVY 32, OMEN 0. Details: Phase 1 plan.
- **Question lead-ins (commit 6d):** a `section_toc` lead-in may also end with `?`
  (ENVY PDF 95, 100). Both sections are left empty: ENVY 34 empty sections.
- **Residue, kept by design:** three Latin-script lines in languages or forms the
  rule cannot judge (a one-word German label, a German label without function words,
  a Turkish line), and one Latin-script line in a language without a list (Serbian),
  which is dropped only because it happens to score as Portuguese.
