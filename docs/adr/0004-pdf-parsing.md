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
  one build and one dependency set. Way out: a uv dependency group `ingest` for
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
- **Font sizes** are rounded to 0.01 pt: pdfminer derives them from glyph boxes and
  reports values such as 8.000000000000057.
