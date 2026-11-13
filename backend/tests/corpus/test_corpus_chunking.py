"""Sections and chunking v1 on the real manuals (PDF page numbers).

The word-counting stand-in tokenizer is used here; the 450-token limit with the
real bge-small tokenizer is checked by the corpus tests of the embedder commit.

CI logs are public: every assertion compares page numbers, counts or identifiers, never
document text, so a failure cannot print the manual (see tests/corpus/conftest.py).
"""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping

import pytest

from app.domain.ingestion import ICON, BlockKind, Chunk, ContentType
from app.ingestion.chunking import StructureChunker
from app.ingestion.front_matter import FRONT_MATTER
from app.ingestion.lines import count_icons
from app.ingestion.outline import clean_title, section_paths
from tests.corpus.conftest import ENVY, OMEN, Hidden, ParsedCorpusDocument
from tests.unit.fakes import WordTokenCounter

pytestmark = pytest.mark.corpus

Corpus = Mapping[str, ParsedCorpusDocument]
Chunks = Mapping[str, list[Chunk]]
BOTH = pytest.mark.parametrize("doc_id", [ENVY, OMEN])

PRINTED_OFFSET = {ENVY: 6, OMEN: 7}  # printed = PDF page - offset after the front matter
# Kept lines before the first bookmarked heading that no front-matter heading covers:
# the cover (PDF 1), copyright and trademark text (PDF 2). Not indexed.
UNSECTIONED = {ENVY: 15, OMEN: 83}
UNSECTIONED_PAGES = {ENVY: {1, 2}, OMEN: {1, 2}}
# Indexed front matter: (heading, PDF page, has_warning) per chunk.
FRONT = {
    ENVY: [("Safety information", 2, False)],
    OMEN: [
        ("Safety warning notice", 3, True),
        ("Important notice about Customer Self-Repair parts", 4, False),
    ],
}
ALL_TYPES = set(ContentType)
CONTENT_TYPES = {ENVY: ALL_TYPES - {ContentType.PARTS}, OMEN: ALL_TYPES}
PARTS = ContentType.PARTS


def build(corpus: Corpus) -> Hidden[list[Chunk]]:
    chunker = StructureChunker(WordTokenCounter(), max_tokens=450, min_tokens=50)
    return Hidden(
        {
            doc_id: chunker.chunk(doc.sectioned.sections, doc.parsed.document)
            for doc_id, doc in corpus.items()
        }
    )


@pytest.fixture(scope="module")
def chunks(corpus: Corpus) -> Chunks:
    return build(corpus)


def lines(chunks: list[Chunk]) -> list[tuple[Chunk, str]]:
    return [(c, text) for c in chunks for text in c.text.split("\n")]


def where(chunks: list[Chunk]) -> list[tuple[int, str]]:
    """How assertions name chunks: first PDF page and chunk ID, never their text."""
    return [(c.pages.pdf_start, c.chunk_id) for c in chunks]


@BOTH
def test_every_chunk_has_a_bookmark_path_and_both_page_ranges(
    corpus: Corpus, chunks: Chunks, doc_id: str
) -> None:
    paths = set(section_paths(corpus[doc_id].parsed.outline))
    offset = PRINTED_OFFSET[doc_id]
    front = [c for c in chunks[doc_id] if c.section_path[0] == FRONT_MATTER]
    body = [c for c in chunks[doc_id] if c.section_path[0] != FRONT_MATTER]
    counts = (len(front), len(body))
    off_outline = where([c for c in body if c.section_path not in paths])
    misnumbered = where(
        [
            c
            for c in body
            if (c.pages.printed_start, c.pages.printed_end)
            != (c.pages.pdf_start - offset, c.pages.pdf_end - offset)
        ]
    )
    numbered_front = where([c for c in front if c.pages.printed_start or c.pages.printed_end])
    longest = max(c.token_count for c in chunks[doc_id])

    assert min(counts) > 0
    assert off_outline == []
    assert misnumbered == []
    assert numbered_front == []
    assert longest <= 450  # stand-in tokens


@BOTH
def test_each_table_row_lands_in_one_section(corpus: Corpus, doc_id: str) -> None:
    """A table that ran across a heading was emitted whole into both sections."""
    owners: defaultdict[tuple[int, str], set[tuple[str, ...]]] = defaultdict(set)
    for section in corpus[doc_id].sectioned.sections:
        for block in section.blocks:
            if block.kind is BlockKind.TABLE_ROW:
                owners[(block.pages.pdf_start, block.text)].add(section.path)
    rows = len(owners)
    # (PDF page, number of sections) for every row that landed in more than one.
    shared = sorted((page, len(paths)) for (page, _), paths in owners.items() if len(paths) > 1)

    assert rows
    assert shared == []


@BOTH
def test_every_icon_placeholder_lands_in_one_chunk(
    corpus: Corpus, chunks: Chunks, doc_id: str
) -> None:
    kept = count_icons(corpus[doc_id].cleaned.lines)
    in_chunks = sum(c.text.count(ICON) for c in chunks[doc_id])

    assert kept
    assert in_chunks == kept


@BOTH
def test_chapters_whose_heading_was_dropped_stay_in_the_paths(
    corpus: Corpus, chunks: Chunks, doc_id: str
) -> None:
    """ENVY drops 8 chapter headings as empty; their sections still carry them.

    Chapters are named by their bookmark's PDF page (0 for the front matter, -1 for a
    chapter that is not a bookmark), not by their titles.
    """
    outline = corpus[doc_id].parsed.outline
    page_of = {clean_title(e.title): e.pdf_page for e in outline if e.level == 1}
    expected = {page for title, page in page_of.items() if title != "Index"} | {0}
    page_of[FRONT_MATTER] = 0
    found = {page_of.get(c.section_path[0], -1) for c in chunks[doc_id]}

    assert found == expected


@BOTH
def test_chunk_ids_are_unique_and_stable(corpus: Corpus, chunks: Chunks, doc_id: str) -> None:
    ids = [c.chunk_id for c in chunks[doc_id]]
    rebuilt = [c.chunk_id for c in build(corpus)[doc_id]]

    assert ids
    assert len(set(ids)) == len(ids)
    assert rebuilt == ids


@BOTH
def test_chunks_hold_only_latin_and_greek_letters_and_no_footer_label(
    chunks: Chunks, doc_id: str
) -> None:
    texts = [c.text for c in chunks[doc_id]]
    scripts = {unicodedata.name(ch).split()[0] for t in texts for ch in t if ch.isalpha()}
    labelled = where([c for c in chunks[doc_id] if "ENWW" in c.text])

    assert "LATIN" in scripts
    assert scripts <= {"LATIN", "GREEK"}
    assert labelled == []


@BOTH
def test_content_types_and_unsectioned_lines(corpus: Corpus, chunks: Chunks, doc_id: str) -> None:
    assert {c.content_type for c in chunks[doc_id]} == CONTENT_TYPES[doc_id]
    unsectioned = corpus[doc_id].sectioned.unsectioned
    count, pages = len(unsectioned), {line.pdf_page for line in unsectioned}

    assert count == UNSECTIONED[doc_id]
    assert pages == UNSECTIONED_PAGES[doc_id]


def test_error_code_rows_share_one_table_chunk(chunks: Chunks) -> None:
    codes = re.compile(r"(E\d) \| \S")
    found = [(c, m.group(1)) for c, text in lines(chunks[ENVY]) if (m := codes.match(text))]
    found_codes = [code for _, code in found]
    homes = {(c.chunk_id, c.pages.pdf_start, c.content_type) for c, _ in found}
    first_home = found[0][0].chunk_id if found else None

    assert found_codes == ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E9"]
    assert homes == {(first_home, 105, ContentType.TABLE)}


def test_spare_part_rows_keep_their_component(chunks: Chunks) -> None:
    rows = [(c, t) for c, t in lines(chunks[OMEN]) if "N96208-001" in t and " | " in t]
    found = len(rows)
    wrong = where([c for c, t in rows if not ("Bottom cover" in t and c.content_type is PARTS)])

    assert found
    assert wrong == []


def test_regulatory_model_number_is_in_a_regulatory_chunk(chunks: Chunks) -> None:
    found = [c for c in chunks[ENVY] if "SNPRC-1807-01" in c.text]
    count, types = len(found), {c.content_type for c in found}

    assert count
    assert types == {ContentType.REGULATORY}


def test_the_wi_fi_direct_device_limit_is_indexed(chunks: Chunks) -> None:
    found = len([c for c in chunks[ENVY] if "Wi-Fi Direct" in c.text and "five" in c.text])

    assert found


def test_hazard_notes_set_has_warning(chunks: Chunks) -> None:
    for doc_chunks in chunks.values():
        flagged = {c.chunk_id for c in doc_chunks if c.has_warning}
        hazards = {
            c.chunk_id for c, t in lines(doc_chunks) if t.startswith(("WARNING!", "CAUTION:"))
        }

        assert flagged
        assert flagged == hazards


@BOTH
def test_front_matter_safety_and_notices_are_indexed_and_legal_text_is_not(
    chunks: Chunks, doc_id: str
) -> None:
    front = [c for c in chunks[doc_id] if c.section_path[0] == FRONT_MATTER]
    expected = FRONT[doc_id]
    found = [(c.pages.pdf_start, c.has_warning) for c in front]
    ends = {(c.pages.pdf_end, c.content_type) for c in front}
    # The bookmarked legal notices stay indexed; the unbookmarked front matter does not.
    legal = where([c for c in front if "©" in c.text or "trademark" in c.text.casefold()])
    cover = where([c for c in chunks[doc_id] if c.pages.pdf_start == 1])
    # Compared outside the assert, so a wrong heading is never printed.
    headings_match = [c.section_path[1] for c in front] == [h for h, _, _ in expected]

    assert found == [(page, warning) for _, page, warning in expected]
    assert headings_match
    assert ends == {(page, ContentType.FRONT_MATTER) for _, page, _ in expected}
    assert legal == []
    assert cover == []


def test_front_matter_notes_keep_their_labels(chunks: Chunks) -> None:
    [warning, important] = [c for c in chunks[OMEN] if c.section_path[0] == FRONT_MATTER]
    labels = ("\nWARNING! " in warning.text, "\nIMPORTANT: " in important.text)

    assert labels == (True, True)
