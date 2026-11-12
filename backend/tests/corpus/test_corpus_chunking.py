"""Sections and chunking v1 on the real manuals (PDF page numbers).

The word-counting stand-in tokenizer is used here; the 450-token limit with the
real bge-small tokenizer is checked by the corpus tests of the embedder commit.
"""

import re
import unicodedata
from collections.abc import Mapping

import pytest

from app.domain.ingestion import Chunk, ContentType
from app.ingestion.chunking import StructureChunker
from app.ingestion.front_matter import FRONT_MATTER
from app.ingestion.outline import clean_title, section_paths
from tests.corpus.conftest import ENVY, OMEN, ParsedCorpusDocument
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


def build(corpus: Corpus) -> dict[str, list[Chunk]]:
    chunker = StructureChunker(WordTokenCounter(), max_tokens=450, min_tokens=50)
    return {
        doc_id: chunker.chunk(doc.sectioned.sections, doc.parsed.document)
        for doc_id, doc in corpus.items()
    }


@pytest.fixture(scope="module")
def chunks(corpus: Corpus) -> Chunks:
    return build(corpus)


def lines(chunks: list[Chunk]) -> list[tuple[Chunk, str]]:
    return [(c, text) for c in chunks for text in c.text.split("\n")]


@BOTH
def test_every_chunk_has_a_bookmark_path_and_both_page_ranges(
    corpus: Corpus, chunks: Chunks, doc_id: str
) -> None:
    paths = set(section_paths(corpus[doc_id].parsed.outline))
    offset = PRINTED_OFFSET[doc_id]
    front = [c for c in chunks[doc_id] if c.section_path[0] == FRONT_MATTER]
    body = [c for c in chunks[doc_id] if c.section_path[0] != FRONT_MATTER]

    assert front
    assert body
    assert all(c.section_path in paths for c in body)
    assert all(
        (c.pages.printed_start, c.pages.printed_end)
        == (c.pages.pdf_start - offset, c.pages.pdf_end - offset)
        for c in body
    )
    assert all((c.pages.printed_start, c.pages.printed_end) == (None, None) for c in front)
    assert max(c.token_count for c in chunks[doc_id]) <= 450  # stand-in tokens


@BOTH
def test_chapters_whose_heading_was_dropped_stay_in_the_paths(
    corpus: Corpus, chunks: Chunks, doc_id: str
) -> None:
    """ENVY drops 8 chapter headings as empty; their sections still carry them."""
    outline = corpus[doc_id].parsed.outline
    chapters = {clean_title(e.title) for e in outline if e.level == 1} - {"Index"} | {FRONT_MATTER}

    assert {c.section_path[0] for c in chunks[doc_id]} == chapters


@BOTH
def test_chunk_ids_are_unique_and_stable(corpus: Corpus, chunks: Chunks, doc_id: str) -> None:
    ids = [c.chunk_id for c in chunks[doc_id]]

    assert ids
    assert len(set(ids)) == len(ids)
    assert [c.chunk_id for c in build(corpus)[doc_id]] == ids


@BOTH
def test_chunks_hold_only_latin_and_greek_letters_and_no_footer_label(
    chunks: Chunks, doc_id: str
) -> None:
    texts = [c.text for c in chunks[doc_id]]
    scripts = {unicodedata.name(ch).split()[0] for t in texts for ch in t if ch.isalpha()}

    assert "LATIN" in scripts
    assert scripts <= {"LATIN", "GREEK"}
    assert not [t for t in texts if "ENWW" in t]


@BOTH
def test_content_types_and_unsectioned_lines(corpus: Corpus, chunks: Chunks, doc_id: str) -> None:
    assert {c.content_type for c in chunks[doc_id]} == CONTENT_TYPES[doc_id]
    unsectioned = corpus[doc_id].sectioned.unsectioned

    assert len(unsectioned) == UNSECTIONED[doc_id]
    assert {line.pdf_page for line in unsectioned} == UNSECTIONED_PAGES[doc_id]


def test_error_code_rows_share_one_table_chunk(chunks: Chunks) -> None:
    codes = re.compile(r"(E\d) \| \S")
    found = [(c, m.group(1)) for c, text in lines(chunks[ENVY]) if (m := codes.match(text))]

    assert [code for _, code in found] == ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E9"]
    assert {(c.chunk_id, c.pages.pdf_start, c.content_type) for c, _ in found} == {
        (found[0][0].chunk_id, 105, ContentType.TABLE)
    }


def test_spare_part_rows_keep_their_component(chunks: Chunks) -> None:
    rows = [(c, t) for c, t in lines(chunks[OMEN]) if "N96208-001" in t and " | " in t]

    assert rows
    assert all("Bottom cover" in t and c.content_type is ContentType.PARTS for c, t in rows)


def test_regulatory_model_number_is_in_a_regulatory_chunk(chunks: Chunks) -> None:
    found = [c for c in chunks[ENVY] if "SNPRC-1807-01" in c.text]

    assert found
    assert {c.content_type for c in found} == {ContentType.REGULATORY}


def test_the_wi_fi_direct_device_limit_is_indexed(chunks: Chunks) -> None:
    assert [c for c in chunks[ENVY] if "Wi-Fi Direct" in c.text and "five" in c.text]


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
    texts = [c.text for c in front]  # the bookmarked legal notices stay indexed

    assert [(c.section_path[1], c.pages.pdf_start, c.has_warning) for c in front] == FRONT[doc_id]
    assert {(c.pages.pdf_end, c.content_type) for c in front} == {
        (page, ContentType.FRONT_MATTER) for _, page, _ in FRONT[doc_id]
    }
    assert not [t for t in texts if "©" in t or "trademark" in t.casefold()]
    assert not [c for c in chunks[doc_id] if c.pages.pdf_start == 1]


def test_front_matter_notes_keep_their_labels(chunks: Chunks) -> None:
    [warning, important] = [c for c in chunks[OMEN] if c.section_path[0] == FRONT_MATTER]

    assert "\nWARNING! " in warning.text
    assert "\nIMPORTANT: " in important.text
