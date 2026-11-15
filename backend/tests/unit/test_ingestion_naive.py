"""Behavior of the naive baseline chunker (Phase 5 ablation): fixed token windows with
overlap over the cleaned text in reading order, the document title as the only prefix.
"""

import hashlib
import threading
import uuid
from collections.abc import Sequence
from itertools import pairwise

import pytest

from app.domain.ingestion import (
    Block,
    BlockKind,
    ChunkTooLargeError,
    ContentType,
    CorpusDocument,
    NoteLabel,
    PageRange,
    Section,
)
from app.ingestion.chunking import CHUNK_NAMESPACE
from app.ingestion.naive import OVERLAP_TOKENS, VERSION, WINDOW_TOKENS, NaiveChunker
from tests.unit.fakes import WordTokenCounter

DOC = CorpusDocument(
    doc_id="test-guide",
    title="Guide",
    product="Printer",
    url="https://example.org/guide.pdf",
    sha256="0" * 64,
)


class SpecialTokenCounter:
    """Counts words plus the two special tokens the real tokenizer adds ([CLS], [SEP])."""

    def count(self, text: str) -> int:
        return len(text.split()) + 2


def pages(start: int, end: int | None = None, printed: int | None = None) -> PageRange:
    end = start if end is None else end
    return PageRange(
        pdf_start=start,
        pdf_end=end,
        printed_start=printed,
        printed_end=None if printed is None else printed + end - start,
    )


def para(text: str, page: int = 1, printed: int | None = None) -> Block:
    return Block(kind=BlockKind.PARAGRAPH, text=text, pages=pages(page, printed=printed))


def warning(text: str, page: int = 1) -> Block:
    return Block(
        kind=BlockKind.NOTE, text=f"WARNING: {text}", pages=pages(page), note=NoteLabel.WARNING
    )


def section(path: tuple[str, ...], *blocks: Block) -> Section:
    return Section(doc_id=DOC.doc_id, path=path, blocks=blocks)


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def body(text: str) -> list[str]:
    """The words of a chunk after its title line."""
    title, _, rest = text.partition("\n\n")
    assert title == DOC.title
    return rest.split()


def chunker(window: int = 10, overlap: int = 3) -> NaiveChunker:
    return NaiveChunker(WordTokenCounter(), window_tokens=window, overlap_tokens=overlap)


# ------------------------------------------------------------------ text and metadata


def test_a_short_document_is_one_chunk_with_the_title_prefix_and_headings_as_text() -> None:
    sections = [section(("Setup", "Unpack"), para("Remove the tape."), para("Open the lid."))]

    [chunk] = chunker(window=50).chunk(sections, DOC)

    assert chunk.text == "Guide\n\nUnpack\nRemove the tape.\nOpen the lid."
    assert "Setup" not in chunk.text  # no breadcrumb: only the heading line on the page
    assert chunk.section_path == (DOC.title,)
    assert chunk.content_type is ContentType.TEXT
    assert chunk.chunking_version == VERSION == "naive-v1"
    assert (chunk.doc_id, chunk.product) == (DOC.doc_id, DOC.product)
    assert chunk.token_count == WordTokenCounter().count(chunk.text)
    assert not chunk.has_warning


def test_no_sections_give_no_chunks() -> None:
    assert chunker().chunk([], DOC) == []


def test_chunk_ids_are_uuid5_of_document_version_index_and_text() -> None:
    sections = [section(("A",), para(words("w", 30)))]

    chunks = chunker().chunk(sections, DOC)

    assert len(chunks) == 5
    for index, chunk in enumerate(chunks):
        digest = hashlib.sha256(chunk.text.encode()).hexdigest()
        expected = uuid.uuid5(CHUNK_NAMESPACE, f"{DOC.doc_id}|{VERSION}|{index}|{digest}")
        assert chunk.chunk_id == str(expected)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [c.chunk_id for c in chunker().chunk(sections, DOC)] == [c.chunk_id for c in chunks]


# ------------------------------------------------------------------ windows


def test_windows_fill_the_budget_and_overlap_by_the_set_amount() -> None:
    sections = [section(("A",), para(words("w", 25)))]  # 26 words with the heading

    chunks = chunker(window=10, overlap=3).chunk(sections, DOC)

    bodies = [body(chunk.text) for chunk in chunks]
    # The title costs 1 of the 10 tokens, so each window holds 9 words and moves by 6.
    assert [len(b) for b in bodies] == [9, 9, 9, 8]
    assert all(chunk.token_count <= 10 for chunk in chunks)
    for earlier, later in pairwise(bodies):
        assert earlier[-3:] == later[:3]


def test_every_word_appears_once_in_reading_order_after_removing_the_overlaps() -> None:
    sections = [
        section(("A",), para(words("a", 7)), para(words("b", 5))),
        section(("B",), para(words("c", 11))),
    ]
    expected = ["A", *words("a", 7).split(), *words("b", 5).split(), "B", *words("c", 11).split()]

    bodies = [body(chunk.text) for chunk in chunker(window=10, overlap=3).chunk(sections, DOC)]

    rebuilt = bodies[0] + [word for b in bodies[1:] for word in b[3:]]
    assert rebuilt == expected


def test_windows_cross_section_boundaries() -> None:
    sections = [section(("A",), para("one two")), section(("B",), para("three four"))]

    [chunk] = chunker(window=20).chunk(sections, DOC)

    assert chunk.text == "Guide\n\nA\none two\nB\nthree four"


def test_a_block_split_by_a_window_keeps_its_words_on_one_line() -> None:
    sections = [section(("A",), para(words("w", 12)), para("tail end"))]

    chunks = chunker(window=10, overlap=3).chunk(sections, DOC)

    assert chunks[0].text == "Guide\n\nA\n" + words("w", 8)
    assert chunks[1].text == "Guide\n\n" + " ".join(f"w{i}" for i in range(5, 12)) + "\ntail end"


def test_the_budget_counts_the_tokenizers_special_tokens() -> None:
    sections = [section(("A",), para(words("w", 30)))]

    chunks = NaiveChunker(SpecialTokenCounter(), window_tokens=12, overlap_tokens=3).chunk(
        sections, DOC
    )

    assert [SpecialTokenCounter().count(chunk.text) for chunk in chunks[:-1]] == [12] * (
        len(chunks) - 1
    )
    assert all(chunk.token_count <= 12 for chunk in chunks)
    assert body(chunks[0].text)[-3:] == body(chunks[1].text)[:3]


def test_a_word_longer_than_the_overlap_still_moves_the_window_forward() -> None:
    class Weighted:
        """Each word costs its length in tokens."""

        def count(self, text: str) -> int:
            return sum(len(word) for word in text.split())

    sections = [section(("A",), para("aaaa bbbbbbbbb cccc dddd"))]

    chunks = NaiveChunker(Weighted(), window_tokens=19, overlap_tokens=3).chunk(sections, DOC)

    # Title "Guide" costs 5; the windows hold (A, aaaa, bbbbbbbbb), then the overlap word
    # bbbbbbbbb is over 3 tokens so the next window starts after it.
    assert [body(chunk.text) for chunk in chunks] == [
        ["A", "aaaa", "bbbbbbbbb"],
        ["cccc", "dddd"],
    ]


def test_a_window_that_fits_inside_the_overlap_still_moves_forward() -> None:
    """Restarting at the window's own first word would loop forever: the chunker runs in
    a daemon thread so that such a bug fails here instead of hanging the suite.
    """

    class Weighted:
        def count(self, text: str) -> int:
            return sum(len(word) for word in text.split())

    sections = [section(("A",), para("bb " + "x" * 9))]  # A and bb cost 3, the overlap
    found: list[list[list[str]]] = []

    def run() -> None:
        chunks = NaiveChunker(Weighted(), window_tokens=14, overlap_tokens=3).chunk(sections, DOC)
        found.append([body(chunk.text) for chunk in chunks])

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert found == [[["A", "bb"], ["bb"], ["x" * 9]]]


def test_a_word_that_fits_no_window_is_an_error_naming_the_document() -> None:
    sections = [section(("A",), para("short " + "x" * 40))]

    class Weighted:
        def count(self, text: str) -> int:
            return sum(len(word) for word in text.split())

    with pytest.raises(ChunkTooLargeError, match="test-guide"):
        NaiveChunker(Weighted(), window_tokens=20, overlap_tokens=2).chunk(sections, DOC)


def test_a_title_that_fills_the_window_is_an_error() -> None:
    with pytest.raises(ChunkTooLargeError, match="test-guide"):
        chunker(window=1, overlap=0).chunk([section(("A",), para("x"))], DOC)


# ------------------------------------------------------------------ pages and warnings


def test_pages_span_the_blocks_each_window_covers() -> None:
    sections = [
        section(("A",), para(words("p", 6), page=3, printed=1)),
        section(("B",), para(words("q", 6), page=4, printed=2)),
        section(("C",), para(words("r", 6), page=6, printed=None)),
    ]

    chunks = chunker(window=10, overlap=0).chunk(sections, DOC)

    assert [(c.pages.pdf_start, c.pages.pdf_end) for c in chunks] == [(3, 4), (4, 6), (6, 6)]
    assert [(c.pages.printed_start, c.pages.printed_end) for c in chunks] == [
        (1, 2),
        (2, 2),  # the unnumbered page keeps the numbered part
        (None, None),
    ]


def test_has_warning_marks_only_windows_holding_words_of_a_hazard_note() -> None:
    sections = [section(("A",), para(words("a", 8)), warning("hot"), para(words("b", 16)))]

    chunks = chunker(window=10, overlap=0).chunk(sections, DOC)

    assert [chunk.has_warning for chunk in chunks] == [False, True, False]


# ------------------------------------------------------------------ settings


def test_the_shipped_window_is_256_tokens_with_64_of_overlap() -> None:
    default = NaiveChunker(WordTokenCounter())

    assert (WINDOW_TOKENS, OVERLAP_TOKENS) == (256, 64)
    assert (default.window_tokens, default.overlap_tokens) == (256, 64)
    assert default.version == VERSION


@pytest.mark.parametrize(("window", "overlap"), [(0, 0), (10, 10), (10, 11), (10, -1)])
def test_the_overlap_must_be_smaller_than_a_positive_window(window: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        NaiveChunker(WordTokenCounter(), window_tokens=window, overlap_tokens=overlap)


def test_the_text_of_every_chunk_is_a_sequence_of_lines_without_blanks() -> None:
    sections: Sequence[Section] = [section(("A",), para(words("w", 40)))]

    chunks = chunker().chunk(sections, DOC)

    assert len(chunks) == 7
    for chunk in chunks:
        title, _, rest = chunk.text.partition("\n\n")
        assert title == DOC.title
        assert all(line.strip() for line in rest.split("\n"))
