"""Behavior of chunking v1 (word-counting fake tokenizer; prefixes count too)."""

import hashlib
import uuid
from collections.abc import Sequence

import pytest

from app.domain.ingestion import (
    Block,
    BlockKind,
    Chunk,
    ChunkTooLargeError,
    ContentType,
    CorpusDocument,
    NoteLabel,
    PageRange,
    Section,
)
from app.ingestion.chunking import CHUNK_NAMESPACE, StructureChunker
from tests.unit.fakes import WordTokenCounter

DOC = CorpusDocument(
    doc_id="test-guide",
    title="Guide",
    product="Printer",
    url="https://example.org/guide.pdf",
    sha256="0" * 64,
)


def block(
    kind: BlockKind,
    text: str,
    *,
    page: int = 1,
    printed: int | None = 5,
    indent: float = 100,
    note: NoteLabel | None = None,
) -> Block:
    return Block(
        kind=kind,
        text=text,
        pages=PageRange(pdf_start=page, pdf_end=page, printed_start=printed, printed_end=printed),
        note=note,
        indent=indent,
    )


def para(text: str, **kwargs: object) -> Block:
    return block(BlockKind.PARAGRAPH, text, **kwargs)  # type: ignore[arg-type]


def step(text: str, indent: float = 100) -> Block:
    return block(BlockKind.STEP, text, indent=indent)


def caution(text: str, indent: float) -> Block:
    return block(BlockKind.NOTE, f"CAUTION: {text}", indent=indent, note=NoteLabel.CAUTION)


def section(path: Sequence[str], *blocks: Block) -> Section:
    return Section(doc_id=DOC.doc_id, path=tuple(path), blocks=blocks)


def chunk(sections: Sequence[Section], max_tokens: int = 450, min_tokens: int = 0) -> list[Chunk]:
    return StructureChunker(WordTokenCounter(), max_tokens, min_tokens).chunk(sections, DOC)


def bodies(chunks: Sequence[Chunk]) -> list[str]:
    return [c.text.split("\n\n", 1)[1] for c in chunks]


def test_a_short_section_is_one_chunk_with_the_path_prefix() -> None:
    sections = [
        section(
            ("Setup", "Connect"),
            para("Connect the cable."),
            para("Plug it in.", page=2, printed=6),
        )
    ]

    [only] = chunk(sections)

    assert only.text == "Guide > Setup > Connect\n\nConnect the cable.\nPlug it in."
    assert (only.doc_id, only.product) == (DOC.doc_id, "Printer")
    assert only.section_path == sections[0].path
    assert only.pages == PageRange(pdf_start=1, pdf_end=2, printed_start=5, printed_end=6)
    assert (only.content_type, only.has_warning) == (ContentType.REFERENCE, False)
    assert (only.chunking_version, only.token_count) == ("v1", 11)


def test_chunk_ids_follow_the_formula_and_are_stable() -> None:
    sections = [section(("Setup",), para("Connect the cable."))]

    first, again = chunk(sections)[0], chunk(sections)[0]
    other = chunk([section(("Setup",), para("Connect the cord."))])[0]
    digest = hashlib.sha256(first.text.encode()).hexdigest()

    assert first.chunk_id == str(uuid.uuid5(CHUNK_NAMESPACE, f"test-guide|Setup|0|{digest}"))
    assert again.chunk_id == first.chunk_id
    assert other.chunk_id != first.chunk_id


def test_procedures_split_at_step_boundaries_and_repeat_the_title() -> None:
    steps = [step(f"{n}. one two three four five six seven eight") for n in range(1, 5)]

    chunks = chunk([section(("Setup",), *steps)], max_tokens=25)

    assert [c.text.split("\n\n")[0] for c in chunks] == [
        "Guide > Setup",
        "Guide > Setup (continued)",
    ]
    assert [len(b.splitlines()) for b in bodies(chunks)] == [2, 2]
    assert {c.content_type for c in chunks} == {ContentType.PROCEDURE}
    assert max(c.token_count for c in chunks) <= 25


def test_hazard_notes_stay_with_their_step() -> None:
    """An indented note belongs to the step above it; an outer one to the step below."""
    blocks = (
        step("1. aa bb cc dd ee"),
        caution("hot inside.", indent=122),
        caution("unplug first.", indent=100),
        step("2. aa bb cc dd ee"),
        step("3. aa bb cc dd ee"),
        block(BlockKind.NOTE, "NOTE: paper optional.", indent=122, note=NoteLabel.NOTE),
    )

    chunks = chunk([section(("Setup",), *blocks)], max_tokens=15)

    assert bodies(chunks) == [
        "1. aa bb cc dd ee\nCAUTION: hot inside.",
        "CAUTION: unplug first.\n2. aa bb cc dd ee",
        "3. aa bb cc dd ee\nNOTE: paper optional.",
    ]
    assert [c.has_warning for c in chunks] == [True, True, False]


def test_a_split_table_repeats_its_header_and_never_cuts_rows() -> None:
    rows = [block(BlockKind.TABLE_ROW, f"E{n} | Paper jam") for n in range(5)]
    blocks = (
        para("See the codes below."),
        block(BlockKind.TABLE_HEADER, "Code | Meaning"),
        *rows,
    )

    chunks = chunk([section(("Codes",), *blocks)], max_tokens=15)

    assert bodies(chunks) == [
        "See the codes below.",
        "Code | Meaning\nE0 | Paper jam\nE1 | Paper jam",
        "Code | Meaning\nE2 | Paper jam\nE3 | Paper jam",
        "Code | Meaning\nE4 | Paper jam",
    ]
    assert [c.content_type for c in chunks] == [ContentType.REFERENCE] + [ContentType.TABLE] * 3


def test_a_caption_travels_with_its_table_and_part_numbers_make_it_parts() -> None:
    blocks = (
        para("Table 3-1 Spare parts"),
        block(BlockKind.TABLE_HEADER, "Component | Spare part number"),
        block(BlockKind.TABLE_ROW, "Bottom cover | N12345-001"),
    )

    [only] = chunk([section(("Parts",), *blocks)])

    assert bodies([only]) == [
        "Table 3-1 Spare parts\nComponent | Spare part number\nBottom cover | N12345-001"
    ]
    assert only.content_type is ContentType.PARTS


def test_a_chunk_holds_prose_and_at_most_one_table() -> None:
    blocks = (
        para("Remove the cover."),
        block(BlockKind.TABLE_HEADER, "Part | Number"),
        block(BlockKind.TABLE_ROW, "Cover | N12345-001"),
        block(BlockKind.TABLE_HEADER, "Code | Meaning"),
        block(BlockKind.TABLE_ROW, "E1 | Jam"),
        para("Done."),
    )

    chunks = chunk([section(("Cover",), *blocks)])

    assert bodies(chunks) == [
        "Remove the cover.\nPart | Number\nCover | N12345-001",
        "Code | Meaning\nE1 | Jam\nDone.",
    ]
    assert [c.content_type for c in chunks] == [ContentType.PARTS, ContentType.TABLE]


def test_a_trailing_hazard_note_stays_with_the_last_step() -> None:
    blocks = (step("1. Open the cover."), step("2. Remove the part."), caution("sharp edges.", 100))

    [only] = chunk([section(("Setup",), *blocks)])

    assert bodies([only]) == ["1. Open the cover.\n2. Remove the part.\nCAUTION: sharp edges."]
    assert only.has_warning


def test_an_oversized_step_splits_between_its_blocks() -> None:
    blocks = (
        step("1. a b c d e"),
        block(BlockKind.NOTE, "NOTE: x y z", indent=122, note=NoteLabel.NOTE),
        para("more words here now", indent=122),
    )

    chunks = chunk([section(("Setup",), *blocks)], max_tokens=13)

    assert bodies(chunks) == ["1. a b c d e\nNOTE: x y z", "more words here now"]


def test_small_sections_merge_into_the_next_sibling_under_the_parent_path() -> None:
    sections = [
        section(("Parent", "A"), para("Short.")),
        section(("Parent", "B"), para("B text with several more words in it.")),
        section(("Q",), para("Short.")),  # its next section is its child: no merge
        section(("Q", "R"), para("Short.")),  # last one: nothing to merge into
    ]

    chunks = chunk(sections, min_tokens=10)

    assert [c.section_path for c in chunks] == [("Parent",), ("Q",), ("Q", "R")]
    assert chunks[0].text == (
        "Guide > Parent\n\nA\nShort.\nB\nB text with several more words in it."
    )


def test_a_merge_never_exceeds_the_limit() -> None:
    sections = [
        section(("Parent", "A"), para("Short.")),
        section(("Parent", "B"), para("one two three four five six seven")),
    ]

    chunks = chunk(sections, max_tokens=12, min_tokens=10)

    assert [c.section_path for c in chunks] == [("Parent", "A"), ("Parent", "B")]


def test_long_prose_splits_at_sentences_then_words() -> None:
    sentences = ["Alpha beta gamma delta epsilon."] * 6
    long_sentence = " ".join(f"w{n}" for n in range(20))
    sections = [section(("Prose",), para(" ".join(sentences)))]

    prose = chunk(sections, max_tokens=15)
    words = chunk([section(("Prose",), para(long_sentence))], max_tokens=15)

    assert bodies(prose) == [" ".join(sentences[:2])] * 3
    assert " ".join(bodies(words)) == long_sentence
    assert max(c.token_count for c in [*prose, *words]) <= 15


def test_a_word_that_fits_no_chunk_raises_instead_of_overflowing() -> None:
    """Words are the smallest piece; one over the limit is an error, never a chunk
    over the limit. Prefixes: "Guide > Prose" (3), with " (continued)" (4).
    """
    sections = [section(("Prose",), para("one two"))]

    with pytest.raises(ChunkTooLargeError, match=r"test-guide: Prose: .* 5 tokens, limit 4"):
        chunk(sections, max_tokens=4)  # "two" fits no continued chunk
    with pytest.raises(ChunkTooLargeError, match=r"test-guide: Prose: .* 4 tokens, limit 3"):
        chunk(sections, max_tokens=3)  # "one" fits no chunk at all
    assert bodies(chunk(sections, max_tokens=5)) == ["one two"]


def test_a_table_row_that_fits_no_chunk_raises_instead_of_being_cut() -> None:
    rows = [block(BlockKind.TABLE_ROW, f"E{n} | Paper jam") for n in range(2)]
    sections = [section(("Codes",), block(BlockKind.TABLE_HEADER, "Code | Meaning"), *rows)]

    with pytest.raises(ChunkTooLargeError, match=r"test-guide: Codes: .* 11 tokens, limit 10"):
        chunk(sections, max_tokens=10)
    assert len(chunk(sections, max_tokens=11)) == 2


def test_content_types_follow_the_bookmark_path() -> None:
    sections = [
        section(("Legal", "Regulatory notices", "Radio"), step("1. Keep a distance.")),
        section(("Product", "Specifications"), para("Weight: 5 kg.")),
        section(("Setup",), step("1. Open the tray.")),
        section(("Setup", "Overview"), para("The printer has a tray.")),
    ]

    types = [c.content_type for c in chunk(sections)]

    assert types == [
        ContentType.REGULATORY,
        ContentType.SPECS,
        ContentType.PROCEDURE,
        ContentType.REFERENCE,
    ]


def test_the_chunker_reports_its_version() -> None:
    assert StructureChunker(WordTokenCounter(), 450, 0).version == "v1"
