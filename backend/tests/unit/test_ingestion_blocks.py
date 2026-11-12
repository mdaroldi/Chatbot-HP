"""Behavior of block building: paragraphs, steps, bullets, notes and tables."""

from collections.abc import Sequence

from app.domain.ingestion import Block, BlockKind, NoteLabel, PageRange, Table, TextLine
from app.ingestion.blocks import build_blocks


def line(
    text: str, top: float, x0: float = 100, *, page: int = 1, size: float = 10, x1: float = 500
) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=size, x0=x0, top=top, x1=x1, bottom=top + size
    )


def blocks(lines: Sequence[TextLine], tables: Sequence[Table] = ()) -> tuple[Block, ...]:
    return build_blocks(lines, tables, {1: 5, 2: 6})


def kinds_and_texts(result: Sequence[Block]) -> list[tuple[BlockKind, str]]:
    return [(b.kind, b.text) for b in result]


def test_lines_of_one_paragraph_join_and_a_gap_or_size_change_splits() -> None:
    result = blocks(
        (
            line("Keep the printer", 100),
            line("away from water.", 112),
            line("Use the supplied cord.", 140),  # paragraph gap
            line("Small print.", 152, size=8),  # other size
        )
    )

    assert kinds_and_texts(result) == [
        (BlockKind.PARAGRAPH, "Keep the printer away from water."),
        (BlockKind.PARAGRAPH, "Use the supplied cord."),
        (BlockKind.PARAGRAPH, "Small print."),
    ]


def test_a_hanging_step_marker_joins_its_text_and_continuation() -> None:
    result = blocks(
        (
            line("To load paper", 80, x0=54),
            line("1.", 100, x0=91, x1=98),
            line("Open the tray and", 100, x0=113),
            line("pull it out.", 112, x0=113),
            line("a.", 130, x0=113, x1=120),
            line("Lift the stop.", 130, x0=135),
            line("2.", 160, x0=91, x1=98),
            line("Close the tray.", 160, x0=113),
        )
    )

    assert kinds_and_texts(result) == [
        (BlockKind.PARAGRAPH, "To load paper"),
        (BlockKind.STEP, "1. Open the tray and pull it out."),
        (BlockKind.STEP, "a. Lift the stop."),
        (BlockKind.STEP, "2. Close the tray."),
    ]
    assert [b.indent for b in result] == [54, 91, 113, 91]


def test_bullets_are_their_own_blocks() -> None:
    result = blocks(
        (
            line("●", 100, x1=106),
            line("Do not bend the cable.", 100, x0=122),
            line("\u2013", 120, x0=122, x1=128),
            line("Not even slightly.", 120, x0=144),
        )
    )

    assert kinds_and_texts(result) == [
        (BlockKind.BULLET, "● Do not bend the cable."),
        (BlockKind.BULLET, "\u2013 Not even slightly."),
    ]


def test_labelled_notes_are_note_blocks_with_their_label() -> None:
    result = blocks(
        (
            line("CAUTION:", 100, x1=140),
            line("Unplug the printer", 100, x0=150),
            line("before cleaning it.", 112, x0=100),
            line("WARNING! Hot surface.", 140),
            line("NOTE: Paper is optional.", 170),
            line("Body text.", 200),
        )
    )

    assert kinds_and_texts(result) == [
        (BlockKind.NOTE, "CAUTION: Unplug the printer before cleaning it."),
        (BlockKind.NOTE, "WARNING! Hot surface."),
        (BlockKind.NOTE, "NOTE: Paper is optional."),
        (BlockKind.PARAGRAPH, "Body text."),
    ]
    assert [b.note for b in result] == [
        NoteLabel.CAUTION,
        NoteLabel.WARNING,
        NoteLabel.NOTE,
        None,
    ]


def test_a_table_replaces_its_lines_with_a_header_and_rows() -> None:
    table = Table(
        pdf_page=2,
        top=200,
        bottom=260,
        x0=90,
        x1=500,
        header=("Code", "Meaning"),
        rows=(("E1", "Paper jam"), ("E2", "")),
    )
    lines = (
        line("Error codes:", 180, page=2),
        line("Code", 205, page=2, size=8),
        line("Meaning", 205, x0=300, page=2, size=8),
        line("E1", 225, page=2, size=8),
        line("Paper jam", 225, x0=300, page=2, size=8),
        line("E2", 245, page=2, size=8),
        line("Restart the printer.", 280, page=2),
    )

    result = blocks(lines, (table,))

    assert kinds_and_texts(result) == [
        (BlockKind.PARAGRAPH, "Error codes:"),
        (BlockKind.TABLE_HEADER, "Code | Meaning"),
        (BlockKind.TABLE_ROW, "E1 | Paper jam"),
        (BlockKind.TABLE_ROW, "E2 |"),
        (BlockKind.PARAGRAPH, "Restart the printer."),
    ]
    assert result[1].pages == PageRange(pdf_start=2, pdf_end=2, printed_start=6, printed_end=6)


def test_blocks_carry_both_page_numberings() -> None:
    result = blocks(
        (
            line("Front matter text.", 700, page=3),
            line("Wraps onto", 700, page=1),
            line("the next page.", 60, page=2),
        )
    )

    assert [b.pages for b in result] == [
        PageRange(pdf_start=3, pdf_end=3, printed_start=None, printed_end=None),
        PageRange(pdf_start=1, pdf_end=1, printed_start=5, printed_end=5),
        PageRange(pdf_start=2, pdf_end=2, printed_start=6, printed_end=6),
    ]
