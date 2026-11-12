"""Behavior of heading spans: where each bookmark's heading and content begin and end."""

from app.domain.ingestion import OutlineEntry, TextLine
from app.ingestion.spans import HeadingSpan, heading_spans


def line(text: str, page: int = 1, top: float = 100, size: float = 10, x0: float = 50) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=size, x0=x0, top=top, x1=x0 + 40, bottom=top + size
    )


def entry(title: str, level: int = 1, page: int = 1) -> OutlineEntry:
    return OutlineEntry(level=level, title=title, pdf_page=page)


def test_a_span_runs_from_its_heading_to_the_next_heading() -> None:
    lines = (
        line("Cover", top=40),
        line("Setup", top=60, size=14),
        line("Body one.", top=100),
        line("Connect", top=200, size=14),
        line("Body two.", top=230),
    )

    spans = heading_spans((entry("Setup"), entry("Connect", 2)), lines)

    assert spans == (
        HeadingSpan(entry=0, start=1, body=2, end=3),
        HeadingSpan(entry=1, start=3, body=4, end=5),
    )


def test_a_label_on_the_heading_line_starts_the_span() -> None:
    lines = (
        line("Body.", top=40),
        line("1", top=82, size=24),
        line("Setup", top=82, size=24, x0=99),
        line("Body.", top=120),
    )

    [span] = heading_spans((entry("Setup"),), lines)

    assert (span.start, span.body, span.end) == (1, 3, 4)


def test_a_wrapped_heading_spans_its_lines_and_unmatched_bookmarks_have_none() -> None:
    lines = (
        line("Connect the", top=60, size=14),
        line("printer", top=78, size=14),
        line("Body.", top=110),
    )

    spans = heading_spans((entry("Connect the printer"), entry("Missing", 2)), lines)

    assert spans == (HeadingSpan(entry=0, start=0, body=2, end=3),)
    assert heading_spans((entry("Missing"),), lines) == ()
