"""Behavior of footer detection and printed page numbers (synthetic lines)."""

from app.domain.ingestion import TextLine
from app.ingestion.footer import PageFooter, split_footers

HEIGHT = 800.0  # footer zone starts at 744 (bottom 7 %)


def line(text: str, top: float, x0: float = 50, page: int = 1) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=10, x0=x0, top=top, x1=x0 + 40, bottom=top + 10
    )


def test_number_at_the_left_is_the_printed_page_and_the_zone_is_dropped() -> None:
    body = line("Body text", 100)
    split = split_footers(
        (body, line("6", 760), line("Chapter 1 Setup", 760, x0=80), line("ID", 760, x0=500)),
        (HEIGHT,),
    )

    assert split.body == (body,)
    assert split.footers == {1: PageFooter(pdf_page=1, printed_page=6, text="6 Chapter 1 Setup ID")}


def test_number_at_the_right_is_the_printed_page() -> None:
    split = split_footers((line("ID Setup", 760), line("7", 760, x0=500)), (HEIGHT,))

    assert split.footers[1].printed_page == 7


def test_number_inside_a_single_segment_is_found_at_either_end() -> None:
    split = split_footers((line("Index 83", 760), line("84 Index", 760, page=2)), (HEIGHT, HEIGHT))

    assert [split.footers[p].printed_page for p in (1, 2)] == [83, 84]


def test_roman_front_matter_footer_is_dropped_but_has_no_printed_page() -> None:
    split = split_footers((line("ID iii", 760),), (HEIGHT,))

    assert split.body == ()
    assert split.footers[1].printed_page is None


def test_text_in_the_zone_without_a_page_number_is_body_text() -> None:
    """A cover paragraph can run into the bottom zone; it is not a footer."""
    cover = line("parts, security, and backing up.", 750)

    split = split_footers((cover,), (HEIGHT,))

    assert split.body == (cover,)
    assert split.footers == {}


def test_a_number_above_the_zone_is_body_text() -> None:
    step = line("12", 700)

    split = split_footers((step,), (HEIGHT,))

    assert split.body == (step,)
    assert split.footers == {}


def test_each_page_uses_its_own_height() -> None:
    split = split_footers(
        (line("5", 760), line("6", 560, page=2), line("7", 500, page=2)), (HEIGHT, 600.0)
    )

    assert [split.footers[p].printed_page for p in (1, 2)] == [5, 6]
    assert [ln.text for ln in split.body] == ["7"]


def test_body_keeps_its_order() -> None:
    lines = (line("a", 100), line("b", 200), line("1", 760), line("c", 100, page=2))

    assert [ln.text for ln in split_footers(lines, (HEIGHT, HEIGHT)).body] == ["a", "b", "c"]
