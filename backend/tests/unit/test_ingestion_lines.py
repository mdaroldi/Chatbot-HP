"""Behavior of word-to-line grouping (synthetic words, no PDF)."""

from app.domain.ingestion import ICON, ImageBox, TextLine, Word
from app.ingestion.lines import count_icons, group_lines


def word(text: str, x0: float, top: float = 100, size: float = 10, page: int = 1) -> Word:
    width = len(text) * size * 0.5
    return Word(
        text=text, pdf_page=page, font_size=size, x0=x0, top=top, x1=x0 + width, bottom=top + size
    )


def texts(lines: tuple[TextLine, ...]) -> list[str]:
    return [line.text for line in lines]


def test_words_on_one_baseline_join_into_one_line_with_its_box() -> None:
    # "Open" ends at x=70; a 3 pt word space keeps it in the same run.
    lines = group_lines((word("the", 73, top=101), word("Open", 50)))

    assert lines == (
        TextLine(text="Open the", pdf_page=1, font_size=10, x0=50, top=100, x1=88, bottom=111),
    )


def test_a_gap_wider_than_one_em_starts_a_new_segment() -> None:
    """Table columns and the gaps left by inline icons both split a visual line."""
    lines = group_lines((word("Code", 50), word("Meaning", 78), word("Press", 200)))

    assert texts(lines) == ["Code Meaning", "Press"]
    assert [line.x0 for line in lines] == [50, 200]


def test_the_gap_threshold_follows_the_font_size() -> None:
    small = group_lines((word("ab", 0, size=8), word("cd", 8 + 9, size=8)))
    large = group_lines((word("ab", 0, size=20), word("cd", 20 + 9, size=20)))

    assert texts(small) == ["ab", "cd"]
    assert texts(large) == ["ab cd"]


def test_lines_are_ordered_by_page_then_top_then_left() -> None:
    lines = group_lines(
        (
            word("third", 50, top=200),
            word("fifth", 50, top=100, page=2),
            word("second", 300, top=100),
            word("first", 50, top=100),
            word("fourth", 50, top=300),
        )
    )

    assert texts(lines) == ["first", "second", "third", "fourth", "fifth"]
    assert [line.pdf_page for line in lines] == [1, 1, 1, 1, 2]


def test_small_baseline_jitter_stays_on_one_line_but_the_next_line_does_not() -> None:
    lines = group_lines((word("a", 50, top=100), word("b", 58, top=101.5), word("c", 50, top=112)))

    assert texts(lines) == ["a b", "c"]


def test_line_font_size_is_the_size_of_most_characters() -> None:
    lines = group_lines((word("Model", 50, size=10), word("TM", 77, size=6)))

    assert lines[0].font_size == 10


# ------------------------------------------------------------------ inline icons


def icon(x0: float, top: float = 97, size: float = 18, page: int = 1) -> ImageBox:
    return ImageBox(pdf_page=page, x0=x0, top=top, x1=x0 + size, bottom=top + size)


# "hold" spans 50-70; the gap up to "the" at 92 is where the button glyph was.
HOLD = (word("hold", 50), word("the", 92))


def test_an_image_in_the_gap_between_words_becomes_an_icon_placeholder() -> None:
    lines = group_lines(HOLD, (icon(72),))

    assert texts(lines) == [f"hold {ICON} the"]
    assert lines[0].top == 100  # the placeholder keeps the line's own box
    assert lines[0].bottom == 110
    assert count_icons(lines) == 1


def test_a_lone_icon_in_a_table_cell_gap_is_its_own_segment() -> None:
    lines = group_lines((word("Power", 50), word("Turn", 300)), (icon(200),))

    assert texts(lines) == ["Power", ICON, "Turn"]


def test_images_at_the_start_or_end_of_a_line_get_no_placeholder() -> None:
    """Margin symbols beside notes sit before the text; nothing marks a lost glyph."""
    lines = group_lines(HOLD, (icon(28), icon(72), icon(114)))

    assert texts(lines) == [f"hold {ICON} the"]
    assert count_icons(lines) == 1


def test_images_outside_text_lines_get_no_placeholder() -> None:
    figure = ImageBox(pdf_page=1, x0=60, top=55, x1=400, bottom=155)  # centred, too tall
    logo = icon(72, top=300)  # no words at its height
    other_page = icon(72, page=2)

    lines = group_lines((*HOLD, word("x", 50, page=2)), (figure, logo, other_page, icon(72)))

    assert texts(lines) == [f"hold {ICON} the", "x"]


def test_each_gap_image_counts_once() -> None:
    words = (word("a", 50), word("b", 79), word("c", 108))

    lines = group_lines(words, (icon(58), icon(87)))

    assert texts(lines) == [f"a {ICON} b {ICON} c"]
    assert count_icons(lines) == 2


def test_a_tall_image_centred_in_a_gap_between_words_is_a_figure_not_an_icon() -> None:
    """Safety guard: a picture placed between words must never become [icon]."""
    words = (*HOLD, word("x", 200))
    figure = ImageBox(pdf_page=1, x0=120, top=55, x1=180, bottom=155)  # 100 pt > 3 em

    lines = group_lines(words, (figure, icon(72)))

    assert texts(lines) == [f"hold {ICON} the", "x"]
    assert count_icons(lines) == 1
