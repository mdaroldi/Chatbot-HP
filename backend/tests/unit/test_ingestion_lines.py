"""Behavior of word-to-line grouping (synthetic words, no PDF)."""

from app.domain.ingestion import TextLine, Word
from app.ingestion.lines import group_lines


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
