"""Behavior of the cleaning stage: what it drops, why, and how it counts (synthetic data)."""

from collections.abc import Sequence

from app.domain.ingestion import DropReason, OutlineEntry, TextLine
from app.ingestion.cleaning import CleanedDocument, clean

HEIGHT = 800.0  # footer zone starts at 744
FRENCH = "Le produit doit être utilisé dans un local fermé et"
ZERO = dict.fromkeys(DropReason, 0)


def line(
    text: str,
    page: int = 1,
    *,
    top: float = 100,
    size: float = 10,
    x0: float = 50,
    x1: float = 500,
) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=size, x0=x0, top=top, x1=x1, bottom=top + size
    )


def heading(title: str, page: int = 1, top: float = 60) -> TextLine:
    return line(title, page, top=top, size=14)


def entry(title: str, level: int = 1, page: int = 1) -> OutlineEntry:
    return OutlineEntry(level=level, title=title, pdf_page=page)


def run(
    lines: Sequence[TextLine], outline: Sequence[OutlineEntry], pages: int = 1
) -> CleanedDocument:
    return clean(lines, outline, (HEIGHT,) * pages)


def kept(result: CleanedDocument) -> list[str]:
    return [ln.text for ln in result.lines]


def dropped(result: CleanedDocument, reason: DropReason) -> list[str]:
    return [d.line.text for d in result.dropped if d.reason is reason]


# ------------------------------------------------------------------ navigation and footers


def test_footers_are_dropped_and_their_page_numbers_kept() -> None:
    lines = (
        heading("Setup"),
        line("Connect the cable to the port."),
        line("6", top=760, x1=60),
        line("Setup", top=760, x0=80, x1=120),
    )

    result = run(lines, (entry("Setup"),))

    assert kept(result) == ["Setup", "Connect the cable to the port."]
    assert dropped(result, DropReason.FOOTER) == ["6", "Setup"]
    assert result.footers[1].printed_page == 6
    assert result.stats.lines == {**ZERO, DropReason.FOOTER: 2}


def test_toc_and_index_pages_are_dropped_whole() -> None:
    lines = (
        line("Contents", 1, top=60),
        *(line(f"Topic {n} ........ {n}", 1, top=100 + 20 * n) for n in range(3)),
        heading("Setup", 2),
        line("Connect the cable to the port.", 2),
        heading("Index", 3),
        line("cable, connect 2", 3),
        line("port 2", 3, top=120),
    )

    result = run(lines, (entry("Setup", page=2), entry("Index", page=3)), pages=3)

    assert kept(result) == ["Setup", "Connect the cable to the port."]
    assert len(dropped(result, DropReason.TOC)) == 4
    assert dropped(result, DropReason.INDEX) == ["Index", "cable, connect 2", "port 2"]
    assert result.stats.pages == {DropReason.TOC: 1, DropReason.INDEX: 1}
    assert result.stats.empty_sections == 0  # the index section went as index


def test_chapter_opener_lists_are_dropped_and_the_intro_kept() -> None:
    def item(text: str, x0: float, top: float) -> TextLine:
        return line(text, top=top, x0=x0, x1=x0 + 40)

    lines = (
        line("Setup", top=60, size=24),
        line("The printer needs a few steps before first use.", top=150),
        line("This chapter covers:", top=180),
        *(item("●", 100, 200), item("Connect", 122, 200)),
        *(item("●", 100, 220), item("Panel", 122, 220)),
        heading("Connect", 2),
        line("Connect the cable to the port.", 2),
        heading("Panel", 3),
        line("The panel has four buttons.", 3),
    )
    outline = (entry("Setup"), entry("Connect", 2, page=2), entry("Panel", 2, page=3))

    result = run(lines, outline, pages=3)

    assert dropped(result, DropReason.CHAPTER_TOC) == [
        "This chapter covers:",
        "●",
        "Connect",
        "●",
        "Panel",
    ]
    assert kept(result)[:2] == ["Setup", "The printer needs a few steps before first use."]
    assert len(kept(result)) == 6


# ------------------------------------------------------------------ other languages


def test_non_latin_lines_are_dropped_and_greek_symbols_kept() -> None:
    lines = (
        heading("Notices"),
        line("Use the supplied power cord.", top=100),
        line("本机支持无线打印", top=130),
        line("Use a 1 MΩ ±10 % resistance strap.", top=160),
    )

    result = run(lines, (entry("Notices"),))

    assert dropped(result, DropReason.NON_LATIN) == ["本机支持无线打印"]
    assert kept(result) == ["Notices", "Use the supplied power cord.", lines[3].text]


def test_non_english_lines_are_dropped_and_english_kept() -> None:
    lines = (heading("Notices"), line(FRENCH, top=100), line("Keep the device dry.", top=130))

    result = run(lines, (entry("Notices"),))

    assert dropped(result, DropReason.NON_ENGLISH_LATIN) == [FRENCH]
    assert kept(result) == ["Notices", "Keep the device dry."]


def test_each_line_is_counted_once_under_the_first_rule() -> None:
    lines = (
        line("本机支持无线打印", 1, top=60),
        *(line(f"Topic {n} ........ {n}", 1, top=100 + 20 * n) for n in range(3)),
        heading("Setup", 2),
        line("本机支持无线打印", 2),
    )

    result = run(lines, (entry("Setup", page=2),), pages=2)

    assert result.stats.lines[DropReason.TOC] == 4
    assert dropped(result, DropReason.NON_LATIN) == ["本机支持无线打印"]
    assert len(result.dropped) == 6  # with the emptied "Setup" heading


# ------------------------------------------------------------------ paragraph tails


def test_a_paragraph_tail_after_a_dropped_line_is_dropped_too() -> None:
    """Tails have too few words to be judged; the block they continue decides."""
    lines = (
        heading("Notices"),
        line(FRENCH, top=100),
        line("sec.", top=112),
        line("loin du mur.", top=124),  # chained: continues a dropped tail
        line("Keep the device dry.", top=150),
    )

    result = run(lines, (entry("Notices"),))

    assert dropped(result, DropReason.NON_ENGLISH_LATIN) == [FRENCH, "sec.", "loin du mur."]
    assert kept(result) == ["Notices", "Keep the device dry."]


def test_lines_outside_the_dropped_block_are_kept() -> None:
    lines = (
        heading("Notices"),
        line(FRENCH, top=100),
        line("sec.", top=112),  # tail: dropped
        line(FRENCH, top=140),
        line("and the cable is in the box.", top=152),  # English, same block
        line(FRENCH, top=180),
        line("Wi-Fi Direct", top=205),  # after a paragraph gap
        line(FRENCH, top=240),
        line("Wi-Fi Direct", top=252, size=8),  # other font size
    )

    result = run(lines, (entry("Notices"),))

    assert dropped(result, DropReason.NON_ENGLISH_LATIN) == [FRENCH, "sec.", *[FRENCH] * 3]
    assert kept(result) == [
        "Notices",
        "and the cable is in the box.",
        "Wi-Fi Direct",
        "Wi-Fi Direct",
    ]


def test_the_block_predecessor_is_the_line_above_in_the_same_column() -> None:
    """Reading order interleaves columns; a tail continues the line above it."""
    lines = (
        heading("Notices"),
        line(FRENCH, top=300, x1=250),
        line("Press the button.", top=300, x0=300),
        line("sec.", top=312, x1=250),  # below the French line: a tail
        line("Wi-Fi Direct", top=312, x0=300),  # below the English line: kept
    )

    result = run(lines, (entry("Notices"),))

    assert dropped(result, DropReason.NON_ENGLISH_LATIN) == [FRENCH, "sec."]
    assert kept(result) == ["Notices", "Press the button.", "Wi-Fi Direct"]


# ------------------------------------------------------------------ empty sections


def test_a_section_left_without_text_is_dropped_as_empty() -> None:
    lines = (
        heading("Notices"),
        line("Read these notices.", top=80),
        heading("Taiwan", top=120),
        line("本机支持无线打印", top=150),
        heading("日本", top=180),  # the heading itself goes as non_latin
        line("本机支持无线打印", top=210),
        heading("Canada", top=240),
        line("Keep the device dry.", top=270),
    )
    outline = (entry("Notices"), entry("Taiwan", 2), entry("日本", 2), entry("Canada", 2))

    result = run(lines, outline)

    assert dropped(result, DropReason.EMPTY_AFTER_CLEANING) == ["Taiwan"]
    assert result.stats.empty_sections == 1
    assert kept(result) == ["Notices", "Read these notices.", "Canada", "Keep the device dry."]


def test_icon_placeholders_are_not_text_for_the_empty_check() -> None:
    lines = (
        heading("Buttons"),
        line("[icon] [icon]", top=80),
        heading("Lights", top=120),
        line("Press [icon] to start.", top=150),
    )

    result = run(lines, (entry("Buttons"), entry("Lights")))

    assert dropped(result, DropReason.EMPTY_AFTER_CLEANING) == ["Buttons", "[icon] [icon]"]
    assert kept(result) == ["Lights", "Press [icon] to start."]


def test_a_heading_only_section_is_empty_and_its_label_goes_with_it() -> None:
    def chapter(text: str, x0: float, page: int) -> TextLine:
        return line(text, page, top=82, size=24, x0=x0, x1=x0 + 60)

    lines = (
        line("Printer guide", 1, top=300),  # cover: before the first heading
        chapter("1", 54, 2),
        chapter("Setup", 99, 2),
        heading("Connect", 2, top=200),
        line("Connect the cable to the port.", 2, top=230),
        chapter("2", 54, 3),
        chapter("Print", 99, 3),
        line("Print from any app.", 3, top=200),
    )
    outline = (entry("Setup", page=2), entry("Connect", 2, page=2), entry("Print", page=3))

    result = run(lines, outline, pages=3)

    assert dropped(result, DropReason.EMPTY_AFTER_CLEANING) == ["1", "Setup"]
    assert kept(result) == [
        "Printer guide",
        "Connect",
        "Connect the cable to the port.",
        "2",
        "Print",
        "Print from any app.",
    ]


def test_spans_index_the_kept_lines_and_skip_dropped_headings() -> None:
    lines = (
        heading("Notices"),
        line("本机支持无线打印", top=80),
        line("Read these notices.", top=100),
        heading("Taiwan", top=120),
        line("本机支持无线打印", top=150),
        heading("Canada", top=240),
        line("Keep the device dry.", top=270),
    )
    outline = (entry("Notices"), entry("Taiwan", 2), entry("Canada", 2))

    result = run(lines, outline)

    assert kept(result) == ["Notices", "Read these notices.", "Canada", "Keep the device dry."]
    assert [(s.entry, s.start, s.body, s.end) for s in result.spans] == [(0, 0, 1, 2), (2, 2, 3, 4)]
