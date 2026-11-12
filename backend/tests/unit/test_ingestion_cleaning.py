"""Behavior of the cleaning stage: what it drops, why, and how it counts (synthetic data)."""

from collections.abc import Sequence

from app.domain.ingestion import DropReason, HorizontalRule, OutlineEntry, TextLine
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
    lines: Sequence[TextLine],
    outline: Sequence[OutlineEntry],
    pages: int = 1,
    rules: Sequence[HorizontalRule] = (),
) -> CleanedDocument:
    return clean(lines, outline, (HEIGHT,) * pages, rules)


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


# ------------------------------------------------------------------ section mini-TOCs


def bullet(text: str, top: float, page: int = 1) -> tuple[TextLine, TextLine]:
    """A hanging bullet: the marker and its text are separate segments."""
    return line("●", page, top=top, x0=100, x1=105), line(text, page, top=top, x0=122, x1=400)


def toc_section(*items: str) -> tuple[TextLine, ...]:
    """Section "Buttons" on page 1: intro, lead-in, the items; children on page 2."""
    return (
        heading("Buttons"),
        line("The printer has buttons on two sides.", top=100),
        line("This section covers:", top=130),
        *(part for n, text in enumerate(items) for part in bullet(text, 150 + 20 * n)),
        heading("Panel buttons", 2),
        line("The panel has four buttons.", 2),
        heading("Back buttons", 2, top=200),
        line("The back has the power button.", 2, top=230),
    )


TOC_OUTLINE = (
    entry("Buttons", 2),
    entry("Panel buttons", 3, page=2),
    entry("Back buttons", 3, page=2),
)


def test_a_list_naming_only_child_sections_is_dropped_with_its_lead_in() -> None:
    result = run(toc_section("Panel buttons", "Back  buttons"), TOC_OUTLINE, pages=2)

    assert dropped(result, DropReason.SECTION_TOC) == [
        "This section covers:",
        "●",
        "Panel buttons",
        "●",
        "Back  buttons",
    ]
    assert kept(result)[:2] == ["Buttons", "The printer has buttons on two sides."]
    assert result.stats.lines[DropReason.SECTION_TOC] == 5


def test_a_list_with_any_item_that_is_not_a_child_title_is_kept_whole() -> None:
    control = run(toc_section("Panel buttons", "Back buttons"), TOC_OUTLINE, pages=2)
    assert len(dropped(control, DropReason.SECTION_TOC)) == 5

    for items in (
        ("Panel buttons", "www.example.com/buttons"),  # a URL
        ("Panel buttons", "Back buttons: power and Wi-Fi"),  # extra text after a title
        ("Panel buttons", "Buttons"),  # the section's own title
    ):
        result = run(toc_section(*items), TOC_OUTLINE, pages=2)

        assert dropped(result, DropReason.SECTION_TOC) == []
        assert "This section covers:" in kept(result)


def test_a_single_item_list_is_kept() -> None:
    pair = run(toc_section("Panel buttons", "Back buttons"), TOC_OUTLINE, pages=2)
    single = run(toc_section("Panel buttons"), TOC_OUTLINE, pages=2)

    assert len(dropped(pair, DropReason.SECTION_TOC)) == 5
    assert dropped(single, DropReason.SECTION_TOC) == []


def test_titles_of_grandchildren_or_siblings_do_not_count_as_children() -> None:
    children = (
        entry("Buttons", 2),
        entry("Front", 3, page=2),
        entry("Panel buttons", 3, page=2),
        entry("Back buttons", 3, page=2),
    )
    outline = (
        entry("Buttons", 2),
        entry("Front", 3, page=2),
        entry("Panel buttons", 4, page=2),  # a grandchild
        entry("Back buttons", 2, page=2),  # a sibling
    )
    lines = (
        *toc_section("Panel buttons", "Back buttons")[:-4],
        heading("Front", 2, top=60),
        heading("Panel buttons", 2, top=120),
        line("The panel has four buttons.", 2, top=150),
        heading("Back buttons", 2, top=200),
        line("The back has the power button.", 2, top=230),
    )

    control, result = run(lines, children, pages=2), run(lines, outline, pages=2)

    assert len(dropped(control, DropReason.SECTION_TOC)) == 5
    assert dropped(result, DropReason.SECTION_TOC) == []


def test_an_item_wrapped_onto_a_second_line_does_not_match() -> None:
    """No corpus item wraps; a wrapped one fails to match, which keeps its list."""

    def items(first: str, second: str | None) -> tuple[TextLine, ...]:
        wrapped = (line(second, top=162, x0=122, x1=160),) if second else ()
        return (
            heading("Buttons"),
            line("The printer has buttons on two sides.", top=100),
            *bullet(first, 150),
            *wrapped,
            *bullet("Back buttons", 180),
            *toc_section()[-4:],
        )

    one_line = run(items("Panel buttons", None), TOC_OUTLINE, pages=2)
    wrapped = run(items("Panel", "buttons"), TOC_OUTLINE, pages=2)

    assert dropped(one_line, DropReason.SECTION_TOC) == ["●", "Panel buttons", "●", "Back buttons"]
    assert dropped(wrapped, DropReason.SECTION_TOC) == []


def test_a_section_left_with_only_its_mini_toc_is_dropped_as_empty() -> None:
    lines = (
        heading("Buttons"),
        *bullet("Panel buttons", 150),
        *bullet("Back buttons", 170),
        *toc_section()[-4:],
    )

    result = run(lines, TOC_OUTLINE, pages=2)

    assert len(dropped(result, DropReason.SECTION_TOC)) == 4
    assert dropped(result, DropReason.EMPTY_AFTER_CLEANING) == ["Buttons"]
    assert kept(result)[0] == "Panel buttons"


# ------------------------------------------------------------------ non-Latin tables


def cell(text: str, x0: float, top: float) -> TextLine:
    return line(text, top=top, size=8, x0=x0, x1=x0 + 40)


def substance_table(
    label: str, marks: tuple[str, str]
) -> tuple[list[TextLine], list[HorizontalRule]]:
    """Header band, then one row: a label cell and two mark cells."""
    lines = [
        cell("部件名称", 55, 105),
        cell("铅 (Pb)", 200, 105),
        cell("汞 (Hg)", 300, 105),
        cell(label, 55, 125),
        cell(marks[0], 200, 125),
        cell(marks[1], 300, 125),
    ]
    grid = [HorizontalRule(pdf_page=1, y=y, x0=50, x1=500) for y in (100, 120, 140)]
    return lines, grid


def with_section(table: Sequence[TextLine]) -> tuple[TextLine, ...]:
    return (heading("Substances"), line("The table lists the substances.", top=80), *table)


def test_a_table_left_with_symbols_and_marks_only_is_dropped() -> None:
    table, grid = substance_table("印刷电路板", ("X", "◯"))

    result = run(with_section(table), (entry("Substances"),), rules=grid)

    assert dropped(result, DropReason.NON_LATIN) == ["部件名称", "印刷电路板"]
    assert dropped(result, DropReason.NON_LATIN_TABLE) == ["铅 (Pb)", "汞 (Hg)", "X", "◯"]
    assert kept(result) == ["Substances", "The table lists the substances."]


def test_a_table_that_keeps_a_latin_word_is_kept() -> None:
    for label, residue in (("PCB", 5), ("Cr", 5), ("Print engine", 0), ("Câble", 0)):
        table, grid = substance_table(label, ("X", "◯"))

        result = run(with_section(table), (entry("Substances"),), rules=grid)

        assert len(dropped(result, DropReason.NON_LATIN_TABLE)) == residue, label
        assert (label in kept(result)) == (residue == 0), label


def test_a_table_without_non_latin_lines_is_never_a_candidate() -> None:
    """Part numbers, numbers and acronyms hold no Latin word: only the lost non-Latin
    header line makes the first table a candidate.
    """
    table, grid = substance_table("N12345-001", ("2", "4"))
    symbols = [cell("PBB", 200, 105), cell("PBDE", 300, 105), *table[3:]]

    candidate = run(with_section([table[0], *symbols]), (entry("Substances"),), rules=grid)
    latin = run(with_section([cell("No.", 55, 105), *symbols]), (entry("Substances"),), rules=grid)

    assert dropped(candidate, DropReason.NON_LATIN_TABLE) == ["PBB", "PBDE", "N12345-001", "2", "4"]
    assert dropped(latin, DropReason.NON_LATIN_TABLE) == []
    assert kept(latin)[2:] == ["No.", "PBB", "PBDE", "N12345-001", "2", "4"]


def test_lines_outside_the_table_box_are_not_table_residue() -> None:
    table, grid = substance_table("印刷电路板", ("X", "◯"))
    below = line("X", top=300, size=8, x0=200, x1=240)  # under the last rule

    result = run((*with_section(table), below), (entry("Substances"),), rules=grid)

    assert "X" in kept(result)
    assert len(dropped(result, DropReason.NON_LATIN_TABLE)) == 4


# ------------------------------------------------------------------ URL-only sections


def test_a_section_with_only_a_url_is_dropped_as_empty() -> None:
    lines = (
        heading("Recycling"),
        line("www.example.com/recycle", top=80),
        heading("Paper", top=120),
        line("See https://example.com/paper for details.", top=150),
    )

    result = run(lines, (entry("Recycling"), entry("Paper")))

    assert dropped(result, DropReason.EMPTY_AFTER_CLEANING) == [
        "Recycling",
        "www.example.com/recycle",
    ]
    assert kept(result) == ["Paper", "See https://example.com/paper for details."]
