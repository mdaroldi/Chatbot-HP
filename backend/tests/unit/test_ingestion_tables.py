"""Behavior of the row-band table extractor (synthetic lines and rules)."""

from app.domain.ingestion import HorizontalRule, Table, TextLine
from app.ingestion.tables import extract_tables, merge_rules


def cell(text: str, x0: float, top: float, size: float = 8, page: int = 1) -> TextLine:
    return TextLine(
        text=text, pdf_page=page, font_size=size, x0=x0, top=top, x1=x0 + 30, bottom=top + size
    )


def rules(*ys: float, x0: float = 50, x1: float = 500, page: int = 1) -> list[HorizontalRule]:
    return [HorizontalRule(pdf_page=page, y=y, x0=x0, x1=x1) for y in ys]


def bordered_table(page: int = 1, top: float = 100) -> tuple[list[TextLine], list[HorizontalRule]]:
    """Top border, header band, then one row per band; the first row wraps its last cell."""
    lines = [
        cell("Table 1-1 Parts", 50, top - 12, size=9, page=page),  # caption: never a row
        cell("Item", 55, top + 5, page=page),
        cell("Component", 100, top + 5, page=page),
        cell("Part number", 400, top + 5, page=page),
        cell("(1)", 55, top + 25, page=page),
        cell("Display", 100, top + 25, page=page),
        cell("not available", 400, top + 25, page=page),
        cell("as a part", 400, top + 36, page=page),
        cell("(2)", 55, top + 55, page=page),
        cell("Bottom cover", 100, top + 55, page=page),
        cell("X1-001", 400, top + 55, page=page),
    ]
    return lines, rules(top, top + 20, top + 50, top + 80, page=page)


BORDERED = Table(
    pdf_page=1,
    top=100,
    bottom=180,
    x0=50,
    x1=500,
    header=("Item", "Component", "Part number"),
    rows=(("(1)", "Display", "not available as a part"), ("(2)", "Bottom cover", "X1-001")),
)


# ------------------------------------------------------------------ rule merging


def test_collinear_touching_segments_merge_into_one_rule() -> None:
    segments = [
        HorizontalRule(pdf_page=1, y=100.3, x0=300, x1=500),
        HorizontalRule(pdf_page=1, y=100, x0=50, x1=101),
        HorizontalRule(pdf_page=1, y=100, x0=100, x1=300),
        HorizontalRule(pdf_page=1, y=100, x0=560, x1=600),  # same y, not touching
        HorizontalRule(pdf_page=1, y=110, x0=50, x1=500),
        HorizontalRule(pdf_page=2, y=100, x0=500, x1=520),
    ]

    assert merge_rules(segments) == (
        HorizontalRule(pdf_page=1, y=100, x0=50, x1=500),
        HorizontalRule(pdf_page=1, y=100, x0=560, x1=600),
        HorizontalRule(pdf_page=1, y=110, x0=50, x1=500),
        HorizontalRule(pdf_page=2, y=100, x0=500, x1=520),
    )


# ------------------------------------------------------------------ tables


def test_header_band_gives_columns_and_each_band_is_one_row() -> None:
    lines, grid = bordered_table()

    assert extract_tables(lines, grid) == (BORDERED,)


def test_header_above_the_first_rule_is_taken_when_it_has_the_table_font_size() -> None:
    """Some tables have no top border: the header sits right above the first rule."""
    lines = [
        cell("Summary of codes:", 60, 170, size=10),  # intro: different size, too far
        cell("Code", 60, 205),
        cell("Meaning", 140, 205),
        cell("E0", 60, 225),
        cell("Bad unit", 140, 225),
        cell("E1", 60, 245),
        cell("Jam", 140, 245),
    ]

    assert extract_tables(lines, rules(220, 240, 260, x0=55)) == (
        Table(
            pdf_page=1,
            top=205,
            bottom=260,
            x0=55,
            x1=500,
            header=("Code", "Meaning"),
            rows=(("E0", "Bad unit"), ("E1", "Jam")),
        ),
    )


def test_a_same_size_line_far_above_the_first_rule_is_not_a_header() -> None:
    lines, grid = bordered_table()
    far = cell("Earlier text", 55, 60)

    assert extract_tables([far, *lines], grid) == (BORDERED,)


def test_short_rules_such_as_link_underlines_do_not_split_rows() -> None:
    lines, grid = bordered_table()
    underline = HorizontalRule(pdf_page=1, y=133, x0=100, x1=160)

    assert extract_tables(lines, [*grid, underline]) == (BORDERED,)


def test_fewer_than_three_rules_of_one_width_are_not_a_table() -> None:
    lines, grid = bordered_table()
    pair = rules(400, 420, x0=60, x1=300)
    stray = [cell("Note", 70, 405), cell("text", 200, 405)]

    assert extract_tables([*lines, *stray], [*grid, *pair]) == (BORDERED,)


def test_empty_bands_end_a_table_instead_of_widening_it() -> None:
    lines, grid = bordered_table()

    assert extract_tables(lines, [*rules(97), *grid, *rules(182)]) == (BORDERED,)


def test_a_band_of_body_size_text_splits_tables_of_the_same_width() -> None:
    """Tables are set below the 10 pt body size; a paragraph between them is no row."""
    first_lines, first_grid = bordered_table(top=100)
    second_lines, second_grid = bordered_table(top=300)
    paragraph = [cell("Body paragraph between tables", 55, 230, size=10)]

    tables = extract_tables([*first_lines, *paragraph, *second_lines], [*first_grid, *second_grid])

    assert [(t.top, t.bottom, t.header) for t in tables] == [
        (100, 180, BORDERED.header),
        (300, 380, BORDERED.header),
    ]


def test_a_band_with_any_body_size_line_splits_tables_even_when_captions_dominate() -> None:
    """A heading and intro over the next table's caption: mostly 9 pt, still no row."""
    first_lines, first_grid = bordered_table(top=100)
    second_lines, second_grid = bordered_table(top=300)
    # 4 + 10 characters above 9 pt, 15 in the 9 pt caption "Table 1-1 Parts".
    between = [
        cell("Keys", 55, 200, size=14),
        cell("Find keys.", 55, 220, size=10),
    ]

    tables = extract_tables([*first_lines, *between, *second_lines], [*first_grid, *second_grid])

    assert [(t.top, t.bottom, t.header) for t in tables] == [
        (100, 180, BORDERED.header),
        (300, 380, BORDERED.header),
    ]


def test_a_band_of_table_size_lines_only_is_a_row() -> None:
    """Cells (8 pt) next to a caption-size line (9 pt) stay one row."""
    lines, grid = bordered_table()
    note = cell("see note", 100, 167, size=9)

    tables = extract_tables([*lines, note], grid)

    assert [row[1] for row in tables[0].rows] == ["Display", "Bottom cover see note"]


def test_boxed_notes_in_body_size_are_not_tables() -> None:
    lines, grid = bordered_table()
    notes = [
        cell("NOTE: first boxed note", 55, 410, size=10),
        cell("Body text between notes", 55, 500, size=10),
        cell("TIP: second boxed note", 55, 610, size=10),
    ]

    assert extract_tables([*lines, *notes], [*grid, *rules(400, 440, 600, 640)]) == (BORDERED,)


def test_a_rule_between_a_line_and_the_table_blocks_it_from_the_header() -> None:
    """The last row of a table just above must not become the next table's header."""
    lines = [
        cell("tail of a row", 60, 280),
        cell("Code", 60, 305),
        cell("Meaning", 140, 305),
        cell("E0", 60, 325),
        cell("Bad unit", 140, 325),
    ]
    other_table_bottom = HorizontalRule(pdf_page=1, y=292, x0=40, x1=520)

    tables = extract_tables(lines, [other_table_bottom, *rules(300, 320, 340, x0=55)])

    assert [(t.top, t.header, t.rows) for t in tables] == [
        (300, ("Code", "Meaning"), (("E0", "Bad unit"),))
    ]


def test_text_beside_the_table_is_not_part_of_it() -> None:
    lines, grid = bordered_table()
    margin = cell("Margin", 510, 130)

    assert extract_tables([*lines, margin], grid) == (BORDERED,)


def test_cells_snap_to_the_nearest_column_start_on_their_left() -> None:
    lines, grid = bordered_table()
    nudged = [*lines[:5], cell("Display", 98.5, 125), *lines[6:]]
    leading = cell("*", 51, 160)  # left of the first column: belongs to it

    rows = extract_tables([*nudged, leading], grid)[0].rows

    assert rows == (
        ("(1)", "Display", "not available as a part"),
        ("(2) *", "Bottom cover", "X1-001"),
    )


def test_wrapped_header_cells_join_and_rows_read_top_to_bottom() -> None:
    lines = [
        cell("Spare part", 300, 105),
        cell("Item", 55, 105),
        cell("number", 300, 113),
        cell("b", 55, 135),
        cell("a", 55, 125),
        cell("X", 300, 125),
    ]

    table = extract_tables(lines, rules(100, 120, 150))[0]

    assert table.header == ("Item", "Spare part number")
    assert table.rows == (("a b", "X"),)


def test_tables_are_found_per_page_and_per_rule_width() -> None:
    first_lines, first_grid = bordered_table(page=2)
    second_lines, second_grid = bordered_table(page=1)
    narrow = [
        cell("Key", 60, 405),
        cell("Value", 150, 405),
        cell("k", 60, 425),
        cell("v", 150, 425),
    ]

    tables = extract_tables(
        [*first_lines, *second_lines, *narrow],
        [*first_grid, *second_grid, *rules(400, 420, 440, x0=55, x1=300)],
    )

    assert [(t.pdf_page, t.top, t.header) for t in tables] == [
        (1, 100, BORDERED.header),
        (1, 400, ("Key", "Value")),
        (2, 100, BORDERED.header),
    ]


def test_rules_without_text_between_them_are_not_a_table() -> None:
    lines, grid = bordered_table()

    assert extract_tables(lines, [*grid, *rules(600, 620, 640)]) == (BORDERED,)
