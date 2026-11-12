"""Behavior of the parser output types: words, rules, tables and the parsed document."""

import pytest

from app.domain.ingestion import (
    CorpusDocument,
    HorizontalRule,
    OutlineEntry,
    ParsedDocument,
    Table,
    Word,
)

DOC = CorpusDocument(
    doc_id="demo-guide",
    title="Demo Guide",
    product="Demo Printer 100",
    url="https://docs.test/a.pdf",
    sha256="a" * 64,
)


def word(page: int = 1, text: str = "Demo") -> Word:
    return Word(text=text, pdf_page=page, font_size=10, x0=1, top=2, x1=3, bottom=4)


def test_word_rejects_blank_text_bad_page_size_and_geometry() -> None:
    assert word().text == "Demo"
    with pytest.raises(ValueError, match="text"):
        Word(text=" ", pdf_page=1, font_size=10, x0=0, top=0, x1=1, bottom=1)
    with pytest.raises(ValueError, match="pdf_page"):
        Word(text="x", pdf_page=0, font_size=10, x0=0, top=0, x1=1, bottom=1)
    with pytest.raises(ValueError, match="font_size"):
        Word(text="x", pdf_page=1, font_size=0, x0=0, top=0, x1=1, bottom=1)
    with pytest.raises(ValueError, match="geometry"):
        Word(text="x", pdf_page=1, font_size=10, x0=0, top=2, x1=1, bottom=1)


def test_horizontal_rule_needs_a_page_and_a_positive_length() -> None:
    assert HorizontalRule(pdf_page=2, y=10, x0=1, x1=5).x1 == 5
    with pytest.raises(ValueError, match="pdf_page"):
        HorizontalRule(pdf_page=0, y=10, x0=1, x1=5)
    with pytest.raises(ValueError, match="length"):
        HorizontalRule(pdf_page=1, y=10, x0=5, x1=5)


def test_table_rows_have_one_cell_per_header_column() -> None:
    table = Table(
        pdf_page=3, top=10, bottom=50, x0=0, x1=100, header=("Code", "Meaning"), rows=(("A", "B"),)
    )

    assert table.rows == (("A", "B"),)
    with pytest.raises(ValueError, match="header"):
        Table(pdf_page=3, top=10, bottom=50, x0=0, x1=100, header=(), rows=())
    with pytest.raises(ValueError, match="cells"):
        Table(pdf_page=3, top=10, bottom=50, x0=0, x1=100, header=("Code",), rows=(("A", "B"),))
    with pytest.raises(ValueError, match="box"):
        Table(pdf_page=3, top=50, bottom=10, x0=0, x1=100, header=("Code",), rows=())


def test_parsed_document_counts_pages_from_their_heights() -> None:
    parsed = ParsedDocument(
        document=DOC,
        words=(word(page=2),),
        rules=(HorizontalRule(pdf_page=2, y=1, x0=0, x1=1),),
        outline=(OutlineEntry(level=1, title="Setup", pdf_page=2),),
        page_heights=(792.0, 792.0),
    )

    assert parsed.page_count == 2
    assert parsed.page_height(2) == 792.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"page_heights": ()}, "at least one page"),
        ({"page_heights": (792.0, 0.0)}, "page heights"),
        ({"words": (word(page=3),)}, "word on page 3"),
        ({"rules": (HorizontalRule(pdf_page=3, y=1, x0=0, x1=1),)}, "rule on page 3"),
        ({"outline": (OutlineEntry(level=1, title="Setup", pdf_page=3),)}, "bookmark to page 3"),
    ],
)
def test_parsed_document_rejects_content_beyond_its_pages(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "document": DOC,
        "words": (),
        "rules": (),
        "outline": (),
        "page_heights": (792.0, 792.0),
    }
    with pytest.raises(ValueError, match=message):
        ParsedDocument(**(values | overrides))  # type: ignore[arg-type]  # test data
