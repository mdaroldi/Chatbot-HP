"""The test PDF builder writes a valid cross-reference table, so parsers never repair it."""

import re

from tests.pdf_builder import Bookmark, Page, Text, build_pdf


def test_every_xref_offset_points_at_its_object() -> None:
    pdf = build_pdf(
        (Page(texts=(Text(50, 50, 10, "Hello (world)"),)), Page()),
        (Bookmark("One", 1), Bookmark("Two", 2, level=2, style="named")),
    )

    startxref = int(re.findall(rb"startxref\n(\d+)\n", pdf)[-1])
    assert pdf[startxref:].startswith(b"xref\n")
    entries = re.findall(rb"(\d{10}) 00000 n \n", pdf[startxref:])
    assert len(entries) >= 8
    for number, offset in enumerate(entries, start=1):
        assert pdf[int(offset) :].startswith(f"{number} 0 obj\n".encode())
