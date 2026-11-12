"""Thin pdfplumber adapter: PDF objects in, plain IR out (ADR 0004).

Every layout decision lives in the pure stages under app/ingestion; this module
only converts words, horizontal lines, image boxes, page sizes and bookmarks.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFDestinationNotFound, PDFDocument, PDFNoOutlines
from pdfminer.pdftypes import PDFObjRef, resolve1
from pdfminer.psparser import LIT

from app.domain.ingestion import (
    CorpusDocument,
    DocumentParseError,
    HorizontalRule,
    ImageBox,
    OutlineEntry,
    ParsedDocument,
    Word,
)

_GOTO = LIT("GoTo")
_FLAT = 0.5  # a line whose ends differ by less than this vertically is horizontal
# pdfminer derives font sizes from glyph boxes, leaving float noise (8.000000000000057);
# rounding makes equal sizes compare equal in the size-based layout rules.
_SIZE_DECIMALS = 2


def _target_page(
    dest: Any,  # Any: pdfminer outline items carry untyped PDF objects
    action: Any,  # Any: as above
    pdf: PDFDocument,
    pages: Mapping[object, int],
) -> int | None:
    """Resolve the bookmark styles seen in the corpus: a direct destination, a GoTo
    action with an explicit destination, or a GoTo action with a named one.
    """
    target = dest
    goto = resolve1(action) or {}
    if target is None and goto.get("S") == _GOTO:
        target = goto.get("D")
    target = resolve1(target)
    if isinstance(target, bytes | str):
        try:
            target = resolve1(pdf.get_dest(target))
        except PDFDestinationNotFound:
            return None
    if isinstance(target, dict):
        target = resolve1(target.get("D"))
    if isinstance(target, list) and target and isinstance(target[0], PDFObjRef):
        return pages.get(target[0].objid)
    return None


def _outline(
    pdf: PDFDocument, pages: Mapping[object, int], doc_id: str
) -> tuple[OutlineEntry, ...]:
    try:
        items = list(pdf.get_outlines())
    except PDFNoOutlines as exc:
        raise DocumentParseError(f"{doc_id}: no bookmarks") from exc
    entries: list[OutlineEntry] = []
    for number, (level, title, dest, action, _) in enumerate(items, start=1):
        page = _target_page(dest, action, pdf, pages)
        if page is None:
            raise DocumentParseError(f"{doc_id}: bookmark {number} does not resolve to a page")
        entries.append(OutlineEntry(level=level, title=title, pdf_page=page))
    return tuple(entries)


class PdfplumberParser:
    """DocumentParser on pdfplumber; CPU-bound, so async callers run it in a thread."""

    def parse(self, path: Path, document: CorpusDocument) -> ParsedDocument:
        words: list[Word] = []
        rules: list[HorizontalRule] = []
        images: list[ImageBox] = []
        heights: list[float] = []
        with pdfplumber.open(path) as pdf:
            pages = {page.page_obj.pageid: page.page_number for page in pdf.pages}
            for page in pdf.pages:
                number = page.page_number
                heights.append(float(page.height))
                words.extend(
                    Word(
                        text=w["text"],
                        pdf_page=number,
                        font_size=round(float(w["size"]), _SIZE_DECIMALS),
                        x0=float(w["x0"]),
                        top=float(w["top"]),
                        x1=float(w["x1"]),
                        bottom=float(w["bottom"]),
                    )
                    for w in page.extract_words(extra_attrs=["size"])
                )
                rules.extend(
                    HorizontalRule(
                        pdf_page=number,
                        y=float(line["top"]),
                        x0=float(line["x0"]),
                        x1=float(line["x1"]),
                    )
                    for line in page.lines
                    if abs(line["bottom"] - line["top"]) < _FLAT and line["x1"] > line["x0"]
                )
                images.extend(
                    ImageBox(
                        pdf_page=number,
                        x0=float(image["x0"]),
                        top=float(image["top"]),
                        x1=float(image["x1"]),
                        bottom=float(image["bottom"]),
                    )
                    for image in page.images
                )
            outline = _outline(pdf.doc, pages, document.doc_id)
        return ParsedDocument(
            document=document,
            words=tuple(words),
            rules=tuple(rules),
            outline=outline,
            page_heights=tuple(heights),
            images=tuple(images),
        )
