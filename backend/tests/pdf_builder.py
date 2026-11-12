"""Minimal PDF writer for parser tests: text, horizontal rules and bookmarks, no HP content.

Writes uncompressed objects with a correct cross-reference table (byte offsets are
computed, not guessed), so pdfminer parses the file without falling back to repair.
Coordinates are given top-down like pdfplumber reports them; the builder converts.
"""

from dataclasses import dataclass, field
from typing import Literal

BookmarkStyle = Literal["dest", "goto", "named", "named-missing", "uri"]

# Helvetica ascent (1/1000 em): puts a text's glyph top at the requested `top`.
_ASCENT = 0.718


@dataclass(frozen=True)
class Text:
    """ASCII text only (non-ASCII literals are encoded for bookmark titles)."""

    x: float
    top: float
    size: float
    text: str


@dataclass(frozen=True)
class Rule:
    top: float
    x0: float
    x1: float
    width: float = 0.5


@dataclass(frozen=True)
class VerticalRule:
    x: float
    top: float
    bottom: float


@dataclass(frozen=True)
class Page:
    texts: tuple[Text, ...] = ()
    rules: tuple[Rule, ...] = ()
    verticals: tuple[VerticalRule, ...] = ()
    height: float = 792.0
    width: float = 612.0


@dataclass(frozen=True)
class Bookmark:
    title: str
    page: int  # 1-based
    level: int = 1
    style: BookmarkStyle = "goto"


@dataclass
class _Objects:
    bodies: list[bytes] = field(default_factory=list)

    def reserve(self) -> int:
        self.bodies.append(b"")
        return len(self.bodies)

    def set(self, number: int, body: bytes) -> None:
        self.bodies[number - 1] = body

    def add(self, body: bytes) -> int:
        number = self.reserve()
        self.set(number, body)
        return number


def _literal(text: str) -> bytes:
    """ASCII as a literal string; anything else as UTF-16BE with a BOM, like real PDFs
    (byte 0xA0 would mean a euro sign in PDFDocEncoding, not a no-break space).
    """
    if not text.isascii():
        return b"<FEFF" + text.encode("utf-16-be").hex().upper().encode() + b">"
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return b"(" + escaped.encode("ascii") + b")"


def _content(page: Page) -> bytes:
    ops: list[bytes] = []
    for rule in page.rules:
        y = page.height - rule.top
        ops.append(f"{rule.width:g} w {rule.x0:g} {y:g} m {rule.x1:g} {y:g} l S".encode())
    for vertical in page.verticals:
        top, bottom = page.height - vertical.top, page.height - vertical.bottom
        ops.append(f"0.5 w {vertical.x:g} {top:g} m {vertical.x:g} {bottom:g} l S".encode())
    for text in page.texts:
        baseline = page.height - text.top - _ASCENT * text.size
        ops.append(
            f"BT /F1 {text.size:g} Tf {text.x:g} {baseline:g} Td ".encode()
            + _literal(text.text)
            + b" Tj ET"
        )
    return b"\n".join(ops)


def _destination(page_ref: int) -> bytes:
    return f"[{page_ref} 0 R /XYZ 0 792 0]".encode()


def _bookmark_target(mark: Bookmark, page_ref: int, name: str) -> bytes:
    if mark.style == "dest":
        return b"/Dest " + _destination(page_ref)
    if mark.style == "goto":
        return b"/A << /S /GoTo /D " + _destination(page_ref) + b" >>"
    if mark.style in ("named", "named-missing"):
        return b"/A << /S /GoTo /D " + _literal(name) + b" >>"
    return b"/A << /S /URI /URI (https://example.test/) >>"


def _outline(
    objects: _Objects, bookmarks: tuple[Bookmark, ...], page_refs: list[int]
) -> tuple[int, list[tuple[str, int]]]:
    """Build the outline tree from flat levels; returns its root and the named dests."""
    root = objects.reserve()
    numbers = [objects.reserve() for _ in bookmarks]
    parents: list[int] = []
    children: dict[int, list[int]] = {root: []}
    stack: list[tuple[int, int]] = [(0, root)]  # (level, object number)
    for index, mark in enumerate(bookmarks):
        while stack[-1][0] >= mark.level:
            stack.pop()
        parent = stack[-1][1]
        parents.append(parent)
        children[parent].append(numbers[index])
        children[numbers[index]] = []
        stack.append((mark.level, numbers[index]))

    named: list[tuple[str, int]] = []
    for index, mark in enumerate(bookmarks):
        number = numbers[index]
        name = f"dest_{index}"
        if mark.style == "named":
            named.append((name, page_refs[mark.page - 1]))
        siblings = children[parents[index]]
        position = siblings.index(number)
        parts = [
            b"<< /Title " + _literal(mark.title),
            f"/Parent {parents[index]} 0 R".encode(),
            _bookmark_target(mark, page_refs[mark.page - 1], name),
        ]
        if position > 0:
            parts.append(f"/Prev {siblings[position - 1]} 0 R".encode())
        if position < len(siblings) - 1:
            parts.append(f"/Next {siblings[position + 1]} 0 R".encode())
        if children[number]:
            kids = children[number]
            parts.append(f"/First {kids[0]} 0 R /Last {kids[-1]} 0 R /Count {len(kids)}".encode())
        objects.set(number, b" ".join(parts) + b" >>")
    top = children[root]
    objects.set(root, f"<< /Type /Outlines /First {top[0]} 0 R /Last {top[-1]} 0 R >>".encode())
    return root, named


def build_pdf(pages: tuple[Page, ...], bookmarks: tuple[Bookmark, ...] = ()) -> bytes:
    objects = _Objects()
    catalog = objects.reserve()
    pages_root = objects.reserve()
    font = objects.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_refs: list[int] = []
    for page in pages:
        stream = _content(page)
        content = objects.add(
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        page_refs.append(
            objects.add(
                f"<< /Type /Page /Parent {pages_root} 0 R "
                f"/MediaBox [0 0 {page.width:g} {page.height:g}] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()
            )
        )
    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects.set(pages_root, f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())

    extra = b""
    if bookmarks:
        outline_root, named = _outline(objects, bookmarks, page_refs)
        extra += f" /Outlines {outline_root} 0 R".encode()
        if named:
            entries = b" ".join(
                _literal(name) + b" << /D " + _destination(ref) + b" >>" for name, ref in named
            )
            tree = objects.add(b"<< /Names [" + entries + b"] >>")
            extra += f" /Names << /Dests {tree} 0 R >>".encode()
    objects.set(catalog, f"<< /Type /Catalog /Pages {pages_root} 0 R".encode() + extra + b" >>")
    return _serialize(objects.bodies, catalog)


def _serialize(bodies: list[bytes], root: int) -> bytes:
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(bodies) + 1} /Root {root} 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
