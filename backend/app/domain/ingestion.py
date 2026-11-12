"""Ingestion intermediate representation and stage interfaces (pure, no I/O).

Data flows parser -> TextLine/OutlineEntry -> Block -> Section -> Chunk. Every
stage after the parser works on these types only, so it is tested with synthetic
data. Page numbers are kept twice: PDF page indices (1-based) and the printed page
numbers from the footers, which are absent for unnumbered front matter.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DOC_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class ContentType(StrEnum):
    PROCEDURE = "procedure"
    TABLE = "table"
    REFERENCE = "reference"
    SPECS = "specs"
    PARTS = "parts"
    REGULATORY = "regulatory"
    FRONT_MATTER = "front_matter"


class DropReason(StrEnum):
    """Why cleaning removed content; every drop is counted in the build manifest."""

    TOC = "toc"
    INDEX = "index"
    CHAPTER_TOC = "chapter_toc"
    SECTION_TOC = "section_toc"
    FOOTER = "footer"
    NON_LATIN = "non_latin"
    NON_ENGLISH_LATIN = "non_english_latin"
    NON_LATIN_TABLE = "non_latin_table"
    EMPTY_AFTER_CLEANING = "empty_after_cleaning"


class NoteLabel(StrEnum):
    WARNING = "WARNING"
    CAUTION = "CAUTION"
    IMPORTANT = "IMPORTANT"
    NOTE = "NOTE"
    TIP = "TIP"

    @property
    def is_hazard(self) -> bool:
        """Hazard notes must travel with their chunk and set has_warning."""
        return self in (NoteLabel.WARNING, NoteLabel.CAUTION)


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    STEP = "step"
    BULLET = "bullet"
    NOTE = "note"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"


class BuildStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CorpusError(Exception):
    """Base for corpus failures; messages name the document, never its content."""


class CorpusManifestError(CorpusError):
    """The corpus manifest is unreadable or breaks the allowlist rules."""


class CorpusNotAllowedError(CorpusError):
    """A document outside the manifest was requested."""


class CorpusDownloadError(CorpusError):
    """The document could not be fetched: network, HTTP status, size, redirect or type."""


class ChecksumMismatchError(CorpusError):
    """The bytes were fetched but are not the pinned document."""


class ChunkTooLargeError(Exception):
    """A piece the chunker may not split (a word or a table row) exceeds the limit."""


class DocumentParseError(Exception):
    """The PDF parsed, but lacks the structure ingestion relies on (e.g. bookmarks)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _non_blank(value: str) -> bool:
    return bool(value.strip())


@dataclass(frozen=True)
class PageRange:
    """Inclusive page span in both numberings; printed pages may be absent."""

    pdf_start: int
    pdf_end: int
    printed_start: int | None
    printed_end: int | None

    def __post_init__(self) -> None:
        _require(1 <= self.pdf_start <= self.pdf_end, "pdf page range must be 1 <= start <= end")
        _require(
            (self.printed_start is None) == (self.printed_end is None),
            "printed page range needs both ends or neither",
        )
        if self.printed_start is not None and self.printed_end is not None:
            _require(
                1 <= self.printed_start <= self.printed_end,
                "printed page range must be 1 <= start <= end",
            )

    def union(self, other: "PageRange") -> "PageRange":
        """Span both ranges; unnumbered pages do not erase the numbered part."""
        starts = [p for p in (self.printed_start, other.printed_start) if p is not None]
        ends = [p for p in (self.printed_end, other.printed_end) if p is not None]
        return PageRange(
            pdf_start=min(self.pdf_start, other.pdf_start),
            pdf_end=max(self.pdf_end, other.pdf_end),
            printed_start=min(starts) if starts else None,
            printed_end=max(ends) if ends else None,
        )


def _check_box(pdf_page: int, font_size: float, box: tuple[float, float, float, float]) -> None:
    x0, top, x1, bottom = box
    _require(pdf_page >= 1, "pdf_page must be >= 1")
    _require(font_size > 0, "font_size must be positive")
    _require(x0 <= x1 and top <= bottom, "geometry is inverted")


@dataclass(frozen=True)
class TextLine:
    """One extracted line with the layout facts the cleaning rules need."""

    text: str
    pdf_page: int
    font_size: float
    x0: float
    top: float
    x1: float
    bottom: float

    def __post_init__(self) -> None:
        _check_box(self.pdf_page, self.font_size, (self.x0, self.top, self.x1, self.bottom))


@dataclass(frozen=True)
class OutlineEntry:
    """A PDF bookmark resolved to the page it points at."""

    level: int
    title: str
    pdf_page: int

    def __post_init__(self) -> None:
        _require(self.level >= 1, "outline level must be >= 1")
        _require(_non_blank(self.title), "outline title must not be blank")
        _require(self.pdf_page >= 1, "outline pdf_page must be >= 1")


@dataclass(frozen=True)
class CorpusDocument:
    """An allowlisted source document, pinned by its SHA-256."""

    doc_id: str
    title: str
    product: str
    url: str
    sha256: str

    def __post_init__(self) -> None:
        _require(bool(_DOC_ID.fullmatch(self.doc_id)), "doc_id must be a lowercase slug")
        _require(_non_blank(self.title), "title must not be blank")
        _require(_non_blank(self.product), "product must not be blank")
        _require(self.url.startswith("https://"), "url must use https")
        _require(bool(_SHA256.fullmatch(self.sha256)), "sha256 must be 64 lowercase hex digits")


@dataclass(frozen=True)
class Block:
    """A cleaned unit of content: heading, paragraph, step, bullet, note or table row."""

    kind: BlockKind
    text: str
    pages: PageRange
    note: NoteLabel | None = None
    # x0 of the block's first line (points): tells a note indented under a step from
    # one at the step's own level.
    indent: float = 0.0

    def __post_init__(self) -> None:
        _require(_non_blank(self.text), "block text must not be blank")
        _require(
            (self.kind is BlockKind.NOTE) == (self.note is not None),
            "a note label is required for note blocks and only for them",
        )

    @property
    def is_hazard(self) -> bool:
        return self.note is not None and self.note.is_hazard


@dataclass(frozen=True)
class Section:
    """Blocks under one bookmark path, in reading order."""

    doc_id: str
    path: tuple[str, ...]
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        _require(bool(self.path) and all(map(_non_blank, self.path)), "section path is invalid")
        _require(bool(self.blocks), "a section needs at least one block (blocks)")

    @property
    def chapter(self) -> str:
        return self.path[0]

    @property
    def pages(self) -> PageRange:
        pages = self.blocks[0].pages
        for block in self.blocks[1:]:
            pages = pages.union(block.pages)
        return pages


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit; `payload()` is exactly what the vector index stores."""

    chunk_id: str
    doc_id: str
    product: str
    section_path: tuple[str, ...]
    pages: PageRange
    content_type: ContentType
    has_warning: bool
    chunking_version: str
    token_count: int
    text: str

    def __post_init__(self) -> None:
        _require(
            bool(self.section_path) and all(map(_non_blank, self.section_path)),
            "section_path must be non-empty without blank parts",
        )
        _require(self.token_count > 0, "token_count must be positive")
        _require(_non_blank(self.text), "text must not be blank")
        _require(_non_blank(self.chunking_version), "chunking_version must not be blank")

    def payload(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "product": self.product,
            "section_path": list(self.section_path),
            "chapter": self.section_path[0],
            "pdf_page_start": self.pages.pdf_start,
            "pdf_page_end": self.pages.pdf_end,
            "printed_page_start": self.pages.printed_start,
            "printed_page_end": self.pages.printed_end,
            "content_type": self.content_type.value,
            "has_warning": self.has_warning,
            "chunking_version": self.chunking_version,
            "token_count": self.token_count,
            "text": self.text,
        }


@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(len(self.indices) == len(self.values), "sparse indices and values differ")
        _require(len(set(self.indices)) == len(self.indices), "sparse indices must be unique")
        _require(all(i >= 0 for i in self.indices), "sparse indices must be non-negative")


@dataclass(frozen=True)
class IndexBuild:
    """Manifest entry: what produced a collection, so builds are traceable."""

    build_id: str
    collection: str
    status: BuildStatus
    document_hashes: Mapping[str, str]
    chunking_version: str
    dense_model: str
    sparse_model: str
    chunk_count: int
    drop_stats: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(_non_blank(self.collection), "collection must not be blank")
        _require(self.chunk_count >= 0, "chunk_count must be >= 0")
        _require(bool(self.document_hashes), "document_hashes must not be empty")
        _require(
            all(_SHA256.fullmatch(h) for h in self.document_hashes.values()),
            "document_hashes must hold SHA-256 digests",
        )


# ---------------------------------------------------------------- stage interfaces


@dataclass(frozen=True)
class Word:
    """One word as the parser saw it: PDF points, origin at the page's top-left corner."""

    text: str
    pdf_page: int
    font_size: float
    x0: float
    top: float
    x1: float
    bottom: float

    def __post_init__(self) -> None:
        _require(_non_blank(self.text), "word text must not be blank")
        _check_box(self.pdf_page, self.font_size, (self.x0, self.top, self.x1, self.bottom))


@dataclass(frozen=True)
class HorizontalRule:
    """A drawn horizontal line; table rows are the bands between such rules."""

    pdf_page: int
    y: float
    x0: float
    x1: float

    def __post_init__(self) -> None:
        _require(self.pdf_page >= 1, "pdf_page must be >= 1")
        _require(self.x0 < self.x1, "rule length must be positive")


# Stands where an inline image (a button glyph) sat in a line of text: no vision pass
# reads the glyph, so later stages and the LLM prompt treat it as "an unnamed button".
ICON = "[icon]"


@dataclass(frozen=True)
class ImageBox:
    """Where the parser saw an image; its pixels are never read."""

    pdf_page: int
    x0: float
    top: float
    x1: float
    bottom: float

    def __post_init__(self) -> None:
        _require(self.pdf_page >= 1, "pdf_page must be >= 1")
        _require(self.x0 <= self.x1 and self.top <= self.bottom, "image box is inverted")


@dataclass(frozen=True)
class Table:
    """Cells by row band: a row is one band between rules and never loses its cells."""

    pdf_page: int
    top: float
    bottom: float
    x0: float
    x1: float
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _require(bool(self.header), "a table needs a header")
        _require(self.top < self.bottom and self.x0 < self.x1, "table box is inverted")
        _require(
            all(len(row) == len(self.header) for row in self.rows),
            "every row needs one cells entry per header column",
        )


@dataclass(frozen=True)
class ParsedDocument:
    """Everything the parser hands to the pure stages; pages are 1-based."""

    document: CorpusDocument
    words: tuple[Word, ...]
    rules: tuple[HorizontalRule, ...]
    outline: tuple[OutlineEntry, ...]
    page_heights: tuple[float, ...]
    images: tuple[ImageBox, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.page_heights), "a document needs at least one page")
        _require(all(h > 0 for h in self.page_heights), "page heights must be positive")
        last = self.page_count
        for kind, pages in (
            ("word on", (w.pdf_page for w in self.words)),
            ("rule on", (r.pdf_page for r in self.rules)),
            ("image on", (i.pdf_page for i in self.images)),
            ("bookmark to", (e.pdf_page for e in self.outline)),
        ):
            beyond = next((p for p in pages if p > last), None)
            _require(beyond is None, f"{kind} page {beyond} is beyond the last page {last}")

    @property
    def page_count(self) -> int:
        return len(self.page_heights)

    def page_height(self, pdf_page: int) -> float:
        return self.page_heights[pdf_page - 1]


class CorpusSource(Protocol):
    async def fetch(self, document: CorpusDocument) -> Path:
        """Return a local file whose SHA-256 matches the document, or raise."""
        ...


class DocumentParser(Protocol):
    def parse(self, path: Path, document: CorpusDocument) -> ParsedDocument: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class Chunker(Protocol):
    @property
    def version(self) -> str: ...

    def chunk(self, sections: Sequence[Section], document: CorpusDocument) -> list[Chunk]: ...


class Embedder(Protocol):
    @property
    def dense_model(self) -> str: ...

    @property
    def sparse_model(self) -> str: ...

    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]: ...


class VectorIndex(Protocol):
    async def create_collection(self, name: str, dense_size: int) -> None: ...

    async def upsert(
        self,
        name: str,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector],
    ) -> None: ...

    async def count(self, name: str) -> int: ...

    async def alias_target(self, alias: str) -> str | None: ...

    async def point_alias(self, alias: str, name: str) -> None: ...

    async def delete_collection(self, name: str) -> None: ...

    async def list_collections(self, prefix: str) -> list[str]: ...


class ManifestStore(Protocol):
    async def record(self, build: IndexBuild) -> None: ...

    async def latest_successful(self, collection: str) -> IndexBuild | None: ...
