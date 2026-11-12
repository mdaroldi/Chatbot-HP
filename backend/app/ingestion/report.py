"""Chunk review report for `make inspect-chunks` (git-ignored `ingest-review/chunks.md`).

It holds document text, so it is for local review only and is never committed or
logged. The per-document summary is the stats table of the Phase 1 acceptance report.
"""

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.ingestion import ICON, Chunk, ContentType
from app.ingestion.pipeline import ProcessedDocument


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    pages: int
    pages_dropped: dict[str, int]
    lines_dropped: dict[str, int]  # reasons with at least one line
    empty_sections: int
    chunks: int
    tokens_min: int
    tokens_median: float
    tokens_max: int
    content_types: dict[ContentType, int]
    warnings: int  # chunks with a WARNING or CAUTION
    icons: int  # [icon] placeholders: button glyphs the text lost (ADR 0004)


def summarize(processed: ProcessedDocument) -> DocumentSummary:
    tokens = [chunk.token_count for chunk in processed.chunks] or [0]
    stats = processed.stats
    return DocumentSummary(
        doc_id=processed.document.doc_id,
        pages=processed.pages,
        pages_dropped={reason.value: n for reason, n in stats.pages.items()},
        lines_dropped={reason.value: n for reason, n in stats.lines.items() if n},
        empty_sections=stats.empty_sections,
        chunks=len(processed.chunks),
        tokens_min=min(tokens),
        tokens_median=statistics.median(tokens),
        tokens_max=max(tokens),
        content_types=dict(Counter(chunk.content_type for chunk in processed.chunks)),
        warnings=sum(chunk.has_warning for chunk in processed.chunks),
        icons=sum(chunk.text.count(ICON) for chunk in processed.chunks),
    )


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {n}" for name, n in sorted(counts.items())) or "-"


def _number(value: float) -> str:
    return f"{value:g}"


def _summary_table(summaries: Sequence[DocumentSummary]) -> list[str]:
    lines = [
        "| Document | Pages | Chunks | Tokens min / median / max | With warning | [icon] |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        tokens = f"{s.tokens_min} / {_number(s.tokens_median)} / {s.tokens_max}"
        lines.append(
            f"| {s.doc_id} | {s.pages} | {s.chunks} | {tokens} | {s.warnings} | {s.icons} |"
        )
    lines.append("")
    for s in summaries:
        types = {kind.value: n for kind, n in s.content_types.items()}
        lines += [
            f"- **{s.doc_id}**: pages dropped {_counts(s.pages_dropped)}; lines dropped "
            f"{_counts(s.lines_dropped)}; empty sections {s.empty_sections}; content types "
            f"{_counts(types)}",
        ]
    return lines


def _pages(first: int | None, last: int | None) -> str:
    return "-" if first is None or last is None else f"{first}-{last}"


def _fence(text: str) -> str:
    """A backtick fence longer than any run of backticks in the text."""
    longest, run = 0, 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def _chunk(chunk: Chunk) -> list[str]:
    pages = chunk.pages
    fence = _fence(chunk.text)
    return [
        f"### {chunk.chunk_id}",
        "",
        f"- {' > '.join(chunk.section_path)}",
        f"- PDF pages {_pages(pages.pdf_start, pages.pdf_end)}, printed pages "
        f"{_pages(pages.printed_start, pages.printed_end)}",
        f"- {chunk.content_type.value}, {chunk.token_count} tokens"
        + (", has warning" if chunk.has_warning else ""),
        "",
        fence,
        chunk.text,
        fence,
        "",
    ]


def render_markdown(documents: Sequence[ProcessedDocument]) -> str:
    lines = [
        "# Chunk review",
        "",
        "Local review only: holds source text, never commit it. Page numbers are PDF page "
        "numbers unless marked printed.",
        "",
        *_summary_table([summarize(d) for d in documents]),
    ]
    for document in documents:
        lines += ["", f"## {document.document.doc_id}", ""]
        for chunk in document.chunks:
            lines += _chunk(chunk)
    return "\n".join(lines) + "\n"
