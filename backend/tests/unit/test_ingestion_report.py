"""The chunk review report written by `make inspect-chunks` (never committed)."""

from dataclasses import replace

from app.domain.ingestion import ICON, ContentType, DropReason
from app.ingestion.cleaning import CleaningStats
from app.ingestion.pipeline import ProcessedDocument
from app.ingestion.report import render_markdown, summarize
from tests.index_data import make_batch
from tests.unit.ingest_fakes import DOCUMENTS

STATS = CleaningStats(
    lines=dict.fromkeys(DropReason, 0) | {DropReason.FOOTER: 12, DropReason.TOC: 30},
    pages={DropReason.TOC: 2, DropReason.INDEX: 1},
    empty_sections=3,
)


def processed(count: int = 4) -> ProcessedDocument:
    chunks = list(make_batch(count, doc_id="demo-guide").chunks)  # token counts 10, 11, 12, 13
    if count > 1:
        chunks[1] = replace(chunks[1], text=f"{chunks[1].text} Press {ICON} or {ICON}.")
    return ProcessedDocument(document=DOCUMENTS[0], pages=9, chunks=tuple(chunks), stats=STATS)


def test_summary_counts_pages_drops_chunks_tokens_types_warnings_and_icons() -> None:
    summary = summarize(processed())

    assert (summary.doc_id, summary.pages, summary.chunks) == ("demo-guide", 9, 4)
    assert summary.pages_dropped == {"toc": 2, "index": 1}
    assert summary.lines_dropped == {"footer": 12, "toc": 30}  # zeros left out
    assert summary.empty_sections == 3
    assert (summary.tokens_min, summary.tokens_median, summary.tokens_max) == (10, 11.5, 13)
    assert summary.content_types == {ContentType.TABLE: 2, ContentType.PROCEDURE: 2}
    assert summary.warnings == 2  # chunks 0 and 3
    assert summary.icons == 2


def test_summary_of_an_odd_number_of_chunks_uses_the_middle_token_count() -> None:
    assert summarize(processed(3)).tokens_median == 11


def test_markdown_has_a_stats_table_and_every_chunk_in_full() -> None:
    text = render_markdown([processed()])

    assert text.startswith("# Chunk review\n")
    assert "| demo-guide | 9 | 4 | 10 / 11.5 / 13 | 2 | 2 |" in text
    assert "footer 12, toc 30" in text
    assert "procedure 2, table 2" in text
    batch = processed().chunks
    for chunk in batch:
        assert f"### {chunk.chunk_id}" in text
        assert chunk.text in text
    assert "Setup > Step 1" in text
    assert "PDF pages 2-2, printed pages -" in text
    assert text.count("```") == 2 * len(batch)


def test_markdown_states_its_page_numbering() -> None:
    assert "Page numbers are PDF page numbers" in render_markdown([processed()])


def test_markdown_never_breaks_out_of_a_code_fence() -> None:
    doc = processed(1)
    fenced = replace(doc.chunks[0], text="Demo > Setup\n\n```\nnot a fence\n```")
    text = render_markdown([replace(doc, chunks=(fenced,))])

    assert "````\nDemo > Setup" in text  # a longer fence than any in the chunk
