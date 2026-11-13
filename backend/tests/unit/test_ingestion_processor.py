"""StructureProcessor: parse, clean, section and chunk one synthetic PDF end to end."""

import pathlib

import pytest

from app.domain.ingestion import ChunkTooLargeError, CorpusDocument, DropReason
from app.infra.pdfplumber_parser import PdfplumberParser
from app.ingestion.chunking import StructureChunker
from app.ingestion.cleaning import CleaningStats
from app.ingestion.pipeline import ProcessedDocument, StructureProcessor
from tests.pdf_builder import Bookmark, Page, Text, build_pdf
from tests.unit.fakes import WordTokenCounter

DOC = CorpusDocument(
    "demo-guide", "Demo Guide", "Demo Printer 100", "https://d.test/g.pdf", "a" * 64
)
PAGES = (
    Page(
        texts=(
            Text(54, 70, 18, "Setup"),
            Text(54, 100, 10, "Open the tray and load plain paper before you print."),
            Text(54, 120, 10, "Close the tray until it clicks into place."),
        )
    ),
    Page(
        texts=(
            Text(54, 70, 14, "Wireless"),
            Text(54, 100, 10, "Press the demo button to join the demo network."),
        )
    ),
)
MARKS = (Bookmark("Setup", 1), Bookmark("Wireless", 2, level=2))


@pytest.fixture
def processed(tmp_path: pathlib.Path) -> ProcessedDocument:
    path = tmp_path / "demo.pdf"
    path.write_bytes(build_pdf(PAGES, MARKS))
    chunker = StructureChunker(WordTokenCounter(), max_tokens=450, min_tokens=0)
    return StructureProcessor(PdfplumberParser(), chunker).process(path, DOC)


def test_a_document_becomes_chunks_under_its_bookmark_paths(processed: ProcessedDocument) -> None:
    assert processed.document == DOC
    assert processed.pages == 2
    assert [c.section_path for c in processed.chunks] == [("Setup",), ("Setup", "Wireless")]
    assert {c.doc_id for c in processed.chunks} == {"demo-guide"}
    assert {c.chunking_version for c in processed.chunks} == {"v2"}
    assert "join the demo network" in processed.chunks[1].text


def test_the_chunker_limit_applies(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "demo.pdf"
    path.write_bytes(build_pdf(PAGES, MARKS))
    # Five words cannot hold the "Demo Guide > Setup" prefix and one word of text.
    chunker = StructureChunker(WordTokenCounter(), max_tokens=5, min_tokens=0)

    with pytest.raises(ChunkTooLargeError):
        StructureProcessor(PdfplumberParser(), chunker).process(path, DOC)


def test_drop_stats_count_every_reason_and_the_page_rules(processed: ProcessedDocument) -> None:
    stats = processed.drop_stats()

    assert {f"lines_{reason.value}" for reason in DropReason} <= set(stats)
    assert {"pages_toc", "pages_index", "sections_empty"} <= set(stats)
    assert all(isinstance(n, int) and n >= 0 for n in stats.values())


def test_drop_stats_carry_the_cleaning_counts() -> None:
    stats = CleaningStats(
        lines={DropReason.FOOTER: 3, DropReason.TOC: 1},
        pages={DropReason.TOC: 2, DropReason.INDEX: 1},
        empty_sections=4,
    )

    found = ProcessedDocument(document=DOC, pages=9, chunks=(), stats=stats).drop_stats()

    assert found == {
        "lines_footer": 3,
        "lines_toc": 1,
        "pages_toc": 2,
        "pages_index": 1,
        "sections_empty": 4,
    }
