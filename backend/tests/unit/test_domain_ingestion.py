"""Behavior of the ingestion intermediate representation (synthetic data only)."""

import pytest

from app.domain.ingestion import (
    Block,
    BlockKind,
    Chunk,
    ContentType,
    CorpusDocument,
    DropReason,
    EmbeddedChunks,
    NoteLabel,
    OutlineEntry,
    PageRange,
    Section,
    SparseVector,
    TextLine,
)

SHA = "a" * 64


def page_range(
    pdf: tuple[int, int] = (3, 4), printed: tuple[int, int] | None = (1, 2)
) -> PageRange:
    start, end = printed if printed else (None, None)
    return PageRange(pdf_start=pdf[0], pdf_end=pdf[1], printed_start=start, printed_end=end)


def chunk(**overrides: object) -> Chunk:
    values: dict[str, object] = {
        "chunk_id": "00000000-0000-5000-8000-000000000001",
        "doc_id": "demo-guide",
        "product": "Demo Printer 100",
        "section_path": ("Setup", "Connect", "Wireless"),
        "pages": page_range(),
        "content_type": ContentType.PROCEDURE,
        "has_warning": True,
        "chunking_version": "v1",
        "token_count": 42,
        "text": "Demo Guide > Setup > Connect > Wireless\n1. Open the demo menu.",
    }
    return Chunk(**(values | overrides))  # type: ignore[arg-type]  # overrides are test data


# ------------------------------------------------------------------ enums


def test_content_types_match_the_payload_contract() -> None:
    assert {c.value for c in ContentType} == {
        "procedure",
        "table",
        "reference",
        "specs",
        "parts",
        "regulatory",
        "front_matter",
    }


def test_drop_reasons_cover_every_cleaning_rule() -> None:
    assert {r.value for r in DropReason} == {
        "toc",
        "index",
        "chapter_toc",
        "section_toc",
        "footer",
        "non_latin",
        "non_english_latin",
        "non_latin_table",
        "empty_after_cleaning",
    }


def test_note_labels_include_warnings_and_cautions() -> None:
    assert {n.value for n in NoteLabel} >= {"WARNING", "CAUTION", "IMPORTANT", "NOTE", "TIP"}
    assert NoteLabel.WARNING.is_hazard
    assert NoteLabel.CAUTION.is_hazard
    assert not NoteLabel.NOTE.is_hazard


# ------------------------------------------------------------------ page ranges


def test_page_range_keeps_pdf_and_printed_numbering_apart() -> None:
    pages = page_range(pdf=(7, 9), printed=(1, 3))

    assert (pages.pdf_start, pages.pdf_end) == (7, 9)
    assert (pages.printed_start, pages.printed_end) == (1, 3)


def test_page_range_allows_unnumbered_front_matter() -> None:
    pages = page_range(pdf=(2, 2), printed=None)

    assert pages.printed_start is None
    assert pages.printed_end is None


@pytest.mark.parametrize(
    ("pdf", "printed"),
    [
        ((0, 1), None),
        ((5, 4), None),
        ((3, 4), (2, 1)),
        ((3, 4), (0, 1)),
    ],
)
def test_page_range_rejects_impossible_ranges(
    pdf: tuple[int, int], printed: tuple[int, int] | None
) -> None:
    with pytest.raises(ValueError, match="page"):
        page_range(pdf=pdf, printed=printed)


def test_page_range_rejects_half_numbered_printed_range() -> None:
    with pytest.raises(ValueError, match="printed"):
        PageRange(pdf_start=1, pdf_end=1, printed_start=1, printed_end=None)


def test_union_spans_both_numberings() -> None:
    merged = page_range(pdf=(7, 8), printed=(1, 2)).union(page_range(pdf=(9, 10), printed=(3, 4)))

    assert merged == page_range(pdf=(7, 10), printed=(1, 4))


def test_union_with_unnumbered_pages_keeps_the_numbered_part() -> None:
    front = page_range(pdf=(5, 6), printed=None)
    body = page_range(pdf=(7, 8), printed=(1, 2))

    assert front.union(body) == page_range(pdf=(5, 8), printed=(1, 2))
    assert front.union(front) == front


# ------------------------------------------------------------------ parsed input


def test_text_line_rejects_bad_geometry_and_sizes() -> None:
    with pytest.raises(ValueError, match="font_size"):
        TextLine(text="x", pdf_page=1, font_size=0, x0=0, top=0, x1=1, bottom=1)
    with pytest.raises(ValueError, match="geometry"):
        TextLine(text="x", pdf_page=1, font_size=10, x0=5, top=0, x1=1, bottom=1)
    with pytest.raises(ValueError, match="pdf_page"):
        TextLine(text="x", pdf_page=0, font_size=10, x0=0, top=0, x1=1, bottom=1)


def test_outline_entry_validates_level_page_and_title() -> None:
    assert OutlineEntry(level=2, title="Connect", pdf_page=12).level == 2
    for bad in (
        {"level": 0, "title": "A", "pdf_page": 1},
        {"level": 1, "title": " ", "pdf_page": 1},
        {"level": 1, "title": "A", "pdf_page": 0},
    ):
        with pytest.raises(ValueError, match="outline"):
            OutlineEntry(**bad)  # type: ignore[arg-type]  # deliberately invalid data


# ------------------------------------------------------------------ corpus documents


def test_corpus_document_accepts_a_pinned_https_source() -> None:
    document = CorpusDocument(
        doc_id="demo-guide",
        title="Demo Guide",
        product="Demo Printer 100",
        url="https://docs.example.com/demo.pdf",
        sha256=SHA,
    )

    assert document.url.startswith("https://")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("doc_id", "Demo Guide"),
        ("doc_id", ""),
        ("url", "http://docs.example.com/demo.pdf"),
        ("url", "ftp://docs.example.com/demo.pdf"),
        ("sha256", "A" * 64),
        ("sha256", "a" * 63),
        ("title", ""),
        ("product", " "),
    ],
)
def test_corpus_document_rejects_invalid_fields(field: str, value: str) -> None:
    values = {
        "doc_id": "demo-guide",
        "title": "Demo Guide",
        "product": "Demo Printer 100",
        "url": "https://docs.example.com/demo.pdf",
        "sha256": SHA,
    } | {field: value}

    with pytest.raises(ValueError, match=field):
        CorpusDocument(**values)


# ------------------------------------------------------------------ blocks and sections


def test_block_carries_kind_text_pages_and_optional_note_label() -> None:
    note = Block(
        kind=BlockKind.NOTE, text="Unplug first.", pages=page_range(), note=NoteLabel.CAUTION
    )

    assert note.is_hazard
    assert not Block(kind=BlockKind.PARAGRAPH, text="Body.", pages=page_range()).is_hazard


def test_block_rejects_empty_text_and_labels_outside_notes() -> None:
    with pytest.raises(ValueError, match="text"):
        Block(kind=BlockKind.PARAGRAPH, text="  ", pages=page_range())
    with pytest.raises(ValueError, match="note"):
        Block(kind=BlockKind.STEP, text="1. Do it.", pages=page_range(), note=NoteLabel.NOTE)
    with pytest.raises(ValueError, match="note"):
        Block(kind=BlockKind.NOTE, text="Careful.", pages=page_range())


def test_section_exposes_chapter_and_spans_its_blocks() -> None:
    blocks = (
        Block(kind=BlockKind.PARAGRAPH, text="A.", pages=page_range(pdf=(7, 7), printed=(1, 1))),
        Block(kind=BlockKind.STEP, text="1. B.", pages=page_range(pdf=(8, 8), printed=(2, 2))),
    )
    section = Section(doc_id="demo-guide", path=("Setup", "Connect"), blocks=blocks)

    assert section.chapter == "Setup"
    assert section.pages == page_range(pdf=(7, 8), printed=(1, 2))


def test_section_requires_a_path_and_blocks() -> None:
    block = Block(kind=BlockKind.PARAGRAPH, text="A.", pages=page_range())
    with pytest.raises(ValueError, match="path"):
        Section(doc_id="demo-guide", path=(), blocks=(block,))
    with pytest.raises(ValueError, match="path"):
        Section(doc_id="demo-guide", path=("Setup", " "), blocks=(block,))
    with pytest.raises(ValueError, match="blocks"):
        Section(doc_id="demo-guide", path=("Setup",), blocks=())


# ------------------------------------------------------------------ chunks


def test_chunk_payload_matches_the_index_contract() -> None:
    assert chunk().payload() == {
        "doc_id": "demo-guide",
        "product": "Demo Printer 100",
        "section_path": ["Setup", "Connect", "Wireless"],
        "chapter": "Setup",
        "pdf_page_start": 3,
        "pdf_page_end": 4,
        "printed_page_start": 1,
        "printed_page_end": 2,
        "content_type": "procedure",
        "has_warning": True,
        "chunking_version": "v1",
        "token_count": 42,
        "text": "Demo Guide > Setup > Connect > Wireless\n1. Open the demo menu.",
    }


def test_chunk_payload_keeps_null_printed_pages_for_front_matter() -> None:
    payload = chunk(pages=page_range(pdf=(2, 2), printed=None)).payload()

    assert (payload["printed_page_start"], payload["printed_page_end"]) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_count", 0),
        ("text", "   "),
        ("section_path", ()),
        ("section_path", ("Setup", "")),
        ("chunking_version", ""),
    ],
)
def test_chunk_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        chunk(**{field: value})


# ------------------------------------------------------------------ vectors and builds


def test_sparse_vector_pairs_indices_with_values() -> None:
    vector = SparseVector(indices=(3, 7), values=(0.5, 1.25))

    assert dict(zip(vector.indices, vector.values, strict=True)) == {3: 0.5, 7: 1.25}


@pytest.mark.parametrize(
    ("indices", "values"),
    [((1, 2), (0.5,)), ((1, 1), (0.5, 0.5)), ((-1,), (0.5,))],
)
def test_sparse_vector_rejects_mismatched_duplicate_or_negative_indices(
    indices: tuple[int, ...], values: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="sparse"):
        SparseVector(indices=indices, values=values)


# ------------------------------------------------------------------ embedded chunks


def embedded(**overrides: object) -> EmbeddedChunks:
    values: dict[str, object] = {
        "chunks": [chunk(), chunk(chunk_id="00000000-0000-5000-8000-000000000002")],
        "dense": [[0.1, 0.2], [0.3, 0.4]],
        "sparse": [SparseVector(indices=(1,), values=(0.5,)), SparseVector(indices=(), values=())],
    }
    return EmbeddedChunks(**(values | overrides))  # type: ignore[arg-type]  # test data


def test_embedded_chunks_pair_each_chunk_with_its_vectors() -> None:
    batch = embedded()

    assert (len(batch.chunks), batch.dense_size) == (2, 2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunks", [chunk()], "one dense and one sparse vector per chunk"),
        ("dense", [[0.1, 0.2]], "one dense and one sparse vector per chunk"),
        ("sparse", [SparseVector(indices=(), values=())], "one dense and one sparse vector"),
        ("dense", [[0.1, 0.2], [0.3]], "dense vectors must share one non-zero size"),
        ("dense", [[], []], "dense vectors must share one non-zero size"),
    ],
)
def test_embedded_chunks_reject_mismatched_vectors(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        embedded(**{field: value})


def test_embedded_chunks_name_their_chunking_version() -> None:
    assert embedded().chunking_version == "v1"


def test_embedded_chunks_come_from_one_chunking_version() -> None:
    mixed = [chunk(), chunk(chunk_id="00000000-0000-5000-8000-000000000002", chunking_version="v2")]

    with pytest.raises(ValueError, match="one chunking version per build"):
        embedded(chunks=mixed)


def test_embedded_chunks_need_at_least_one_chunk() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        embedded(chunks=[], dense=[], sparse=[])
