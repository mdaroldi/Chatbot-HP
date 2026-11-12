"""Synthetic chunks and vectors for vector-index tests (unit and integration)."""

import uuid
from dataclasses import dataclass

from app.domain.ingestion import Chunk, ContentType, PageRange, SparseVector

DENSE_SIZE = 4
_NAMESPACE = uuid.UUID("2f1d3c5e-8a47-4b6e-9a51-0c7e5d2b9f10")


@dataclass(frozen=True)
class Batch:
    chunks: list[Chunk]
    dense: list[list[float]]
    sparse: list[SparseVector]


def make_batch(count: int, doc_id: str = "demo-guide", first: int = 0) -> Batch:
    """`count` chunks with deterministic ids; the last sparse vector is empty, as BM25
    gives for a text of stopwords only.
    """
    numbers = range(first, first + count)
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid5(_NAMESPACE, f"{doc_id}|{n}")),
            doc_id=doc_id,
            product="Demo Printer 100",
            section_path=("Setup", f"Step {n}"),
            pages=PageRange(pdf_start=n + 1, pdf_end=n + 1, printed_start=None, printed_end=None),
            content_type=ContentType.PROCEDURE if n % 2 else ContentType.TABLE,
            has_warning=n % 3 == 0,
            chunking_version="v1",
            token_count=10 + n,
            text=f"Demo Guide > Setup > Step {n}\n\n1. Press the demo button {n} times.",
        )
        for n in numbers
    ]
    dense = [[1.0, float(n), 0.5, 0.25] for n in numbers]
    sparse = [SparseVector(indices=(n, n + 100), values=(1.0, 0.5)) for n in numbers]
    if sparse:
        sparse[-1] = SparseVector(indices=(), values=())
    return Batch(chunks, dense, sparse)
