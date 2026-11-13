"""BM25 document lengths, checked against fastembed's own BM25 (offline, no network).

fastembed's `Bm25` needs only its stopword list on disk, so these tests load the real
tokenizer and stemmer from a temporary folder: the length the adapter reports must be
the `|d|` that fastembed's formula uses for the same text.
"""

import inspect
from pathlib import Path

import pytest
from fastembed.sparse.bm25 import Bm25

from app.domain.ingestion import FASTEMBED_BM25, VerifiedModel
from app.infra.fastembed_bm25 import FastembedBm25Lengths
from tests.model_data import pin

TEXTS = [
    "The printer prints pages; printing is slow.",
    "Press the Power button, then wait.",
    "the a the",  # stopwords only
    "",
]


@pytest.fixture
def bm25_dir(tmp_path: Path) -> Path:
    (tmp_path / "english.txt").write_text("the\na\nis\nthen\n", encoding="utf-8")
    return tmp_path


def model(folder: Path) -> VerifiedModel:
    return VerifiedModel(files=pin(name="Qdrant/bm25", source="Qdrant/bm25"), path=folder)


def lengths(folder: Path, texts: list[str]) -> list[int]:
    return FastembedBm25Lengths(model(folder), folder / "cache").lengths(texts)


def test_lengths_count_stemmed_tokens_without_stopwords_or_punctuation(bm25_dir: Path) -> None:
    # Stems kept: "printer print page print slow", then "press power button wait".
    assert lengths(bm25_dir, TEXTS) == [5, 4, 0, 0]


def test_the_stopwords_come_from_the_verified_folder(bm25_dir: Path) -> None:
    (bm25_dir / "english.txt").write_text("the\na\nis\nthen\nprinter\n", encoding="utf-8")

    assert lengths(bm25_dir, TEXTS[:1]) == [4]


@pytest.mark.parametrize("text", TEXTS[:2])
def test_the_length_is_the_one_fastembeds_formula_uses(bm25_dir: Path, text: str) -> None:
    """With b = 1 and avg_len = |d|, a term seen once scores (k+1)/(1+k) = 1 exactly."""
    [length] = lengths(bm25_dir, [text])
    reference = Bm25(
        "Qdrant/bm25",
        cache_dir=str(bm25_dir),
        specific_model_path=str(bm25_dir),
        local_files_only=True,
        b=1.0,
        avg_len=float(length),
    )
    [vector] = reference.embed([text])
    once = [value for value in vector.values if value < 1.2]  # twice scores above 1.2

    assert once
    assert once == pytest.approx([1.0] * len(once))


def test_the_fastembed_defaults_are_the_recorded_defaults() -> None:
    """A build before migration 0004 has no params: they were fastembed's defaults."""
    parameters = inspect.signature(Bm25.__init__).parameters

    assert (
        parameters["k"].default,
        parameters["b"].default,
        parameters["avg_len"].default,
    ) == (FASTEMBED_BM25.k, FASTEMBED_BM25.b, FASTEMBED_BM25.avg_len)
