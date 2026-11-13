"""BM25 document parameters of a build: the recorded values and the measured mean."""

import math

import pytest

from app.domain.ingestion import (
    BM25_B,
    BM25_DEFAULT_AVG_LEN,
    BM25_K,
    FASTEMBED_BM25,
    Bm25Params,
    mean_length,
)
from tests.manifest_data import make_build


def test_the_defaults_are_fastembeds_and_a_fixed_avg_len() -> None:
    assert (BM25_K, BM25_B, BM25_DEFAULT_AVG_LEN) == (1.2, 0.75, 256.0)
    assert Bm25Params(avg_len=256.0, avg_len_auto=False, k=1.2, b=0.75) == FASTEMBED_BM25


def test_params_keep_every_value() -> None:
    params = Bm25Params(avg_len=107.25, avg_len_auto=True, k=1.5, b=0.5)

    assert (params.avg_len, params.avg_len_auto, params.k, params.b) == (107.25, True, 1.5, 0.5)


@pytest.mark.parametrize(
    "values",
    [
        {"avg_len": 0.0},
        {"avg_len": -1.0},
        {"avg_len": math.inf},
        {"avg_len": math.nan},
        {"avg_len": 10.0, "k": -0.1},
        {"avg_len": 10.0, "k": math.nan},
        {"avg_len": 10.0, "b": -0.01},
        {"avg_len": 10.0, "b": 1.01},
    ],
)
def test_params_outside_the_bm25_ranges_are_rejected(values: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="BM25"):
        Bm25Params(**values)  # type: ignore[arg-type]  # test data


@pytest.mark.parametrize(
    "values", [{"avg_len": 0.5, "k": 0.0, "b": 0.0}, {"avg_len": 1.0, "b": 1.0}]
)
def test_the_range_edges_are_allowed(values: dict[str, float]) -> None:
    assert Bm25Params(**values).avg_len == values["avg_len"]  # type: ignore[arg-type]


def test_the_mean_length_is_the_exact_arithmetic_mean() -> None:
    assert mean_length([100, 101, 121]) == pytest.approx(322 / 3, rel=0, abs=1e-12)
    assert mean_length([7]) == 7.0


@pytest.mark.parametrize("lengths", [[], [0, 0]])
def test_no_chunks_or_no_tokens_have_no_mean_length(lengths: list[int]) -> None:
    with pytest.raises(ValueError, match="BM25"):
        mean_length(lengths)


def test_a_build_recorded_before_the_params_existed_has_none() -> None:
    assert make_build().sparse_params is None
