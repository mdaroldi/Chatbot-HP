"""The --if-needed rule: a live build is reused only when every input still matches.

Each condition has its own test, next to the matching case, so a check that always
answers "current" or always answers "stale" fails here.
"""

from dataclasses import replace

import pytest

from app.domain.ingestion import Bm25Params, IndexBuild, ModelFiles
from app.ingestion.freshness import Expected, StaleReason, stale_reason
from tests.manifest_data import make_build


def expected_for(build: IndexBuild, **overrides: object) -> Expected:
    """What a build made now would record, if nothing changed since `build`."""
    values: dict[str, object] = {
        "document_hashes": dict(build.document_hashes),
        "chunking_version": build.chunking_version,
        "dense_model": build.dense_model,
        "sparse_model": build.sparse_model,
        "dense_pin": build.dense,
        "sparse_pin": build.sparse,
        "bm25_avg_len": 256.0,
    }
    return Expected(**(values | overrides))  # type: ignore[arg-type]  # test data


LIVE = make_build()


def changed_file(files: ModelFiles, path: str) -> ModelFiles:
    return replace(files, files={**files.files, path: "f" * 64})


def test_a_build_that_matches_everything_is_current() -> None:
    assert stale_reason(LIVE, expected_for(LIVE), point_count=LIVE.chunk_count) is None


def test_pins_compare_by_content_not_by_identity() -> None:
    copied = replace(LIVE.dense, files=dict(LIVE.dense.files))

    assert stale_reason(LIVE, expected_for(LIVE, dense_pin=copied), LIVE.chunk_count) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"document_hashes": {"demo-guide": "a" * 64, "demo-manual": "9" * 64}}, "documents"),
        ({"document_hashes": {"demo-guide": "a" * 64}}, "documents"),  # a document removed
        (
            {"document_hashes": {"demo-guide": "a" * 64, "demo-manual": "b" * 64, "x": "c" * 64}},
            "documents",
        ),
        ({"chunking_version": "v2"}, "chunking_version"),
        ({"dense_model": "demo/other-dense"}, "dense_model"),
        ({"sparse_model": "demo/other-sparse"}, "sparse_model"),
        ({"dense_pin": replace(LIVE.dense, revision="1" * 40)}, "dense_pin"),
        ({"dense_pin": replace(LIVE.dense, source="demo/elsewhere")}, "dense_pin"),
        ({"dense_pin": changed_file(LIVE.dense, "tokenizer.json")}, "dense_pin"),
        ({"dense_pin": changed_file(LIVE.dense, "extra.json")}, "dense_pin"),  # a file added
        ({"dense_pin": None}, "dense_pin"),  # the configured model lost its pin
        ({"sparse_pin": replace(LIVE.sparse, revision="2" * 40)}, "sparse_pin"),
        ({"sparse_pin": changed_file(LIVE.sparse, "x.txt")}, "sparse_pin"),
        ({"sparse_pin": None}, "sparse_pin"),
    ],
)
def test_any_changed_input_makes_the_build_stale(overrides: dict[str, object], reason: str) -> None:
    found = stale_reason(LIVE, expected_for(LIVE, **overrides), point_count=LIVE.chunk_count)

    assert found is StaleReason(reason)


@pytest.mark.parametrize("count", [0, LIVE.chunk_count - 1, LIVE.chunk_count + 1])
def test_a_point_count_other_than_the_chunk_count_makes_the_build_stale(count: int) -> None:
    assert stale_reason(LIVE, expected_for(LIVE), point_count=count) is StaleReason.POINT_COUNT


def test_the_reasons_are_the_documented_ones() -> None:
    assert {reason.value for reason in StaleReason} == {
        "no_live_index",
        "no_manifest",
        "documents",
        "chunking_version",
        "dense_model",
        "sparse_model",
        "dense_pin",
        "sparse_pin",
        "sparse_params",
        "point_count",
    }


# ------------------------------------------------------------------ BM25 parameters

FIXED_256 = Bm25Params(avg_len=256.0)
AUTO_107 = Bm25Params(avg_len=107.3, avg_len_auto=True)


@pytest.mark.parametrize(
    ("recorded", "setting"),
    [
        (None, 256.0),  # a row from before migration 0004 used fastembed's defaults
        (FIXED_256, 256.0),
        (Bm25Params(avg_len=120.5), 120.5),
        (AUTO_107, "auto"),  # measured on chunks the other checks prove unchanged
        (AUTO_107, 107.3),  # a fixed value equal to the measured one gives the same vectors
    ],
)
def test_bm25_params_that_give_the_same_vectors_keep_the_build_current(
    recorded: Bm25Params | None, setting: float | str
) -> None:
    live = replace(LIVE, sparse_params=recorded)

    assert stale_reason(live, expected_for(live, bm25_avg_len=setting), LIVE.chunk_count) is None


@pytest.mark.parametrize(
    ("recorded", "setting"),
    [
        (None, 107.0),
        (None, "auto"),
        (FIXED_256, 255.9),
        (FIXED_256, "auto"),
        (Bm25Params(avg_len=107.3), "auto"),  # fixed by hand: not known to be the mean
        (AUTO_107, 256.0),
        (Bm25Params(avg_len=256.0, k=1.5), 256.0),
        (Bm25Params(avg_len=256.0, b=0.5), 256.0),
        (Bm25Params(avg_len=107.3, avg_len_auto=True, k=0.9), "auto"),
    ],
)
def test_other_bm25_params_make_the_build_stale(
    recorded: Bm25Params | None, setting: float | str
) -> None:
    live = replace(LIVE, sparse_params=recorded)

    found = stale_reason(live, expected_for(live, bm25_avg_len=setting), LIVE.chunk_count)

    assert found is StaleReason.SPARSE_PARAMS


def test_the_pins_are_checked_before_the_bm25_params() -> None:
    live = replace(LIVE, sparse_params=FIXED_256)
    expected = expected_for(live, bm25_avg_len=100.0, sparse_pin=None)

    assert stale_reason(live, expected, LIVE.chunk_count) is StaleReason.SPARSE_PIN
