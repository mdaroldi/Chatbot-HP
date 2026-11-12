"""The --if-needed rule: a live build is reused only when every input still matches.

Each condition has its own test, next to the matching case, so a check that always
answers "current" or always answers "stale" fails here.
"""

from dataclasses import replace

import pytest

from app.domain.ingestion import IndexBuild, ModelFiles
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
        "point_count",
    }
