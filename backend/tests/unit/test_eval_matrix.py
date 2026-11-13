"""The named retrieval configurations of the benchmark (eval/retrieval-matrix.toml)."""

import dataclasses
import pathlib

import pytest

from app.core.settings import Settings
from app.domain.retrieval import SearchMode
from app.evaluation.matrix import MatrixError, load_matrix, parse_matrix
from tests.unit.conftest import SettingsFactory

REPO_MATRIX = pathlib.Path(__file__).parents[3] / "eval" / "retrieval-matrix.toml"
BASELINE = ("dense", "sparse", "hybrid", "hybrid-rerank")

VALID = """
[bootstrap]
seed = 11
resamples = 2000

[selection]
p95_tie_margin_ms = 1302.5

[[configs]]
name = "dense"
mode = "dense"
rerank = false

[[configs]]
name = "hybrid-rerank-k2"
rrf_k = 2
product_filter = true
dense_query_prefix = "Represent: "
reranker = "other/reranker"
"""


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings(retrieval_rerank_candidates=20, retrieval_top_k=5)


def test_entries_override_the_settings_and_are_scored_over_all_candidates(
    settings: Settings,
) -> None:
    matrix = parse_matrix(VALID, settings)

    assert (matrix.seed, matrix.resamples) == (11, 2000)
    assert matrix.p95_tie_margin_ms == 1302.5
    dense, tuned = matrix.configs
    assert dense.name == "dense"
    assert dense.config.mode is SearchMode.DENSE
    assert dense.config.rerank is False
    assert dense.config.top_k == 20  # the candidate list, so Recall@20 is defined
    assert dense.reranker == settings.reranker_model
    assert tuned.config.mode is SearchMode.HYBRID
    assert tuned.config.rerank is True
    assert (tuned.config.rrf_k, tuned.config.product_filter) == (2, True)
    assert tuned.config.dense_query_prefix == "Represent: "
    assert tuned.reranker == "other/reranker"


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[bootstrap]\nseed = 1\nresamples = 10\n", "configs"),
        (VALID.replace("[selection]\np95_tie_margin_ms = 1302.5\n", ""), "selection"),
        (VALID.replace("= 1302.5", "= -1"), r"selection\.p95_tie_margin_ms"),
        (VALID.replace("= 1302.5", '= "1302"'), r"selection\.p95_tie_margin_ms"),
        ('[[configs]]\nname = "a"\n', "bootstrap"),
        (VALID.replace("rrf_k = 2", "rrf_k = 2\ntop_k = 3"), r"configs\.1\.top_k"),
        (VALID.replace("rerank = false", 'rerank = "no"'), "rerank"),
        (VALID.replace('mode = "dense"', 'mode = "fuzzy"'), "mode"),
        (VALID.replace('name = "dense"', 'name = "Dense One"'), "name"),
        (VALID.replace("resamples = 2000", "resamples = 0"), "resamples"),
        (VALID.replace("rrf_k = 2", "rrf_k = 0"), "rrf_k"),
        (VALID.replace('name = "hybrid-rerank-k2"', 'name = "dense"'), "duplicate"),
        ("[bootstrap\n", "not valid TOML"),
        (VALID.replace('name = "dense"', 'name = "dense"\nx = 1'), r"configs\.0\.x"),
    ],
)
def test_an_invalid_matrix_names_what_is_wrong(
    settings: Settings, text: str, fragment: str
) -> None:
    with pytest.raises(MatrixError, match=fragment):
        parse_matrix(text, settings)


def test_selecting_names_keeps_the_matrix_order_and_none_means_all(settings: Settings) -> None:
    matrix = parse_matrix(VALID, settings)

    assert [c.name for c in matrix.select([])] == ["dense", "hybrid-rerank-k2"]
    assert [c.name for c in matrix.select(["hybrid-rerank-k2", "dense"])] == [
        "dense",
        "hybrid-rerank-k2",
    ]


def test_selecting_an_unknown_name_fails_and_names_it(settings: Settings) -> None:
    matrix = parse_matrix(VALID, settings)

    with pytest.raises(MatrixError, match="no-such-config"):
        matrix.select(["dense", "no-such-config"])


def test_an_unreadable_matrix_fails(settings: Settings, tmp_path: pathlib.Path) -> None:
    with pytest.raises(MatrixError, match="cannot read"):
        load_matrix(tmp_path / "missing.toml", settings)


# ------------------------------------------------------------------ the committed matrix


def test_the_committed_matrix_has_the_approved_bootstrap_and_the_four_way_baseline(
    settings: Settings,
) -> None:
    matrix = load_matrix(REPO_MATRIX, settings)

    assert matrix.resamples == 2000
    # The spread of the reference's p95 over its three Phase 2 runs (ADR 0009).
    assert matrix.p95_tie_margin_ms == 1302
    names = [c.name for c in matrix.configs]
    assert tuple(names[:4]) == BASELINE
    by_name = {c.name: c.config for c in matrix.configs}
    assert [(by_name[n].mode, by_name[n].rerank) for n in BASELINE] == [
        (SearchMode.DENSE, False),
        (SearchMode.SPARSE, False),
        (SearchMode.HYBRID, False),
        (SearchMode.HYBRID, True),
    ]


def test_every_ablation_changes_exactly_one_thing_from_hybrid_rerank(settings: Settings) -> None:
    matrix = load_matrix(REPO_MATRIX, settings)
    base = next(c for c in matrix.configs if c.name == "hybrid-rerank")
    ablations = [c for c in matrix.configs if c.name not in BASELINE]

    assert ablations
    for ablation in ablations:
        changed = {
            field.name
            for field in dataclasses.fields(base.config)
            if getattr(base.config, field.name) != getattr(ablation.config, field.name)
        }
        changed |= {"reranker"} if ablation.reranker != base.reranker else set()
        assert len(changed) == 1, (ablation.name, changed)
