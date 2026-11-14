"""The answer benchmark's matrix (`eval/answer-matrix.toml`): budget, prices, token
bounds, the bootstrap seed and the runs in priority order, loaded strictly.
"""

from pathlib import Path

import pytest

from app.evaluation.answers.answer_matrix import (
    ENV_ALLOWLIST,
    AnswerMatrix,
    ItemSet,
    MatrixError,
    load_answer_matrix,
    parse_answer_matrix,
)

REPO_FILE = Path(__file__).parents[3] / "eval" / "answer-matrix.toml"

BASE = """
[budget]
cap_usd = 10.0
start_margin = 1.25
batch_discount = 0.5
prices_as_of = "2026-11-03"
prices_source = "Anthropic pricing"

[models]
answer = "claude-sonnet-5-5"
rewrite = "claude-haiku-5-5"
judge = "claude-opus-5-5"

[prices.claude-sonnet-5-5]
input_per_mtok = 2.0
output_per_mtok = 10.0

[prices.claude-haiku-5-5]
input_per_mtok = 0.1
output_per_mtok = 0.5

[prices.claude-opus-5-5]
input_per_mtok = 4.0
output_per_mtok = 20.0

[worst_case]
answer_input_tokens = 12000
answer_output_tokens = 2048
rewrite_input_tokens = 4000
rewrite_output_tokens = 100
judge_input_tokens = 9000
judge_output_tokens = 4000

[estimate]
answer_input_tokens = 4500
answer_output_tokens = 600
judge_input_tokens = 5000
judge_output_tokens = 1500

[bootstrap]
seed = 20261114
resamples = 2000
"""

REFERENCE = """
[[runs]]
name = "reference-1"
items = "dev"
judge = true
judge_repeat = 20
"""

HELDOUT = """
[[runs]]
name = "heldout"
items = "heldout"
judge = true
reserved = true
"""

DENSE = """
[[runs]]
name = "dense"
items = "answers-dev"
env = { RETRIEVAL_MODE = "dense", RETRIEVAL_RERANK_ENABLED = "false" }
"""


def parse(text: str) -> AnswerMatrix:
    return parse_answer_matrix(text)


def test_a_valid_matrix_is_parsed() -> None:
    matrix = parse(BASE + REFERENCE + HELDOUT + DENSE)

    assert matrix.cap_usd == 10.0
    assert matrix.start_margin == 1.25
    assert matrix.batch_discount == 0.5
    assert matrix.prices_as_of == "2026-11-03"
    assert matrix.models.answer == "claude-sonnet-5-5"
    assert matrix.prices["claude-opus-5-5"].output_per_mtok == 20.0
    assert matrix.worst_case.answer_output_tokens == 2048
    assert matrix.estimate.judge_input_tokens == 5000
    assert (matrix.seed, matrix.resamples) == (20261114, 2000)
    assert [run.name for run in matrix.runs] == ["reference-1", "heldout", "dense"]
    reference, heldout, dense = matrix.runs
    assert (reference.items, reference.judge, reference.judge_repeat) == (ItemSet.DEV, True, 20)
    assert not reference.reserved
    assert dict(reference.env) == {}
    assert heldout.reserved
    assert dense.items is ItemSet.ANSWERS_DEV
    assert dict(dense.env) == {"RETRIEVAL_MODE": "dense", "RETRIEVAL_RERANK_ENABLED": "false"}


def test_runs_are_found_by_name() -> None:
    matrix = parse(BASE + REFERENCE + DENSE)

    assert matrix.run("dense").items is ItemSet.ANSWERS_DEV
    with pytest.raises(MatrixError, match="no run named nope"):
        matrix.run("nope")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("not toml [", "not valid TOML"),
        (BASE.replace("cap_usd = 10.0", "cap_usd = 0.0"), "cap_usd"),
        (BASE.replace("start_margin = 1.25", "start_margin = 0.9"), "start_margin"),
        (BASE.replace("batch_discount = 0.5", "batch_discount = 1.5"), "batch_discount"),
        (BASE.replace("[bootstrap]", "[bootstrap]\nextra = 1"), "extra"),
        (
            BASE.replace('judge = "claude-opus-5-5"', 'judge = "claude-x"') + REFERENCE,
            "no price for claude-x",
        ),
        (BASE.replace("answer_output_tokens = 2048", "answer_output_tokens = 0"), "answer_output"),
        (BASE, "runs"),
        (BASE + REFERENCE + REFERENCE, "duplicate run name: reference-1"),
        (BASE + REFERENCE.replace('"dev"', '"everything"'), "items"),
        (BASE + REFERENCE.replace("reference-1", "Reference 1"), "name"),
        (BASE + HELDOUT.replace("reserved = true", "reserved = false"), "heldout run must be"),
        (BASE + HELDOUT.replace("judge = true", "judge = false"), "heldout run must be"),
        (BASE + HELDOUT + HELDOUT.replace('"heldout"\nitems', '"heldout-2"\nitems'), "one heldout"),
        (BASE + REFERENCE.replace("judge = true", "judge = false"), "judge_repeat needs judge"),
        (BASE + REFERENCE.replace("judge_repeat = 20", "judge_repeat = 200"), "judge_repeat"),
        (BASE + DENSE.replace("RETRIEVAL_MODE", "LLM_ANSWER_MODEL"), "not allowed: LLM_ANSWER"),
        (BASE + DENSE.replace('"false" }', '"  " }'), "empty value"),
    ],
)
def test_an_invalid_matrix_says_what_is_wrong(text: str, match: str) -> None:
    with pytest.raises(MatrixError, match=match):
        parse(text)


def test_the_env_allowlist_is_the_settings_a_run_may_change() -> None:
    assert (
        frozenset(
            {
                "RETRIEVAL_MODE",
                "RETRIEVAL_RERANK_ENABLED",
                "RETRIEVAL_TOP_K",
                "CHAT_REWRITE_HISTORY_TURNS",
                "QDRANT_COLLECTION_ALIAS",
            }
        )
        == ENV_ALLOWLIST
    )


def test_an_unreadable_file_is_a_matrix_error(tmp_path: Path) -> None:
    with pytest.raises(MatrixError, match="cannot read"):
        load_answer_matrix(tmp_path / "missing.toml")


def test_the_repository_matrix_holds_the_owner_budget_and_run_order() -> None:
    matrix = load_answer_matrix(REPO_FILE)

    assert matrix.cap_usd == 10.0
    assert matrix.start_margin == 1.25
    assert [run.name for run in matrix.runs] == [
        "reference-1",
        "reference-2",
        "reference-3",
        "heldout",
        "improvement",
        "hybrid-no-rerank",
        "dense",
        "naive",
        "rewrite-off",
        "k3",
        "sparse",
        "k8",
    ]
    assert {run.name for run in matrix.runs if run.reserved} == {"heldout", "improvement"}
    assert {run.name for run in matrix.runs if run.judge} == {"reference-1", "heldout"}
    assert matrix.run("reference-1").judge_repeat == 20
