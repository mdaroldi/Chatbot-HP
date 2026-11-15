"""Per-run config checks: the backend's environment, /ready and every stored row must
show the configuration the run claims, or the run stops as invalid.
"""

import dataclasses

import pytest

from app.core.settings import LLMProviderName, Settings
from app.domain.chat import Outcome, RefusalReason
from app.domain.ingestion import ChunkerKind
from app.domain.retrieval import SearchMode
from app.evaluation.answers.answer_matrix import RunSpec, load_answer_matrix
from app.evaluation.answers.config_check import (
    CHECKED_NAMES,
    INDEX_CHUNKERS,
    ConfigMismatchError,
    RowExpectation,
    RunSettings,
    backend_env,
    check_index,
    check_printenv,
    check_ready,
    check_rows,
    overrides,
    row_expectation,
    run_settings,
)
from tests.unit.test_eval_answers_budget import HAIKU, MATRIX, SONNET
from tests.unit.test_eval_answers_charging import row
from tests.unit.test_eval_answers_matrix import REPO_FILE

BUILD = "20261113t173108z"
REFERENCE = RunSpec(name="reference-1", items="dev")  # type: ignore[arg-type]  # TOML-shaped


def spec(**env: str) -> RunSpec:
    return RunSpec(name="ablation", items="answers-dev", env=env)  # type: ignore[arg-type]


# ------------------------------------------------------------------ run settings


def test_a_run_without_env_gets_the_backend_defaults() -> None:
    assert run_settings(REFERENCE) == RunSettings(
        mode=SearchMode.HYBRID, rerank=True, top_k=5, rewrite_turns=3, alias="hpdocs_chunks"
    )


def test_run_env_values_are_parsed_with_the_settings_types() -> None:
    run = spec(
        RETRIEVAL_MODE="dense",
        RETRIEVAL_RERANK_ENABLED="false",
        RETRIEVAL_TOP_K="3",
        CHAT_REWRITE_HISTORY_TURNS="0",
        QDRANT_COLLECTION_ALIAS="hpdocs_chunks_naive",
    )
    assert run_settings(run) == RunSettings(
        mode=SearchMode.DENSE, rerank=False, top_k=3, rewrite_turns=0, alias="hpdocs_chunks_naive"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETRIEVAL_MODE", "bm25"),
        ("RETRIEVAL_RERANK_ENABLED", "maybe"),
        ("RETRIEVAL_TOP_K", "0"),
        ("CHAT_REWRITE_HISTORY_TURNS", "-1"),
        ("QDRANT_COLLECTION_ALIAS", "bad alias"),
    ],
)
def test_a_value_the_backend_would_reject_fails_naming_the_setting(name: str, value: str) -> None:
    with pytest.raises(ConfigMismatchError, match=name):
        run_settings(spec(**{name: value}))


def test_every_run_in_the_repo_matrix_has_settings_the_backend_accepts() -> None:
    repo = load_answer_matrix(REPO_FILE)
    settings = {run.name: run_settings(run) for run in repo.runs}
    assert settings["k3"].top_k == 3
    assert settings["naive"].alias == "hpdocs_chunks_naive"
    assert settings["rewrite-off"].rewrite_turns == 0
    assert (settings["dense"].mode, settings["dense"].rerank) == (SearchMode.DENSE, False)


# ------------------------------------------------------------------ backend env


def test_overrides_pin_every_allowlisted_setting_and_the_real_provider() -> None:
    assert overrides(spec(RETRIEVAL_TOP_K="8")) == {
        "LLM_PROVIDER": "anthropic",
        "RETRIEVAL_MODE": "hybrid",
        "RETRIEVAL_RERANK_ENABLED": "true",
        "RETRIEVAL_TOP_K": "8",
        "CHAT_REWRITE_HISTORY_TURNS": "3",
        "QDRANT_COLLECTION_ALIAS": "hpdocs_chunks",
    }


def test_the_expected_env_adds_the_matrix_models_and_token_bounds() -> None:
    expected = backend_env(MATRIX, REFERENCE)
    assert expected == overrides(REFERENCE) | {
        "LLM_ANSWER_MODEL": SONNET,
        "LLM_REWRITE_MODEL": HAIKU,
        "LLM_ANSWER_MAX_TOKENS": str(MATRIX.worst_case.answer_output_tokens),
        "LLM_REWRITE_MAX_TOKENS": str(MATRIX.worst_case.rewrite_output_tokens),
    }
    assert tuple(expected) == CHECKED_NAMES


def test_the_key_is_never_a_checked_name() -> None:
    assert "LLM_PROVIDER" in CHECKED_NAMES
    assert not any("KEY" in name for name in CHECKED_NAMES)


def printed(expected: dict[str, str], **changes: str) -> str:
    return "".join(f"{(expected | changes)[name]}\n" for name in CHECKED_NAMES)


def test_printenv_output_matching_every_value_passes() -> None:
    expected = backend_env(MATRIX, REFERENCE)
    check_printenv(expected, printed(expected))


def test_a_differing_value_fails_naming_the_setting_and_both_values() -> None:
    expected = backend_env(MATRIX, REFERENCE)
    output = printed(expected, LLM_ANSWER_MAX_TOKENS="4096", RETRIEVAL_TOP_K="8")
    with pytest.raises(ConfigMismatchError) as caught:
        check_printenv(expected, output)
    message = str(caught.value)
    assert "LLM_ANSWER_MAX_TOKENS=4096 (expected 2048)" in message
    assert "RETRIEVAL_TOP_K=8 (expected 5)" in message


def test_a_missing_value_fails_because_the_values_no_longer_line_up() -> None:
    expected = backend_env(MATRIX, REFERENCE)
    output = "".join(f"{expected[name]}\n" for name in CHECKED_NAMES[:-1])
    with pytest.raises(ConfigMismatchError, match="printenv returned 9 values for 10 names"):
        check_printenv(expected, output)


# ------------------------------------------------------------------ /ready


def ready(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "status": "ready",
        "llm": "anthropic",
        "checks": {
            "postgres": {"status": "ok"},
            "index": {"status": "ok", "info": {"collection": "hpdocs_chunks_x", "build_id": BUILD}},
        },
    }
    return body | changes


def test_a_ready_backend_serving_the_expected_build_passes() -> None:
    check_ready(ready(), collection="hpdocs_chunks_x", build_id=BUILD)


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (ready(status="not_ready"), "status not_ready"),
        (ready(llm="stub"), "llm stub (expected anthropic)"),
        (ready(checks={}), "no index info"),
        (
            ready(
                checks={
                    "index": {"status": "ok", "info": {"collection": "other", "build_id": BUILD}}
                }
            ),
            "collection other (expected hpdocs_chunks_x)",
        ),
        (
            ready(
                checks={
                    "index": {
                        "status": "ok",
                        "info": {"collection": "hpdocs_chunks_x", "build_id": "b2"},
                    }
                }
            ),
            f"build_id b2 (expected {BUILD})",
        ),
    ],
)
def test_ready_fails_on_any_difference(body: dict[str, object], problem: str) -> None:
    with pytest.raises(ConfigMismatchError, match=problem.replace("(", r"\(").replace(")", r"\)")):
        check_ready(body, collection="hpdocs_chunks_x", build_id=BUILD)


# ------------------------------------------------------------------ stored rows


def expectation(**changes: object) -> RowExpectation:
    base = RowExpectation(model=SONNET, build_id=BUILD, top_k=2, rerank=True, rewrite=True)
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]  # fields per test


def test_the_row_expectation_comes_from_the_matrix_and_the_run() -> None:
    run = spec(
        RETRIEVAL_RERANK_ENABLED="false", RETRIEVAL_TOP_K="3", CHAT_REWRITE_HISTORY_TURNS="0"
    )
    assert row_expectation(MATRIX, run, BUILD) == RowExpectation(
        model=SONNET, build_id=BUILD, top_k=3, rerank=False, rewrite=False
    )


def test_rows_that_match_pass() -> None:
    rewritten = row(2, rewritten_query="How do I X on the ENVY?", rewrite_ms=300.0)
    layer1 = row(
        3,
        outcome=Outcome.REFUSED,
        refusal_reason=RefusalReason.BELOW_THRESHOLD,
        model=None,
        input_tokens=None,
        output_tokens=None,
    )
    check_rows((row(1), rewritten, layer1), expectation())


def test_rows_without_rerank_or_rewrite_pass_when_the_run_turns_them_off() -> None:
    check_rows((row(1, relevance_signal=None),), expectation(rerank=False, rewrite=False))


@pytest.mark.parametrize(
    ("stored", "expected", "problem"),
    [
        (row(1, model=HAIKU), expectation(), f"row m-1: model {HAIKU} (expected {SONNET})"),
        (
            row(1, index_build_id="b2"),
            expectation(),
            f"row m-1: index_build_id b2 (expected {BUILD})",
        ),
        (row(1, index_build_id=None), expectation(), "row m-1: index_build_id None"),
        (row(1, chunk_ids=("c-1",)), expectation(), "row m-1: 1 chunk ids (expected 2)"),
        (
            row(1, relevance_signal=None),
            expectation(),
            "row m-1: no relevance signal with rerank on",
        ),
        (row(1), expectation(rerank=False), "row m-1: a relevance signal with rerank off"),
        (
            row(1, rewrite_ms=250.0),
            expectation(rewrite=False),
            "row m-1: a rewrite with rewrite off",
        ),
        (
            row(1, rewritten_query="q"),
            expectation(rewrite=False),
            "row m-1: a rewrite with rewrite off",
        ),
    ],
)
def test_any_row_difference_fails_naming_the_row(
    stored: object, expected: RowExpectation, problem: str
) -> None:
    with pytest.raises(ConfigMismatchError, match=problem.replace("(", r"\(").replace(")", r"\)")):
        check_rows((row(5), stored), expected)  # type: ignore[arg-type]  # a StoredAnswer


def test_every_problem_is_reported_not_only_the_first() -> None:
    with pytest.raises(ConfigMismatchError) as caught:
        check_rows((row(1, model=HAIKU), row(2, chunk_ids=())), expectation())
    assert "row m-1" in str(caught.value)
    assert "row m-2" in str(caught.value)


STUB = LLMProviderName.STUB


def test_stub_overrides_and_env_pin_the_stub_provider() -> None:
    assert overrides(REFERENCE, STUB)["LLM_PROVIDER"] == "stub"
    assert backend_env(MATRIX, REFERENCE, STUB)["LLM_PROVIDER"] == "stub"
    assert backend_env(MATRIX, REFERENCE, STUB)["LLM_ANSWER_MODEL"] == SONNET


def test_ready_checks_the_provider_it_is_given() -> None:
    check_ready(ready(llm="stub"), collection="hpdocs_chunks_x", build_id=BUILD, provider=STUB)
    with pytest.raises(ConfigMismatchError, match="expected stub"):
        check_ready(ready(), collection="hpdocs_chunks_x", build_id=BUILD, provider=STUB)


# ------------------------------------------------------------------ index


def test_each_known_alias_names_its_chunker_and_the_default_alias_is_the_settings() -> None:
    assert {
        Settings.model_fields["qdrant_collection_alias"].default: ChunkerKind.STRUCTURE,
        "hpdocs_chunks_naive": ChunkerKind.NAIVE,
    } == INDEX_CHUNKERS


@pytest.mark.parametrize(
    ("alias", "collection"),
    [
        ("hpdocs_chunks", f"hpdocs_chunks_v2_{BUILD}"),
        ("hpdocs_chunks_naive", f"hpdocs_chunks_naive-v1_{BUILD}"),
    ],
)
def test_an_alias_on_its_chunkers_collection_passes(alias: str, collection: str) -> None:
    check_index(alias, collection)


@pytest.mark.parametrize(
    ("alias", "collection"),
    [
        ("hpdocs_chunks_naive", f"hpdocs_chunks_v2_{BUILD}"),  # an image without CHUNKER
        ("hpdocs_chunks", f"hpdocs_chunks_naive-v1_{BUILD}"),
        ("hpdocs_chunks", f"hpdocs_chunks_v1_{BUILD}"),  # an older chunking version
    ],
)
def test_an_alias_on_another_chunkers_collection_stops_the_run(alias: str, collection: str) -> None:
    with pytest.raises(ConfigMismatchError, match=f"{alias}.*{collection}"):
        check_index(alias, collection)


def test_an_alias_with_no_known_chunker_stops_the_run() -> None:
    with pytest.raises(ConfigMismatchError, match="hpdocs_other"):
        check_index("hpdocs_other", f"hpdocs_chunks_v2_{BUILD}")
