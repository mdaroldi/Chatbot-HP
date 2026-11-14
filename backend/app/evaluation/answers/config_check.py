"""Per-run config checks: the running backend must serve the configuration a run claims.

A run's matrix row sets only allowlisted retrieval and rewrite settings; the answer
model and the token bounds come from the matrix and must already be what the backend
runs with, because the budget's hard stop is computed from them. Three checks, any
mismatch stops the run as invalid:
1. The backend container's environment (`printenv` of `CHECKED_NAMES`, never the key)
   equals `backend_env`: the run's settings, the real provider, the matrix's answer
   and rewrite models and the matrix's output-token bounds.
2. `/ready` reports the real provider and the build the run's alias points at.
3. Every stored row shows it: the answer model, that build, `top_k` chunk ids, a
   relevance signal exactly when rerank is on and no rewrite when rewriting is off.

The driver passes `overrides` for every allowlisted setting, not only the row's, so a
local `.env` cannot silently change a reference run. Messages name settings, rows by
id, models, builds and counts: never message content or the key.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.settings import LLMProviderName, Settings
from app.domain.retrieval import SearchMode
from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec
from app.evaluation.answers.store import StoredAnswer

_RUN_NAMES = (
    "RETRIEVAL_MODE",
    "RETRIEVAL_RERANK_ENABLED",
    "RETRIEVAL_TOP_K",
    "CHAT_REWRITE_HISTORY_TURNS",
    "QDRANT_COLLECTION_ALIAS",
)
CHECKED_NAMES = (
    "LLM_PROVIDER",
    *_RUN_NAMES,
    "LLM_ANSWER_MODEL",
    "LLM_REWRITE_MODEL",
    "LLM_ANSWER_MAX_TOKENS",
    "LLM_REWRITE_MAX_TOKENS",
)


class ConfigMismatchError(Exception):
    """The backend does not serve the run's configuration; the run is invalid."""


@dataclass(frozen=True)
class RunSettings:
    mode: SearchMode
    rerank: bool
    top_k: int
    rewrite_turns: int  # 0 turns rewriting off
    alias: str


@dataclass(frozen=True)
class RowExpectation:
    model: str
    build_id: str
    top_k: int
    rerank: bool
    rewrite: bool


_FIELDS = Settings.model_fields


class _RunEnv(BaseModel):
    """The run settings with the backend's own field definitions (bounds, defaults)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_mode: Annotated[SearchMode, _FIELDS["retrieval_mode"]]
    retrieval_rerank_enabled: Annotated[bool, _FIELDS["retrieval_rerank_enabled"]]
    retrieval_top_k: Annotated[int, _FIELDS["retrieval_top_k"]]
    chat_rewrite_history_turns: Annotated[int, _FIELDS["chat_rewrite_history_turns"]]
    qdrant_collection_alias: Annotated[str, _FIELDS["qdrant_collection_alias"]]


def run_settings(run: RunSpec) -> RunSettings:
    """The row's env parsed as the backend's Settings would, with the backend defaults."""
    try:
        parsed = _RunEnv.model_validate({name.lower(): value for name, value in run.env.items()})
    except ValidationError as exc:
        names = sorted({str(error["loc"][0]).upper() for error in exc.errors()})
        raise ConfigMismatchError(
            f"run {run.name}: the backend would reject {', '.join(names)}"
        ) from None
    return RunSettings(
        mode=parsed.retrieval_mode,
        rerank=parsed.retrieval_rerank_enabled,
        top_k=parsed.retrieval_top_k,
        rewrite_turns=parsed.chat_rewrite_history_turns,
        alias=parsed.qdrant_collection_alias,
    )


def _render(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def overrides(run: RunSpec) -> dict[str, str]:
    """What the driver sets when it restarts the backend for this run."""
    settings = run_settings(run)
    return {
        "LLM_PROVIDER": LLMProviderName.ANTHROPIC.value,
        "RETRIEVAL_MODE": _render(settings.mode),
        "RETRIEVAL_RERANK_ENABLED": _render(settings.rerank),
        "RETRIEVAL_TOP_K": _render(settings.top_k),
        "CHAT_REWRITE_HISTORY_TURNS": _render(settings.rewrite_turns),
        "QDRANT_COLLECTION_ALIAS": settings.alias,
    }


def backend_env(matrix: AnswerMatrix, run: RunSpec) -> dict[str, str]:
    """Every checked name with the value the backend must show, in `CHECKED_NAMES` order."""
    worst = matrix.worst_case
    return overrides(run) | {
        "LLM_ANSWER_MODEL": matrix.models.answer,
        "LLM_REWRITE_MODEL": matrix.models.rewrite,
        "LLM_ANSWER_MAX_TOKENS": str(worst.answer_output_tokens),
        "LLM_REWRITE_MAX_TOKENS": str(worst.rewrite_output_tokens),
    }


def _raise(problems: Sequence[str]) -> None:
    if problems:
        raise ConfigMismatchError("; ".join(problems))


def check_printenv(expected: Mapping[str, str], output: str) -> None:
    """`output` is `printenv` of the expected names in order: one value per line."""
    values = output.splitlines()
    if len(values) != len(expected):
        raise ConfigMismatchError(
            f"printenv returned {len(values)} values for {len(expected)} names"
        )
    _raise(
        [
            f"{name}={value} (expected {want})"
            for (name, want), value in zip(expected.items(), values, strict=True)
            if value != want
        ]
    )


def _get(body: Mapping[str, object], *keys: str) -> object:
    node: object = body
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = cast(Mapping[str, object], node).get(key)
    return node


def check_ready(body: Mapping[str, object], *, collection: str, build_id: str) -> None:
    """`collection` and `build_id` are what the run's alias resolves to in Qdrant."""
    problems = []
    if body.get("status") != "ready":
        problems.append(f"status {body.get('status')}")
    if body.get("llm") != LLMProviderName.ANTHROPIC.value:
        problems.append(f"llm {body.get('llm')} (expected anthropic)")
    info = _get(body, "checks", "index", "info")
    if not isinstance(info, Mapping):
        problems.append("no index info")
    else:
        for key, want in (("collection", collection), ("build_id", build_id)):
            if info.get(key) != want:
                problems.append(f"{key} {info.get(key)} (expected {want})")
    _raise(problems)


def row_expectation(matrix: AnswerMatrix, run: RunSpec, build_id: str) -> RowExpectation:
    settings = run_settings(run)
    return RowExpectation(
        model=matrix.models.answer,
        build_id=build_id,
        top_k=settings.top_k,
        rerank=settings.rerank,
        rewrite=settings.rewrite_turns > 0,
    )


def _row_problems(row: StoredAnswer, expected: RowExpectation) -> list[str]:
    problems = []
    if row.model is not None and row.model != expected.model:
        problems.append(f"model {row.model} (expected {expected.model})")
    if row.index_build_id != expected.build_id:
        problems.append(f"index_build_id {row.index_build_id} (expected {expected.build_id})")
    if len(row.chunk_ids) != expected.top_k:
        problems.append(f"{len(row.chunk_ids)} chunk ids (expected {expected.top_k})")
    if expected.rerank and row.relevance_signal is None:
        problems.append("no relevance signal with rerank on")
    if not expected.rerank and row.relevance_signal is not None:
        problems.append("a relevance signal with rerank off")
    if not expected.rewrite and (row.rewrite_ms is not None or row.rewritten_query is not None):
        problems.append("a rewrite with rewrite off")
    return [f"row {row.message_id}: {problem}" for problem in problems]


def check_rows(rows: Sequence[StoredAnswer], expected: RowExpectation) -> None:
    """A row without a model never reached the answer call (a layer 1 refusal)."""
    _raise([problem for row in rows for problem in _row_problems(row, expected)])
