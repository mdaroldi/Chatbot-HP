"""The answer benchmark's matrix (`eval/answer-matrix.toml`): the hard budget, dated
prices, token bounds, the bootstrap seed and the runs in priority order.

Loaded strictly, like the retrieval matrix: unknown keys, out-of-range values and
rules across rows fail with a message naming the key. Rules across rows: run names are
unique; every model has a price; the held-out split is read by one run at most, which
is judged and reserved (rule 8: held-out once, funded up front); a repeatability
sample needs a judged run of at least that many items; a run's env may set only
`ENV_ALLOWLIST` settings, so no run can change the model or the prompt.
"""

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.infra.llm_stub import SERVED_MODEL as STUB_MODEL

ENV_ALLOWLIST = frozenset(
    {
        "RETRIEVAL_MODE",
        "RETRIEVAL_RERANK_ENABLED",
        "RETRIEVAL_TOP_K",
        "CHAT_REWRITE_HISTORY_TURNS",
        "QDRANT_COLLECTION_ALIAS",
    }
)
_NAME = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
# The smallest item set a judged run can have (held-out); a larger sample cannot be drawn.
_SMALLEST_JUDGED_SET = 30


class MatrixError(Exception):
    """The answer matrix breaks a rule; the message names the key, never a value."""


class ItemSet(StrEnum):
    DEV = "dev"
    ANSWERS_DEV = "answers-dev"
    MULTI_TURN_DEV = "multi-turn-dev"
    HELDOUT = "heldout"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Price(_Strict):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class Models(_Strict):
    answer: str = Field(min_length=1)
    rewrite: str = Field(min_length=1)
    judge: str = Field(min_length=1)


class Tokens(_Strict):
    answer_input_tokens: int = Field(ge=1)
    answer_output_tokens: int = Field(ge=1)
    judge_input_tokens: int = Field(ge=1)
    judge_output_tokens: int = Field(ge=1)
    rewrite_input_tokens: int = Field(default=0, ge=0)
    rewrite_output_tokens: int = Field(default=0, ge=0)


class _Budget(_Strict):
    cap_usd: float = Field(gt=0)
    start_margin: float = Field(ge=1)
    batch_discount: float = Field(gt=0, le=1)
    prices_as_of: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    prices_source: str = Field(min_length=1)


class _Bootstrap(_Strict):
    seed: int = Field(ge=0)
    resamples: int = Field(ge=1)


class RunSpec(_Strict):
    name: str = Field(pattern=_NAME)
    # TOML gives a string; strict mode would accept only an enum member.
    items: Annotated[ItemSet, Field(strict=False)]
    judge: bool = False
    judge_repeat: int = Field(default=0, ge=0, le=_SMALLEST_JUDGED_SET)
    reserved: bool = False
    env: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _rules(self) -> Self:
        if self.judge_repeat and not self.judge:
            raise ValueError("judge_repeat needs judge = true")
        if self.items is ItemSet.HELDOUT and not (self.judge and self.reserved):
            raise ValueError("the heldout run must be judged and reserved")
        unknown = sorted(set(self.env) - ENV_ALLOWLIST)
        if unknown:
            raise ValueError(f"env setting not allowed: {', '.join(unknown)}")
        empty = sorted(key for key, value in self.env.items() if not value.strip())
        if empty:
            raise ValueError(f"env setting with an empty value: {', '.join(empty)}")
        return self


class _File(_Strict):
    budget: _Budget
    models: Models
    prices: dict[str, Price]
    worst_case: Tokens
    estimate: Tokens
    bootstrap: _Bootstrap
    runs: list[RunSpec] = Field(min_length=1)


@dataclass(frozen=True)
class AnswerMatrix:
    cap_usd: float
    start_margin: float
    batch_discount: float
    prices_as_of: str
    prices_source: str
    models: Models
    prices: Mapping[str, Price]
    worst_case: Tokens
    estimate: Tokens
    seed: int
    resamples: int
    runs: tuple[RunSpec, ...]

    def run(self, name: str) -> RunSpec:
        for run in self.runs:
            if run.name == name:
                return run
        raise MatrixError(f"no run named {name}")


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


def _check_rows(parsed: _File) -> None:
    names = [run.name for run in parsed.runs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise MatrixError(f"duplicate run name: {', '.join(duplicates)}")
    if sum(run.items is ItemSet.HELDOUT for run in parsed.runs) > 1:
        raise MatrixError("only one heldout run is allowed (rule 8)")
    for model in (parsed.models.answer, parsed.models.rewrite, parsed.models.judge):
        if model not in parsed.prices:
            raise MatrixError(f"no price for {model}")


def stub_matrix(matrix: AnswerMatrix) -> AnswerMatrix:
    """The matrix for a stub-provider pipeline check: every model is the stub's served
    model at $0, so the same guards, checks and files run without a key or spend.
    """
    models = Models(answer=STUB_MODEL, rewrite=STUB_MODEL, judge=STUB_MODEL)
    free = Price(input_per_mtok=0.0, output_per_mtok=0.0)
    return dataclasses.replace(matrix, models=models, prices={STUB_MODEL: free})


def parse_answer_matrix(text: str) -> AnswerMatrix:
    try:
        parsed = _File.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        raise MatrixError(f"answer matrix is not valid TOML: {exc}") from exc
    except ValidationError as exc:
        raise MatrixError(_describe(exc)) from exc
    _check_rows(parsed)
    return AnswerMatrix(
        cap_usd=parsed.budget.cap_usd,
        start_margin=parsed.budget.start_margin,
        batch_discount=parsed.budget.batch_discount,
        prices_as_of=parsed.budget.prices_as_of,
        prices_source=parsed.budget.prices_source,
        models=parsed.models,
        prices=parsed.prices,
        worst_case=parsed.worst_case,
        estimate=parsed.estimate,
        seed=parsed.bootstrap.seed,
        resamples=parsed.bootstrap.resamples,
        runs=tuple(parsed.runs),
    )


def load_answer_matrix(path: Path) -> AnswerMatrix:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MatrixError(f"cannot read answer matrix {path.name}") from exc
    return parse_answer_matrix(text)
