"""Named retrieval configurations of the benchmark (`eval/retrieval-matrix.toml`).

Each entry overrides a few settings over the configuration the API would use, so a
row names only what it changes, and the report records every resolved value. Every
configuration is scored over its whole candidate list: `top_k` is set to
`rerank_candidates`, so Recall@20 is defined and Recall@5 reads the first five.
"""

import dataclasses
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.settings import Settings
from app.domain.retrieval import RetrievalConfig, SearchMode
from app.services.retrieval import config_from_settings

_NAME = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class MatrixError(Exception):
    """The matrix file is unreadable or breaks its schema; the message says where."""


@dataclass(frozen=True)
class NamedConfig:
    name: str
    config: RetrievalConfig
    reranker: str  # the model it reranks with; unused when rerank is off


@dataclass(frozen=True)
class Matrix:
    seed: int
    resamples: int
    configs: tuple[NamedConfig, ...]

    def select(self, names: Sequence[str]) -> tuple[NamedConfig, ...]:
        """The named configurations in matrix order; no names means all of them."""
        known = {named.name for named in self.configs}
        unknown = sorted(set(names) - known)
        if unknown:
            raise MatrixError(f"unknown config: {', '.join(unknown)}")
        return tuple(c for c in self.configs if not names or c.name in names)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Bootstrap(_Strict):
    seed: int = Field(ge=0)
    resamples: int = Field(ge=1)


class _Entry(_Strict):
    name: str = Field(pattern=_NAME)
    # TOML gives a string; strict mode would accept only an enum member.
    mode: Annotated[SearchMode, Field(strict=False)] | None = None
    rerank: bool | None = None
    rrf_k: int | None = Field(default=None, ge=1)
    product_filter: bool | None = None
    dense_query_prefix: str | None = Field(default=None, max_length=200)
    reranker: str | None = Field(default=None, min_length=1)


class _File(_Strict):
    bootstrap: _Bootstrap
    configs: list[_Entry] = Field(min_length=1)


def _named(entry: _Entry, base: RetrievalConfig, reranker: str) -> NamedConfig:
    overrides = entry.model_dump(exclude={"name", "reranker"}, exclude_none=True)
    return NamedConfig(
        name=entry.name,
        config=dataclasses.replace(base, **overrides),
        reranker=entry.reranker or reranker,
    )


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


def parse_matrix(text: str, settings: Settings) -> Matrix:
    try:
        parsed = _File.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        raise MatrixError(f"retrieval matrix is not valid TOML: {exc}") from exc
    except ValidationError as exc:
        raise MatrixError(_describe(exc)) from exc
    names = [entry.name for entry in parsed.configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise MatrixError(f"duplicate config name: {', '.join(duplicates)}")
    base = dataclasses.replace(
        config_from_settings(settings), top_k=settings.retrieval_rerank_candidates
    )
    return Matrix(
        seed=parsed.bootstrap.seed,
        resamples=parsed.bootstrap.resamples,
        configs=tuple(_named(entry, base, settings.reranker_model) for entry in parsed.configs),
    )


def load_matrix(path: Path, settings: Settings) -> Matrix:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MatrixError(f"cannot read retrieval matrix {path.name}") from exc
    return parse_matrix(text, settings)
