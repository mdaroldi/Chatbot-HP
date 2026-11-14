"""Asking one run of the answer matrix, inside the backend image (`ask` command).

The host-side driver has already restarted the backend with the run's settings and
checked its environment. This side checks everything else before the first paid call,
cheapest first:
1. The start check (`estimate.run_start`): the rest of the run, with margin, must fit
   in the budget left after the reserved runs. Files only.
2. The run's alias resolves to a live collection with a succeeded build.
3. `/ready` reports the real provider and exactly that collection and build, once it
   is ready (it is polled while the restarted backend still loads its query models).
4. The run directory's config record (`config.json`: run, item set, expected backend
   env, collection, build) is written on the first ask. A run directory that exists
   needs `--resume` (or `--retry-errors`), and its record must equal the new one, so a
   run is never continued against another build or other settings.

Then each item, one at a time: the hard stop (`guard_item`, at the item's worst case)
-> the API call -> settlement (`settle.Settler`: ledger, trace, delete) -> the stored
rows must show the run's config (`check_rows`; a mismatch stops the run as invalid with
its charges kept, and the item gets no journal line) -> the journal line.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import httpx

from app.core.settings import LLMProviderName
from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.answer_matrix import AnswerMatrix, RunSpec
from app.evaluation.answers.budget import (
    Ledger,
    guard_item,
    item_worst_case,
    require_start,
)
from app.evaluation.answers.client import ChatClient, Failed, ItemResult, run_item
from app.evaluation.answers.config_check import (
    backend_env,
    check_ready,
    check_rows,
    row_expectation,
    run_settings,
)
from app.evaluation.answers.estimate import run_start
from app.evaluation.answers.excerpts import ChunkSource
from app.evaluation.answers.health import RunHealth, run_health
from app.evaluation.answers.journal import (
    Journal,
    latest,
    pending,
    read_journal,
    run_pending,
    spread,
)
from app.evaluation.answers.jsonl import Json
from app.evaluation.answers.settle import Settler, TraceLog
from app.evaluation.answers.store import AnswerRows

logger = logging.getLogger(__name__)

CONFIG_SCHEMA = "answers-run-v1"
# Right after a restart the compose healthcheck passes while /ready still waits for the
# query models to load; two minutes covers that on a loaded laptop.
READY_ATTEMPTS = 60
READY_INTERVAL_SECONDS = 2.0


class AskError(Exception):
    """The run cannot start or go on; the message names files, settings and ids only."""


class RunDirError(AskError):
    """The run directory does not belong to this run as configured now."""


@dataclass(frozen=True)
class IndexRef:
    collection: str
    build_id: str


class IndexResolver(Protocol):
    async def resolve(self, alias: str) -> IndexRef | None:
        """The alias's live collection and its succeeded build; None if either is missing."""
        ...


class Backend(Protocol):
    async def ready(self) -> Mapping[str, object]: ...

    async def answer(self, item: EvalItem, client_id: str) -> ItemResult: ...

    async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None: ...


@dataclass(frozen=True)
class AskPorts:
    backend: Backend
    index: IndexResolver
    rows: AnswerRows
    chunks: Callable[[str], ChunkSource]  # by collection name
    titles: Mapping[str, str]
    new_client_id: Callable[[], str]
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass(frozen=True)
class AskOptions:
    resume: bool = False
    retry_errors: bool = False  # implies resume
    limit: int | None = None  # the pilot: a spread of the pending items
    provider: LLMProviderName = LLMProviderName.ANTHROPIC


@dataclass(frozen=True)
class RunFiles:
    """The git-ignored files of one run, and the phase's shared ledger."""

    run_dir: Path
    ledger: Path

    @property
    def results(self) -> Path:
        return self.run_dir / "results.jsonl"

    @property
    def traces(self) -> Path:
        return self.run_dir / "traces.jsonl"

    @property
    def config(self) -> Path:
        return self.run_dir / "config.json"


@dataclass(frozen=True)
class AskOutcome:
    asked: int
    results: tuple[ItemResult, ...]  # every recorded item's final result
    health: RunHealth | None  # None when nothing was asked


class ApiBackend:
    """The backend API as the harness sees it, at the compose service address."""

    def __init__(self, http: httpx.AsyncClient, chat: ChatClient) -> None:
        self._http, self._chat = http, chat

    async def ready(self) -> Mapping[str, object]:
        """The body whatever the status: a 503 body says which check failed."""
        response = await self._http.get("/ready")
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise AskError(f"/ready returned no JSON object (HTTP {response.status_code})")
        return cast(Mapping[str, object], body)

    async def answer(self, item: EvalItem, client_id: str) -> ItemResult:
        return await run_item(self._chat, item, client_id)

    async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None:
        return await self._chat.delete_conversation(conversation_id, client_id)


async def _wait_ready(
    backend: Backend, sleep: Callable[[float], Awaitable[None]]
) -> Mapping[str, object]:
    """The first ready body, or the last body once the wait is over (check_ready then
    names its status). A ready body is returned at once, whatever build it serves.
    """
    for _ in range(READY_ATTEMPTS - 1):
        body = await backend.ready()
        if body.get("status") == "ready":
            return body
        await sleep(READY_INTERVAL_SECONDS)
    return await backend.ready()


def _record(matrix: AnswerMatrix, run: RunSpec, index: IndexRef, provider: LLMProviderName) -> Json:
    return {
        "schema": CONFIG_SCHEMA,
        "run": run.name,
        "items": run.items.value,
        "env": backend_env(matrix, run, provider),
        "collection": index.collection,
        "build_id": index.build_id,
    }


def _claim_run_dir(files: RunFiles, record: Json, *, resume: bool) -> None:
    if files.config.exists():
        if not resume:
            raise RunDirError(f"run directory {files.run_dir.name} exists; pass --resume")
        try:
            recorded = json.loads(files.config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RunDirError(f"cannot read the config record of {files.run_dir.name}") from None
        if not isinstance(recorded, dict):
            raise RunDirError(f"the config record of {files.run_dir.name} is not an object")
        differs = sorted(
            key for key in record.keys() | recorded.keys() if recorded.get(key) != record.get(key)
        )
        if differs:
            raise RunDirError(
                f"run {files.run_dir.name} was recorded with another config: {', '.join(differs)}"
            )
        return
    if files.results.exists() or files.traces.exists():
        raise RunDirError(f"run directory {files.run_dir.name} has results but no config record")
    files.run_dir.mkdir(parents=True, exist_ok=True)
    files.config.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def ask_run(  # noqa: PLR0913, PLR0917 - the run, its inputs, files, ports and options
    matrix: AnswerMatrix,
    run: RunSpec,
    items_by_run: Mapping[str, Sequence[EvalItem]],
    files: RunFiles,
    ports: AskPorts,
    options: AskOptions,
) -> AskOutcome:
    """`items_by_run` holds every run's items (the start check holds back the reserved
    runs' estimates); this run asks `items_by_run[run.name]`.
    """
    items = items_by_run[run.name]
    resume = options.resume or options.retry_errors
    recorded = read_journal(files.results)
    todo = pending(items, recorded, retry_errors=options.retry_errors)
    if options.limit is not None:
        todo = spread(todo, options.limit)
    if not todo:
        final = latest(recorded)
        return AskOutcome(0, tuple(final[i.id] for i in items if i.id in final), None)

    ledger = Ledger(files.ledger)
    require_start(run_start(matrix, run, items_by_run, ledger.charges()))
    alias = run_settings(run).alias
    index = await ports.index.resolve(alias)
    if index is None:
        raise AskError(f"alias {alias} has no live collection with a succeeded build")
    check_ready(
        await _wait_ready(ports.backend, ports.sleep),
        collection=index.collection,
        build_id=index.build_id,
        provider=options.provider,
    )
    _claim_run_dir(files, _record(matrix, run, index, options.provider), resume=resume)

    expected = row_expectation(matrix, run, index.build_id)
    settle = Settler(
        matrix,
        run.name,
        ports.rows,
        ports.chunks(index.collection),
        ports.titles,
        ports.backend,
        ledger,
        TraceLog(files.traces),
    )

    async def ask(item: EvalItem, client_id: str, attempt: int) -> ItemResult:
        guard_item(matrix, spent=ledger.spent(), worst=item_worst_case(matrix, item))
        result = await ports.backend.answer(item, client_id)
        settled = await settle(item, result, attempt=attempt)
        check_rows(settled.rows, expected)
        logger.info("item asked", extra={"item": item.id, "error": result.error})
        return result

    results = await run_pending(
        items,
        Journal(files.results),
        ask,
        ports.new_client_id,
        retry_errors=options.retry_errors,
        limit=options.limit,
    )
    return AskOutcome(len(todo), tuple(results), run_health(results))
