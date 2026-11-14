"""Asking a run inside the backend image: start check, index and /ready checks, the run
directory's config record, then per item the hard stop, the API call, settlement and
the stored-row check. Fakes implement the API, index, store and chunk ports; the
ledger, journal, traces and config record are real files.
"""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from app.domain.answer_dataset import EvalItem
from app.domain.chat import AnswerFlags
from app.evaluation.answers.ask import (
    CONFIG_SCHEMA,
    ApiBackend,
    AskError,
    AskOptions,
    AskPorts,
    IndexRef,
    RunDirError,
    RunFiles,
    ask_run,
)
from app.evaluation.answers.budget import (
    BudgetExceededError,
    Kind,
    Ledger,
    RunNotFundedError,
    charge,
)
from app.evaluation.answers.client import (
    Answered,
    ChatClient,
    Failed,
    HarnessCode,
    ItemResult,
    Timeouts,
)
from app.evaluation.answers.config_check import ConfigMismatchError, backend_env
from app.evaluation.answers.journal import read_journal
from app.evaluation.answers.sse import Done
from app.evaluation.answers.store import StoredAnswer
from tests.unit.test_eval_answers_budget import MATRIX, SONNET, item
from tests.unit.test_eval_answers_charging import row
from tests.unit.test_eval_answers_settle import TITLES, FakeChunks

REFERENCE = MATRIX.run("reference-1")
BUILD = "20261113t173108z"
COLLECTION = "hpdocs_chunks_20261113t173108z"
LIVE = IndexRef(COLLECTION, BUILD)
CHUNKS = tuple(f"c-{n}" for n in range(1, 6))  # top_k 5, the backend default


def ready_body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "status": "ready",
        "llm": "anthropic",
        "checks": {"index": {"info": {"collection": COLLECTION, "build_id": BUILD}}},
    }
    return body | changes


class FakeBackend:
    def __init__(
        self,
        ready: Mapping[str, object] | None = None,
        errors: Mapping[str, str] | None = None,
    ) -> None:
        self.body = ready_body() if ready is None else ready
        self.errors = errors or {}
        self.log: list[str] = []

    async def ready(self) -> Mapping[str, object]:
        self.log.append("ready")
        return self.body

    async def answer(self, item: EvalItem, client_id: str) -> ItemResult:
        self.log.append(f"answer {item.id} {client_id}")
        conversation = f"conv-{item.id}"
        if item.id in self.errors:
            return ItemResult(item.id, client_id, conversation, 0, None, self.errors[item.id])
        done = Done(conversation, f"m-{item.id}", "Answer [S1].", None, ("S1",), AnswerFlags())
        return ItemResult(item.id, client_id, conversation, 0, Answered(done, 0, 400.0, 900.0))

    async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None:
        self.log.append(f"delete {conversation_id} {client_id}")
        return None


class FakeRows:
    """One stored row per conversation, its message id taken from the conversation."""

    def __init__(self, **changes: object) -> None:
        self.changes = changes

    async def answers(self, conversation_id: str, client_id: str) -> tuple[StoredAnswer, ...]:
        assert client_id.startswith("client-")  # the item's own client id
        item_id = conversation_id.removeprefix("conv-")
        fields: dict[str, object] = {
            "message_id": f"m-{item_id}",
            "chunk_ids": CHUNKS,
            "index_build_id": BUILD,
        }
        return (row(1, **(fields | self.changes)),)


class FakeIndex:
    def __init__(self, ref: IndexRef | None = LIVE) -> None:
        self.ref = ref
        self.asked: list[str] = []

    async def resolve(self, alias: str) -> IndexRef | None:
        self.asked.append(alias)
        return self.ref


ITEMS = [item(1, f"q-{n}") for n in range(1, 6)]
HELDOUT = [item(1, "h-1")]


def by_run(items: Sequence[EvalItem] = ITEMS) -> dict[str, Sequence[EvalItem]]:
    return {"reference-1": items, "heldout": HELDOUT}


def chunks_of(collection: str) -> FakeChunks:
    assert collection == COLLECTION  # the live collection, not the alias
    return FakeChunks()


def ports(
    backend: FakeBackend | None = None,
    rows: FakeRows | None = None,
    index: FakeIndex | None = None,
) -> AskPorts:
    ids = iter(f"client-{n}" for n in range(1, 100))
    return AskPorts(
        backend=backend or FakeBackend(),
        index=index or FakeIndex(),
        rows=rows or FakeRows(),
        chunks=chunks_of,
        titles=TITLES,
        new_client_id=lambda: next(ids),
    )


@pytest.fixture
def files(tmp_path: Path) -> RunFiles:
    return RunFiles(tmp_path / "answer-runs" / "reference-1", tmp_path / "ledger.jsonl")


async def ask(
    files: RunFiles,
    made: AskPorts | None = None,
    options: AskOptions | None = None,
    items: Sequence[EvalItem] = ITEMS,
    matrix: object = MATRIX,
) -> object:
    return await ask_run(
        matrix,  # type: ignore[arg-type]  # a replaced matrix in some tests
        REFERENCE,
        by_run(items),
        files,
        made or ports(),
        options or AskOptions(),
    )


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ a run


async def test_a_run_asks_every_item_and_saves_results_charges_traces_and_config(
    files: RunFiles,
) -> None:
    backend = FakeBackend()
    index = FakeIndex()

    outcome = await ask(files, ports(backend, index=index))

    assert index.asked == ["hpdocs_chunks"]
    assert backend.log[0] == "ready"
    assert backend.log[1:4] == [
        "answer q-1 client-1",
        "delete conv-q-1 client-1",
        "answer q-2 client-2",
    ]
    assert [r.item_id for r in read_journal(files.results)] == [i.id for i in ITEMS]
    charges = Ledger(files.ledger).charges()
    assert [(c.item_id, c.kind, c.run) for c in charges] == [
        (i.id, Kind.ANSWER, "reference-1") for i in ITEMS
    ]
    assert [t["item_id"] for t in lines(files.traces)] == [i.id for i in ITEMS]
    record = json.loads(files.config.read_text(encoding="utf-8"))
    assert record == {
        "schema": CONFIG_SCHEMA,
        "run": "reference-1",
        "items": "dev",
        "env": backend_env(MATRIX, REFERENCE),
        "collection": COLLECTION,
        "build_id": BUILD,
    }
    assert outcome.asked == 5  # type: ignore[attr-defined]
    assert outcome.health.valid  # type: ignore[attr-defined]


async def test_the_config_record_never_holds_the_key(files: RunFiles) -> None:
    await ask(files)
    assert "ANTHROPIC_API_KEY" not in files.config.read_text(encoding="utf-8")


async def test_a_limited_run_asks_a_spread_of_the_items(files: RunFiles) -> None:
    backend = FakeBackend()

    outcome = await ask(files, ports(backend), AskOptions(limit=2))

    assert [r.item_id for r in read_journal(files.results)] == ["q-1", "q-3"]
    assert outcome.asked == 2  # type: ignore[attr-defined]


async def test_an_error_result_is_recorded_and_counts_against_the_run(files: RunFiles) -> None:
    backend = FakeBackend(errors={"q-2": HarnessCode.READ_TIMEOUT})

    outcome = await ask(files, ports(backend))

    assert [r.error for r in read_journal(files.results)][1] == "read_timeout"
    assert outcome.health.errors_by_code == {"read_timeout": 1}  # type: ignore[attr-defined]
    assert not outcome.health.valid  # type: ignore[attr-defined]  # 1 in 5 is above 5%


async def test_with_nothing_pending_no_backend_call_is_made(files: RunFiles) -> None:
    await ask(files)
    backend = FakeBackend()

    outcome = await ask(files, ports(backend), AskOptions(resume=True))

    assert backend.log == []
    assert outcome.asked == 0  # type: ignore[attr-defined]
    assert outcome.health is None  # type: ignore[attr-defined]


# ------------------------------------------------------------------ budget


async def test_a_run_the_budget_cannot_fund_stops_before_any_backend_call(
    files: RunFiles,
) -> None:
    Ledger(files.ledger).append(
        charge(MATRIX, "other", "x", Kind.ANSWER, SONNET, 4_000_000, 0)  # $8.00
    )
    backend = FakeBackend()

    with pytest.raises(RunNotFundedError):
        await ask(files, ports(backend))

    assert backend.log == []
    assert not files.run_dir.exists()


async def test_the_hard_stop_ends_the_run_before_an_item_that_could_cross_the_cap(
    files: RunFiles,
) -> None:
    # An item's bound is $5.02 and it costs about $0.013: after $4.975 spent the first
    # item fits under the $10 cap and the second no longer does. The spent charge is a
    # bound (estimated), so it does not become the typical tokens of the start check.
    worst = MATRIX.worst_case.model_copy(update={"answer_input_tokens": 2_500_000})
    matrix = dataclasses.replace(MATRIX, worst_case=worst)
    spent = charge(MATRIX, "other", "x", Kind.ANSWER, SONNET, 2_487_500, 0, estimated=True)
    Ledger(files.ledger).append(spent)
    backend = FakeBackend()

    with pytest.raises(BudgetExceededError):
        await ask(files, ports(backend), items=ITEMS[:3], matrix=matrix)

    assert [r.item_id for r in read_journal(files.results)] == ["q-1"]
    assert [entry for entry in backend.log if entry.startswith("answer")] == ["answer q-1 client-1"]


# ------------------------------------------------------------------ config checks


async def test_no_live_index_for_the_runs_alias_stops_before_any_backend_call(
    files: RunFiles,
) -> None:
    backend = FakeBackend()

    with pytest.raises(AskError, match="hpdocs_chunks"):
        await ask(files, ports(backend, index=FakeIndex(None)))

    assert backend.log == []


async def test_a_backend_serving_another_build_stops_before_any_item(files: RunFiles) -> None:
    other = ready_body(checks={"index": {"info": {"collection": "x", "build_id": "y"}}})
    backend = FakeBackend(ready=other)

    with pytest.raises(ConfigMismatchError, match="build_id"):
        await ask(files, ports(backend))

    assert backend.log == ["ready"]
    assert not files.config.exists()


async def test_a_stored_row_with_the_wrong_config_stops_the_run_with_its_charges_kept(
    files: RunFiles,
) -> None:
    with pytest.raises(ConfigMismatchError, match="relevance signal"):
        await ask(files, ports(rows=FakeRows(relevance_signal=None)))

    assert [c.item_id for c in Ledger(files.ledger).charges()] == ["q-1"]
    assert read_journal(files.results) == []


# ------------------------------------------------------------------ run directory


async def test_an_existing_run_directory_needs_resume(files: RunFiles) -> None:
    await ask(files, options=AskOptions(limit=1))
    backend = FakeBackend()

    with pytest.raises(RunDirError, match="--resume"):
        await ask(files, ports(backend))

    assert backend.log == ["ready"]


@pytest.mark.parametrize("options", [AskOptions(resume=True), AskOptions(retry_errors=True)])
async def test_a_resumed_run_with_the_same_config_asks_the_rest(
    files: RunFiles, options: AskOptions
) -> None:
    await ask(files, options=AskOptions(limit=2))

    outcome = await ask(files, options=options)

    assert [r.item_id for r in read_journal(files.results)] == [
        "q-1",
        "q-3",
        "q-2",
        "q-4",
        "q-5",
    ]
    assert outcome.asked == 3  # type: ignore[attr-defined]
    assert len(outcome.results) == 5  # type: ignore[attr-defined]


async def test_a_resume_against_another_build_is_refused_naming_what_differs(
    files: RunFiles,
) -> None:
    await ask(files, options=AskOptions(limit=1))
    other = IndexRef(COLLECTION, "20261201t000000z")
    backend = FakeBackend(
        ready=ready_body(
            checks={"index": {"info": {"collection": COLLECTION, "build_id": other.build_id}}}
        )
    )

    with pytest.raises(RunDirError, match="build_id"):
        await ask(files, ports(backend, index=FakeIndex(other)), AskOptions(resume=True))

    assert [r.item_id for r in read_journal(files.results)] == ["q-1"]


async def test_results_without_a_config_record_are_refused(files: RunFiles) -> None:
    files.run_dir.mkdir(parents=True)
    files.results.write_text("", encoding="utf-8")

    with pytest.raises(RunDirError, match="config"):
        await ask(files, options=AskOptions(resume=True))


@pytest.mark.parametrize("text", ["{not json", "[1, 2]"])
async def test_an_unreadable_config_record_is_refused(files: RunFiles, text: str) -> None:
    files.run_dir.mkdir(parents=True)
    files.config.write_text(text, encoding="utf-8")

    with pytest.raises(RunDirError, match="config"):
        await ask(files, options=AskOptions(resume=True))


# ------------------------------------------------------------------ the API adapter


def api(handler: httpx.MockTransport) -> tuple[ApiBackend, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=handler, base_url="http://backend:8000")
    chat = ChatClient(http, timeouts=Timeouts(), clock=lambda: 0.0, sleep=_no_sleep)
    return ApiBackend(http, chat), http


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.parametrize("status", [200, 503])
async def test_ready_returns_the_body_whatever_the_status(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ready"
        return httpx.Response(status, json={"status": "x"})

    backend, http = api(httpx.MockTransport(handler))
    async with http:
        assert await backend.ready() == {"status": "x"}


@pytest.mark.parametrize("content", [b"not json", b"[1, 2]"])
async def test_a_ready_body_that_is_not_a_json_object_fails(content: bytes) -> None:
    backend, http = api(httpx.MockTransport(lambda _: httpx.Response(200, content=content)))
    async with http:
        with pytest.raises(AskError, match="/ready"):
            await backend.ready()


async def test_the_adapter_creates_a_conversation_asks_and_deletes_through_the_api() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(204)

    backend, http = api(httpx.MockTransport(handler))
    async with http:
        result = await backend.answer(item(1, "q-1"), "client-1")
        deleted = await backend.delete_conversation("conv-1", "client-1")

    assert result.error is not None
    assert result.retries == 2
    assert deleted is None
    assert seen[0] == ("POST", "/v1/conversations")
    assert seen[-1] == ("DELETE", "/v1/conversations/conv-1")
