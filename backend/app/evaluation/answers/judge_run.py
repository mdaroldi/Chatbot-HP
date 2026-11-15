"""One judge pass over a run's answers through the Message Batches API.

Order matters for money. Before anything is sent, the cap is checked at every
request's worst case. The batch id is recorded before waiting, so a crash, a timeout
or a second call collects that batch instead of paying for it again. Each result is
charged at its measured tokens (the batch discount comes from the matrix), then its
verdict or judge error is recorded; a measured call above the matrix bound stops the
pass only after every result is on disk, as the answer settler does.

The cache, shared by all runs, is keyed by `judge.cache_key`, so re-scoring stored
answers is free. The repeatability pass (pass 2) bypasses it and is never cached:
its point is an independent second verdict on the same inputs.

A request that errored, was canceled or expired is not billed. A response that stopped
on a refusal or at `max_tokens`, or whose text is not a valid verdict, is billed and
recorded as a judge error: the judge never falls back to another model (rule 1).
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.evaluation.answers.answer_matrix import AnswerMatrix
from app.evaluation.answers.budget import (
    Charge,
    Kind,
    Ledger,
    charge,
    check_measured,
    guard_item,
    judge_cost,
)
from app.evaluation.answers.journal import spread
from app.evaluation.answers.jsonl import Json, append_object, read_objects
from app.evaluation.answers.judge import (
    JudgeInput,
    JudgeOutputError,
    Verdict,
    cache_key,
    custom_id,
    parse_verdict,
    request_params,
)

SCHEMA = "answers-judge-v1"
BATCH_SCHEMA = "answers-judge-batch-v1"
INVALID_VERDICT = "invalid_verdict"


class JudgeRunError(Exception):
    """A judge file or a batch result breaks the pass; names ids and lines only."""


class BatchPendingError(JudgeRunError):
    """A batch is still running after the wait; the next call collects it."""


@dataclass(frozen=True)
class BatchOutcome:
    """One batch result. `text` is the verdict JSON when the response ended normally;
    tokens are zero for a request that was never billed.
    """

    custom_id: str
    text: str | None
    input_tokens: int
    output_tokens: int
    error: str | None


class BatchPort(Protocol):
    async def submit(self, requests: Mapping[str, Mapping[str, object]]) -> str: ...

    async def ended(self, batch_id: str) -> bool: ...

    async def results(self, batch_id: str) -> list[BatchOutcome]: ...


@dataclass(frozen=True)
class JudgeFiles:
    verdicts: Path  # judge.jsonl in the run dir
    batches: Path  # judge-batches.jsonl in the run dir
    cache: Path  # shared by every run


@dataclass(frozen=True)
class JudgeOptions:
    limit: int | None = None  # a smoke: an even spread of first passes only
    poll_seconds: float = 30.0
    max_wait_seconds: float = 7200.0


@dataclass(frozen=True)
class JudgeRecord:
    item_id: str
    judge_pass: int
    cache_key: str
    verdict: Verdict | None
    error: str | None
    cached: bool


@dataclass(frozen=True)
class JudgeSummary:
    submitted: int
    cached: int
    judged: int  # results recorded from batches in this call
    errors: int


@dataclass(frozen=True)
class _Request:
    judged: JudgeInput
    judge_pass: int
    key: str

    @property
    def cid(self) -> str:
        return custom_id(self.judged.item_id, self.judge_pass)


# ---------------------------------------------------------------- files


def _record_json(record: JudgeRecord) -> Json:
    return {
        "schema": SCHEMA,
        "item_id": record.item_id,
        "pass": record.judge_pass,
        "cache_key": record.cache_key,
        "verdict": None if record.verdict is None else record.verdict.model_dump(mode="json"),
        "error": record.error,
        "cached": record.cached,
    }


def read_verdicts(path: Path) -> list[JudgeRecord]:
    records = []
    for number, obj in read_objects(path, JudgeRunError):
        try:
            verdict = obj["verdict"]
            records.append(
                JudgeRecord(
                    item_id=str(obj["item_id"]),
                    judge_pass=int(str(obj["pass"])),
                    cache_key=str(obj["cache_key"]),
                    verdict=None if verdict is None else Verdict.model_validate(verdict),
                    error=None if obj["error"] is None else str(obj["error"]),
                    cached=obj["cached"] is True,
                )
            )
        except (KeyError, ValueError) as exc:
            raise JudgeRunError(f"line {number}: not a judge record") from exc
    return records


def _read_cache(path: Path) -> dict[str, Verdict]:
    cache = {}
    for number, obj in read_objects(path, JudgeRunError):
        try:
            cache[str(obj["key"])] = Verdict.model_validate(obj["verdict"])
        except (KeyError, ValueError) as exc:
            raise JudgeRunError(f"cache line {number}: not a cached verdict") from exc
    return cache


def _open_batches(path: Path, recorded: set[str]) -> list[tuple[str, dict[str, str]]]:
    """Recorded batches with a request whose result is not recorded yet."""
    open_batches = []
    for number, obj in read_objects(path, JudgeRunError):
        requests = obj.get("requests")
        if not isinstance(requests, dict):
            raise JudgeRunError(f"batch line {number}: no requests")
        keys = {str(cid): str(key) for cid, key in requests.items()}
        if set(keys) - recorded:
            open_batches.append((str(obj.get("batch_id")), keys))
    return open_batches


# ---------------------------------------------------------------- settle


@dataclass
class _Pass:
    matrix: AnswerMatrix
    run: str
    files: JudgeFiles
    ledger: Ledger
    port: BatchPort
    sleep: Callable[[float], Awaitable[None]]
    options: JudgeOptions
    judged: int = 0
    errors: int = 0

    async def wait(self, batch_id: str) -> None:
        waited = 0.0
        while not await self.port.ended(batch_id):
            if waited >= self.options.max_wait_seconds:
                raise BatchPendingError(
                    f"batch {batch_id} is still running; run the judge again to collect it"
                )
            await self.sleep(self.options.poll_seconds)
            waited += self.options.poll_seconds

    async def collect(self, batch_id: str, keys: Mapping[str, str], recorded: set[str]) -> None:
        await self.wait(batch_id)
        made: list[Charge] = []
        for outcome in await self.port.results(batch_id):
            if outcome.custom_id not in keys:
                raise JudgeRunError(f"batch {batch_id} returned {outcome.custom_id}, never sent")
            if outcome.custom_id in recorded:
                continue
            item_id, _, judge_pass = outcome.custom_id.rpartition("--p")
            if outcome.input_tokens or outcome.output_tokens:
                made.append(self._charge(item_id, outcome))
            self._record(item_id, int(judge_pass), keys[outcome.custom_id], outcome)
            recorded.add(outcome.custom_id)
        for billed in made:
            check_measured(self.matrix, billed)

    def _charge(self, item_id: str, outcome: BatchOutcome) -> Charge:
        billed = charge(
            self.matrix,
            self.run,
            item_id,
            Kind.JUDGE,
            self.matrix.models.judge,
            outcome.input_tokens,
            outcome.output_tokens,
        )
        self.ledger.append(billed)
        return billed

    def _record(self, item_id: str, judge_pass: int, key: str, outcome: BatchOutcome) -> None:
        verdict, error = None, outcome.error
        if error is None:
            try:
                verdict = parse_verdict(outcome.text or "")
            except JudgeOutputError:
                error = INVALID_VERDICT
        record = JudgeRecord(item_id, judge_pass, key, verdict, error, cached=False)
        append_object(self.files.verdicts, _record_json(record))
        self.judged += 1
        if verdict is None:
            self.errors += 1
        elif judge_pass == 1:
            append_object(
                self.files.cache, {"key": key, "verdict": verdict.model_dump(mode="json")}
            )


# ---------------------------------------------------------------- pass


def _wanted(
    model: str, inputs: Sequence[JudgeInput], repeat: Sequence[str], limit: int | None
) -> list[_Request]:
    first = [_Request(judged, 1, cache_key(model, judged)) for judged in inputs]
    if limit is not None:
        return spread(first, limit)
    again = set(repeat)
    second = [_Request(r.judged, 2, r.key) for r in first if r.judged.item_id in again]
    return first + second


async def judge_run(  # noqa: PLR0913, PLR0917 - the pass's ports and files are named at the call site
    matrix: AnswerMatrix,
    run: str,
    inputs: Sequence[JudgeInput],
    repeat: Sequence[str],
    files: JudgeFiles,
    port: BatchPort,
    ledger: Ledger,
    sleep: Callable[[float], Awaitable[None]],
    options: JudgeOptions,
) -> JudgeSummary:
    state = _Pass(matrix, run, files, ledger, port, sleep, options)
    recorded = {custom_id(r.item_id, r.judge_pass) for r in read_verdicts(files.verdicts)}
    for batch_id, keys in _open_batches(files.batches, recorded):
        await state.collect(batch_id, keys, recorded)

    pending = [
        r
        for r in _wanted(matrix.models.judge, inputs, repeat, options.limit)
        if r.cid not in recorded
    ]
    cache = _read_cache(files.cache)
    hits = [r for r in pending if r.judge_pass == 1 and r.key in cache]
    for hit in hits:
        record = JudgeRecord(hit.judged.item_id, 1, hit.key, cache[hit.key], None, cached=True)
        append_object(files.verdicts, _record_json(record))
    to_send = [r for r in pending if r not in hits]
    if to_send:
        guard_item(
            matrix,
            spent=ledger.spent(),
            worst=judge_cost(matrix, len(to_send), matrix.worst_case),
        )
        requests = {r.cid: request_params(matrix, r.judged) for r in to_send}
        batch_id = await port.submit(requests)
        keys = {r.cid: r.key for r in to_send}
        append_object(
            files.batches, {"schema": BATCH_SCHEMA, "batch_id": batch_id, "requests": keys}
        )
        await state.collect(batch_id, keys, recorded)
    return JudgeSummary(len(to_send), len(hits), state.judged, state.errors)
