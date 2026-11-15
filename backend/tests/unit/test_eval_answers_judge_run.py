"""One judge pass over a run through the Batch API: requests built from the judge inputs,
the batch recorded before waiting (so a crash never pays twice), measured charges in
the ledger, verdicts and judge errors recorded, the cache, the repeat pass and the cap.
"""

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from app.evaluation.answers.budget import (
    BudgetExceededError,
    Kind,
    Ledger,
    UnsoundBoundError,
    charge,
    judge_cost,
)
from app.evaluation.answers.judge import JudgeInput, cache_key, custom_id
from app.evaluation.answers.judge_run import (
    BatchOutcome,
    BatchPendingError,
    JudgeFiles,
    JudgeOptions,
    JudgeRunError,
    JudgeSummary,
    judge_run,
    read_verdicts,
)
from tests.unit.test_eval_answers_budget import MATRIX, OPUS


def judge_input(item_id: str) -> JudgeInput:
    return JudgeInput(item_id, ("q",), ("<excerpt>e</excerpt>",), ("p",), (("f",),), f"a {item_id}")


VERDICT = json.dumps(
    {
        "claims": [{"claim": "c", "support": "unsupported", "tags": []}],
        "correctness": "partial",
        "reasons": ["extra_unsupported"],
    }
)


def ok(cid: str, text: str = VERDICT, tokens: tuple[int, int] = (3000, 900)) -> BatchOutcome:
    return BatchOutcome(cid, text, tokens[0], tokens[1], None)


class FakeBatch:
    def __init__(self, polls_until_ended: int = 0) -> None:
        self.submitted: list[dict[str, Mapping[str, object]]] = []
        self.outcomes: dict[str, BatchOutcome] = {}  # by custom id; default: ok
        self.extra: list[BatchOutcome] = []
        self.polls_until_ended = polls_until_ended
        self.polls = 0

    async def submit(self, requests: Mapping[str, Mapping[str, object]]) -> str:
        self.submitted.append(dict(requests))
        return f"batch-{len(self.submitted)}"

    async def ended(self, batch_id: str) -> bool:
        assert batch_id.startswith("batch-")
        self.polls += 1
        return self.polls > self.polls_until_ended

    async def results(self, batch_id: str) -> list[BatchOutcome]:
        requests = self.submitted[int(batch_id.split("-")[1]) - 1]
        return [self.outcomes.get(cid, ok(cid)) for cid in requests] + self.extra


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def files(tmp_path: Path) -> JudgeFiles:
    return JudgeFiles(
        tmp_path / "judge.jsonl", tmp_path / "judge-batches.jsonl", tmp_path / "cache.jsonl"
    )


async def run(
    files: JudgeFiles,
    batch: FakeBatch,
    ledger: Ledger,
    inputs: list[JudgeInput],
    *,
    repeat: tuple[str, ...] = (),
    options: JudgeOptions | None = None,
    sleep: Sleeps | None = None,
) -> JudgeSummary:
    return await judge_run(
        MATRIX,
        "reference-1",
        inputs,
        repeat,
        files,
        batch,
        ledger,
        sleep or Sleeps(),
        options or JudgeOptions(poll_seconds=5.0, max_wait_seconds=60.0),
    )


async def test_every_input_is_judged_in_one_batch_and_charged_at_measured_tokens(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    inputs = [judge_input("a"), judge_input("b")]

    summary = await run(files, batch, ledger, inputs)

    assert list(batch.submitted[0]) == ["a--p1", "b--p1"]
    assert batch.submitted[0]["a--p1"]["model"] == OPUS
    records = read_verdicts(files.verdicts)
    assert [(r.item_id, r.judge_pass, r.error, r.cached) for r in records] == [
        ("a", 1, None, False),
        ("b", 1, None, False),
    ]
    assert records[0].verdict is not None
    assert records[0].verdict.faithful is False
    assert records[0].cache_key == cache_key(OPUS, inputs[0])
    expected = charge(MATRIX, "reference-1", "a", Kind.JUDGE, OPUS, 3000, 900)
    assert ledger.charges() == [expected, dataclasses.replace(expected, item_id="b")]
    assert (summary.submitted, summary.cached, summary.judged, summary.errors) == (2, 0, 2, 0)


async def test_a_second_pass_over_recorded_items_submits_nothing(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    await run(files, batch, ledger, [judge_input("a")])

    summary = await run(files, batch, ledger, [judge_input("a")])

    assert len(batch.submitted) == 1
    assert summary.submitted == summary.judged == 0
    assert len(ledger.charges()) == 1


async def test_a_cached_verdict_is_recorded_free_without_a_request(
    files: JudgeFiles, tmp_path: Path
) -> None:
    first = dataclasses.replace(
        files, verdicts=tmp_path / "other-run.jsonl", batches=tmp_path / "other-batches.jsonl"
    )
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    await run(first, batch, ledger, [judge_input("a")])

    summary = await run(files, batch, ledger, [judge_input("a")])

    assert len(batch.submitted) == 1
    (record,) = read_verdicts(files.verdicts)
    assert record.cached
    assert record.verdict is not None
    assert record.error is None
    assert len(ledger.charges()) == 1
    assert (summary.submitted, summary.cached) == (0, 1)


async def test_the_repeat_sample_is_judged_again_bypassing_the_cache(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    await run(files, batch, ledger, [judge_input("a"), judge_input("b")])

    await run(files, batch, ledger, [judge_input("a"), judge_input("b")], repeat=("b",))

    assert list(batch.submitted[1]) == ["b--p2"]
    passes = [(r.item_id, r.judge_pass, r.cached) for r in read_verdicts(files.verdicts)]
    assert passes[-1] == ("b", 2, False)
    cache_lines = files.cache.read_text().splitlines()
    assert len(cache_lines) == 2  # the second pass is never cached


@pytest.mark.parametrize(
    ("outcome", "error", "charged"),
    [
        (BatchOutcome("a--p1", None, 0, 0, "errored"), "errored", False),
        (BatchOutcome("a--p1", None, 0, 0, "expired"), "expired", False),
        (BatchOutcome("a--p1", None, 2000, 4000, "max_tokens"), "max_tokens", True),
        (BatchOutcome("a--p1", None, 2000, 10, "refusal"), "refusal", True),
        (ok("a--p1", text='{"claims": []}'), "invalid_verdict", True),
    ],
)
async def test_judge_errors_are_recorded_and_billed_only_when_tokens_were_used(
    files: JudgeFiles, tmp_path: Path, outcome: BatchOutcome, error: str, charged: bool
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    batch.outcomes["a--p1"] = outcome

    summary = await run(files, batch, ledger, [judge_input("a")])

    (record,) = read_verdicts(files.verdicts)
    assert (record.error, record.verdict) == (error, None)
    assert bool(ledger.charges()) is charged
    assert summary.errors == 1
    assert not files.cache.exists()


async def test_the_cap_is_checked_at_the_worst_case_before_anything_is_submitted(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    worst_two = judge_cost(MATRIX, 2, MATRIX.worst_case)
    big = MATRIX.cap_usd - worst_two + 0.01
    ledger.append(charge(MATRIX, "reference-1", "x", Kind.ANSWER, "claude-sonnet-5-5", 0, 0))
    ledger.append(
        dataclasses.replace(ledger.charges()[0], cost_usd=big)  # spent close to the cap
    )

    with pytest.raises(BudgetExceededError):
        await run(files, batch, ledger, [judge_input("a"), judge_input("b")])
    assert batch.submitted == []
    assert not files.batches.exists()


async def test_a_batch_left_running_is_collected_on_the_next_call_not_resubmitted(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(polls_until_ended=100), Ledger(tmp_path / "ledger.jsonl")
    sleep = Sleeps()

    with pytest.raises(BatchPendingError, match="batch-1"):
        await run(files, batch, ledger, [judge_input("a")], sleep=sleep)
    assert sum(sleep.calls) >= 60.0
    assert read_verdicts(files.verdicts) == []
    assert files.batches.exists()

    batch.polls_until_ended = 0
    batch.polls = 0
    summary = await run(files, batch, ledger, [judge_input("a")])

    assert len(batch.submitted) == 1
    assert [r.item_id for r in read_verdicts(files.verdicts)] == ["a"]
    assert summary.judged == 1


async def test_a_limit_judges_an_even_spread_of_first_passes_only(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    inputs = [judge_input(f"q-{n}") for n in range(9)]

    await run(files, batch, ledger, inputs, repeat=("q-0",), options=JudgeOptions(limit=3))

    assert list(batch.submitted[0]) == ["q-0--p1", "q-3--p1", "q-6--p1"]


async def test_a_measured_call_above_its_bound_stops_after_everything_is_recorded(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    too_big = (MATRIX.worst_case.judge_input_tokens + 1, 10)
    batch.outcomes["a--p1"] = ok("a--p1", tokens=too_big)

    with pytest.raises(UnsoundBoundError):
        await run(files, batch, ledger, [judge_input("a"), judge_input("b")])
    assert len(read_verdicts(files.verdicts)) == 2
    assert len(ledger.charges()) == 2


async def test_a_result_for_a_request_never_sent_is_an_error(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    batch.extra = [ok(custom_id("zzz", 1))]

    with pytest.raises(JudgeRunError, match="zzz--p1"):
        await run(files, batch, ledger, [judge_input("a")])


async def test_a_damaged_verdicts_file_is_an_error(files: JudgeFiles) -> None:
    files.verdicts.write_text('{"schema": "answers-judge-v1"}\n')
    with pytest.raises(JudgeRunError, match="line 1"):
        read_verdicts(files.verdicts)


async def test_damaged_cache_and_batch_files_are_errors(files: JudgeFiles, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    files.cache.write_text('{"key": "k"}\n')
    with pytest.raises(JudgeRunError, match="cache line 1"):
        await run(files, FakeBatch(), ledger, [judge_input("a")])
    files.cache.unlink()
    files.batches.write_text('{"batch_id": "b"}\n')
    with pytest.raises(JudgeRunError, match="batch line 1"):
        await run(files, FakeBatch(), ledger, [judge_input("a")])


async def test_a_batch_partly_recorded_before_a_crash_records_only_the_rest(
    files: JudgeFiles, tmp_path: Path
) -> None:
    batch, ledger = FakeBatch(), Ledger(tmp_path / "ledger.jsonl")
    await run(files, batch, ledger, [judge_input("a"), judge_input("b")])
    first_line = files.verdicts.read_text().splitlines()[0]
    files.verdicts.write_text(first_line + "\n")  # b's line was lost in the crash

    summary = await run(files, batch, ledger, [judge_input("a"), judge_input("b")])

    assert len(batch.submitted) == 1
    assert [r.item_id for r in read_verdicts(files.verdicts)] == ["a", "b"]
    assert summary.judged == 1
