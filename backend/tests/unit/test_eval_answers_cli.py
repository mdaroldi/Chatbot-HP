"""The answer benchmark CLI (`python -m app.evaluation.answers`): `estimate`, which
`make eval-estimate` runs on the host before and between runs; `ask`, a run inside the
backend image (fake runtime); and `drive`, its host side (fake command runner).
"""

import dataclasses
import io
import json
import runpy
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from app.core.settings import LLMProviderName, Settings
from app.domain.evaluation import IndexedChunk
from app.domain.retrieval import SearchHit
from app.evaluation.answers import cli
from app.evaluation.answers.answer_matrix import load_answer_matrix, stub_matrix
from app.evaluation.answers.ask import AskError
from app.evaluation.answers.budget import Kind, Ledger, charge
from app.evaluation.answers.cli import JUDGE_CACHE_FILE, LEDGER_FILE, MAX_TURN_CHARS, RUNS_DIR, main
from app.evaluation.answers.config_check import backend_env
from app.evaluation.answers.dataset import load_dataset
from app.evaluation.answers.driver import Completed
from app.evaluation.answers.journal import read_journal
from app.evaluation.answers.judge_run import BatchOutcome, BatchPendingError, read_verdicts
from app.evaluation.answers.runtime import AskRuntime, JudgeRuntime
from app.ingestion.corpus import load_corpus
from tests.unit.conftest import LogReader
from tests.unit.test_eval_answers_ask import FakeBackend, FakeIndex, FakeRows, ready_body
from tests.unit.test_eval_answers_budget import MATRIX, OPUS, SONNET
from tests.unit.test_eval_answers_judge_run import VERDICT
from tests.unit.test_eval_answers_pin import EVAL_DIR
from tests.unit.test_eval_answers_settle import hit

REPO = EVAL_DIR.parent


@pytest.fixture(autouse=True)
def keep_test_log_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() installs the production handler; keep the capture handler instead."""
    monkeypatch.setattr(cli, "configure_logging", lambda *_, **__: None)


def estimate(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["estimate", "--eval-dir", str(EVAL_DIR), *args], stdout=out)
    return code, out.getvalue()


def test_estimate_prints_every_repo_run_and_the_cap(tmp_path: Path) -> None:
    code, text = estimate("--ledger", str(tmp_path / "ledger.jsonl"))
    assert code == 0
    for run in ("reference-1", "heldout", "improvement", "naive", "rewrite-off", "k8"):
        assert f"| {run} |" in text
    assert "| reference-1 | 105 |" in text
    assert "| heldout | 30 |" in text
    assert "| rewrite-off | 10 |" in text
    assert "cap $10.00" in text
    assert "measured $0.0000" in text


def test_estimate_reads_the_ledger_it_is_given(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    made = charge(MATRIX, "reference-1", "q-1", Kind.ANSWER, SONNET, 3000, 400)
    ledger.append(made)
    code, text = estimate("--ledger", str(ledger.path))
    assert code == 0
    assert f"measured ${made.cost_usd:.4f}" in text
    assert "1 measured answer calls" in text


def messages(json_logs: LogReader) -> list[object]:
    return [record["message"] for record in json_logs()]


def test_a_corrupt_ledger_fails_with_one_and_prints_no_plan(
    tmp_path: Path, json_logs: LogReader
) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text("secret partial", encoding="utf-8")
    code, text = estimate("--ledger", str(path))
    assert (code, text) == (1, "")
    assert "ledger is invalid" in messages(json_logs)
    assert "secret" not in json.dumps(json_logs())


def test_a_bad_matrix_fails_with_one(tmp_path: Path, json_logs: LogReader) -> None:
    (tmp_path / "answer-matrix.toml").write_text("[budget]\n", encoding="utf-8")
    out = io.StringIO()
    code = main(["estimate", "--eval-dir", str(tmp_path)], stdout=out)
    assert (code, out.getvalue()) == (1, "")
    assert "answer matrix is invalid" in messages(json_logs)


def test_a_missing_dataset_fails_with_one(tmp_path: Path, json_logs: LogReader) -> None:
    (tmp_path / "answer-matrix.toml").write_bytes((EVAL_DIR / "answer-matrix.toml").read_bytes())
    code = main(["estimate", "--eval-dir", str(tmp_path)], stdout=io.StringIO())
    assert code == 1
    assert "dataset is invalid" in messages(json_logs)


def test_a_missing_answers_file_fails_with_one_naming_it(
    tmp_path: Path, json_logs: LogReader
) -> None:
    (tmp_path / "golden").mkdir()
    for name in ("answer-matrix.toml", "golden/retrieval-v1.toml"):
        (tmp_path / name).write_bytes((EVAL_DIR / name).read_bytes())
    code = main(["estimate", "--eval-dir", str(tmp_path)], stdout=io.StringIO())
    assert code == 1
    (failure,) = [r for r in json_logs() if r["message"] == "dataset is invalid"]
    assert "answers-v1.toml" in failure["detail"]


def test_the_ledger_lives_in_the_git_ignored_run_directory() -> None:
    assert LEDGER_FILE.parent == RUNS_DIR
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert f"/eval/{RUNS_DIR.as_posix()}/" in ignored


def test_bad_arguments_exit_with_two() -> None:
    with pytest.raises(SystemExit) as exited:
        main(["estimate", "--unknown"])
    assert exited.value.code == 2


def test_the_module_runs_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 7)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("app.evaluation.answers", run_name="__main__")
    assert exited.value.code == 7


# ------------------------------------------------------------------ ask


@pytest.fixture
def eval_dir(tmp_path: Path) -> Path:
    """A copy of eval/: ask writes the ledger and the run directory under it."""
    (tmp_path / "golden").mkdir()
    for name in (
        "answer-matrix.toml",
        "golden/retrieval-v1.toml",
        "golden/answers-v1.toml",
        "golden/slices-v1.toml",
    ):
        (tmp_path / name).write_bytes((EVAL_DIR / name).read_bytes())
    return tmp_path


class CorpusChunks:
    """Hits from a real corpus document: the CLI renders them with the corpus titles."""

    async def hits(self, chunk_ids: Sequence[str]) -> Mapping[str, SearchHit]:
        doc_id = load_corpus()[0].doc_id
        return {i: dataclasses.replace(hit(i), doc_id=doc_id) for i in chunk_ids}


def corpus_chunks(collection: str) -> CorpusChunks:
    assert collection.startswith("hpdocs_chunks")
    return CorpusChunks()


class FakeRuntime:
    def __init__(
        self,
        backend: FakeBackend | None = None,
        index: FakeIndex | None = None,
        rows: FakeRows | None = None,
    ) -> None:
        self.backend, self.index = backend or FakeBackend(), index or FakeIndex()
        self.rows = rows or FakeRows()
        self.opened: list[str] = []

    @asynccontextmanager
    async def __call__(self, settings: Settings, api_url: str) -> AsyncIterator[AskRuntime]:
        assert isinstance(settings, Settings)
        self.opened.append(api_url)
        yield AskRuntime(self.backend, self.index, self.rows, corpus_chunks)


def run_ask(eval_dir: Path, runtime: object, *args: str) -> int:
    return main(
        ["ask", "--eval-dir", str(eval_dir), *args],
        stdout=io.StringIO(),
        ask_runtime=runtime,  # type: ignore[arg-type]  # a fake factory
    )


def test_ask_runs_a_pilot_of_the_repo_reference_run_and_logs_its_health(
    eval_dir: Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime()

    code = run_ask(eval_dir, runtime, "--run", "reference-1", "--limit", "3")

    assert code == 0
    assert runtime.opened == ["http://backend:8000"]
    results = read_journal(eval_dir / RUNS_DIR / "reference-1" / "results.jsonl")
    assert len(results) == 3
    assert len({r.client_id for r in results}) == 3  # a fresh client id per item
    assert len(Ledger(eval_dir / LEDGER_FILE).charges()) >= 3
    (done,) = [r for r in json_logs() if r["message"] == "answer run done"]
    assert (done["run"], done["asked"], done["recorded"], done["valid"]) == (
        "reference-1",
        3,
        3,
        True,
    )


def test_ask_takes_the_api_url_it_is_given(eval_dir: Path) -> None:
    runtime = FakeRuntime()
    run_ask(eval_dir, runtime, "--run", "reference-1", "--limit", "1", "--api-url", "http://x")
    assert runtime.opened == ["http://x"]


class _ErrorsForAll(dict[str, str]):
    """Every item fails with a read timeout."""

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: str) -> str:
        return "read_timeout"


def test_an_invalid_run_exits_with_one(eval_dir: Path, json_logs: LogReader) -> None:
    runtime = FakeRuntime(FakeBackend(errors=_ErrorsForAll()))

    code = run_ask(eval_dir, runtime, "--run", "reference-1", "--limit", "2")

    assert code == 1
    (done,) = [r for r in json_logs() if r["message"] == "answer run is invalid"]
    assert done["errors_by_code"] == {"read_timeout": 2}


def test_allow_heldout_on_another_run_is_refused_with_two(
    eval_dir: Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime()
    code = run_ask(eval_dir, runtime, "--run", "reference-1", "--allow-heldout")
    assert code == 2
    assert runtime.opened == []
    assert "--allow-heldout is for the heldout run only" in messages(json_logs)


def test_the_heldout_run_without_allow_heldout_is_refused_with_two(
    eval_dir: Path, json_logs: LogReader
) -> None:
    runtime = FakeRuntime()
    code = run_ask(eval_dir, runtime, "--run", "heldout")
    assert code == 2
    assert runtime.opened == []
    assert "the held-out split needs --allow-heldout" in messages(json_logs)


def test_an_unknown_run_fails_with_one(eval_dir: Path, json_logs: LogReader) -> None:
    code = run_ask(eval_dir, FakeRuntime(), "--run", "nope")
    assert code == 1
    assert "answer matrix is invalid" in messages(json_logs)


def test_a_stopped_run_fails_with_one_naming_the_reason(
    eval_dir: Path, json_logs: LogReader
) -> None:
    code = run_ask(eval_dir, FakeRuntime(index=FakeIndex(None)), "--run", "reference-1")
    assert code == 1
    (stopped,) = [r for r in json_logs() if r["message"] == "answer run stopped"]
    assert stopped["error"] == AskError.__name__
    assert "hpdocs_chunks" in stopped["detail"]


def test_an_unexpected_failure_fails_with_one_and_logs_only_its_type(
    eval_dir: Path, json_logs: LogReader
) -> None:
    def broken(_settings: Settings, _api_url: str) -> object:
        raise ConnectionError("secret detail")

    code = run_ask(eval_dir, broken, "--run", "reference-1")

    assert code == 1
    (failed,) = [r for r in json_logs() if r["message"] == "answer run failed"]
    assert failed["error"] == "ConnectionError"
    assert "secret" not in json.dumps(json_logs())


def test_a_resume_with_nothing_left_exits_with_zero_and_asks_nothing(
    eval_dir: Path, json_logs: LogReader
) -> None:
    rerun = RUNS_DIR / "rewrite-off"
    assert run_ask(eval_dir, FakeRuntime(), "--run", "rewrite-off") == 0
    backend = FakeBackend()

    code = run_ask(eval_dir, FakeRuntime(backend), "--run", "rewrite-off", "--resume")

    assert code == 0
    assert backend.log == []
    last = [r for r in json_logs() if r["message"] == "answer run done"][-1]
    assert (last["asked"], last["recorded"]) == (0, 10)
    assert "valid" not in last
    assert (eval_dir / rerun / "config.json").exists()


# ------------------------------------------------------------------ drive


class DriveRunners:
    """A runner factory whose runner answers every docker step as the backend would."""

    def __init__(self, ask_code: int = 0, fail_up: bool = False) -> None:
        self.ask_code, self.fail_up = ask_code, fail_up
        self.cwd: list[Path] = []
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, cwd: Path) -> object:
        self.cwd.append(cwd)
        matrix = load_answer_matrix(EVAL_DIR / "answer-matrix.toml")

        def run(argv: Sequence[str], *, env: Mapping[str, str], capture: bool) -> Completed:
            call = tuple(argv)
            self.calls.append(call)
            assert capture == (call[2] != "run")  # only the harness streams its logs
            if call[2] == "up":
                return Completed(1 if self.fail_up else 0, "")
            if call[2] == "exec":
                expected = backend_env(matrix, matrix.run(env_run(env)))
                return Completed(0, "".join(f"{v}\n" for v in expected.values()))
            return Completed(self.ask_code, "")

        return run


def env_run(env: Mapping[str, str]) -> str:
    return "k3" if env.get("RETRIEVAL_TOP_K") == "3" else "reference-1"


def run_drive(runners: DriveRunners, *args: str) -> int:
    return main(
        ["drive", "--eval-dir", str(EVAL_DIR), *args],
        stdout=io.StringIO(),
        drive_runner=runners,  # type: ignore[arg-type]  # a fake factory
        environ={"PATH": "/usr/bin"},
    )


def test_drive_runs_compose_from_the_repo_root_and_passes_ask_options_through() -> None:
    runners = DriveRunners()

    code = run_drive(
        runners, "--run", "k3", "--limit", "10", "--resume", "--retry-errors", "--allow-heldout"
    )

    assert code == 0
    assert runners.cwd == [REPO.resolve()]
    ask = next(call for call in runners.calls if call[2] == "run")
    assert f"{EVAL_DIR.resolve()}:/eval" in ask
    assert ask[ask.index("ask") :] == (
        "ask",
        "--run",
        "k3",
        "--limit",
        "10",
        "--resume",
        "--retry-errors",
        "--allow-heldout",
    )


def test_drive_passes_only_the_options_given() -> None:
    runners = DriveRunners()
    run_drive(runners, "--run", "reference-1")
    ask = next(call for call in runners.calls if call[2] == "run")
    assert ask[-3:] == ("ask", "--run", "reference-1")


def test_drive_exits_with_the_ask_exit_code() -> None:
    assert run_drive(DriveRunners(ask_code=1), "--run", "k3") == 1


def test_a_failed_docker_step_stops_the_drive_with_one(json_logs: LogReader) -> None:
    code = run_drive(DriveRunners(fail_up=True), "--run", "k3")
    assert code == 1
    (stopped,) = [r for r in json_logs() if r["message"] == "answer run stopped"]
    assert stopped["error"] == "DriverError"


def test_a_stub_ask_uses_its_own_run_directory_and_ledger(eval_dir: Path) -> None:
    stub_rows = FakeRows(model="stub")  # what the stub provider stores
    runtime = FakeRuntime(FakeBackend(ready=ready_body(llm="stub")), rows=stub_rows)

    code = run_ask(eval_dir, runtime, "--run", "reference-1", "--limit", "2", "--provider", "stub")

    assert code == 0
    stub_dir = eval_dir / RUNS_DIR / "stub" / "reference-1"
    assert len(read_journal(stub_dir / "results.jsonl")) == 2
    charges = Ledger(stub_dir / "ledger.jsonl").charges()
    assert charges
    assert {(c.model, c.cost_usd) for c in charges} == {("stub", 0.0)}
    assert not (eval_dir / LEDGER_FILE).exists()
    assert not (eval_dir / RUNS_DIR / "reference-1").exists()


def test_a_stub_drive_uses_the_stub_matrix_and_tells_ask() -> None:
    runners = StubDriveRunners()

    code = run_drive(runners, "--run", "k3", "--provider", "stub")

    assert code == 0
    ask = next(call for call in runners.calls if call[2] == "run")
    assert ask[-2:] == ("--provider", "stub")


class StubDriveRunners(DriveRunners):
    """Answers printenv as a backend restarted with the stub pins would."""

    def __call__(self, cwd: Path) -> object:
        stub = stub_matrix(load_answer_matrix(EVAL_DIR / "answer-matrix.toml"))

        def run(argv: Sequence[str], *, env: Mapping[str, str], capture: bool) -> Completed:
            call = tuple(argv)
            self.calls.append(call)
            assert capture == (call[2] != "run")
            if call[2] == "exec":
                expected = backend_env(stub, stub.run("k3"), LLMProviderName.STUB)
                assert {k: env[k] for k in expected} == expected
                return Completed(0, "".join(f"{v}\n" for v in expected.values()))
            return Completed(0, "")

        self.cwd.append(cwd)
        return run


# ------------------------------------------------------------------ judge


def anchor_chunks() -> list[IndexedChunk]:
    """One indexed chunk per dataset anchor, so every anchor resolves as in the index."""
    doc_ids = {document.doc_id for document in load_corpus()}
    dataset = load_dataset(EVAL_DIR, doc_ids, max_turn_chars=MAX_TURN_CHARS)
    anchors = {anchor for item in dataset.items for anchor in item.anchors}
    return [
        IndexedChunk(f"a-{n}", a.doc_id, (a.section or "Section",), " ".join(a.facts))
        for n, a in enumerate(sorted(anchors, key=repr))
    ]


class FakePort:
    def __init__(self, pending: bool = False) -> None:
        self.sent: list[dict[str, Mapping[str, object]]] = []
        self.pending = pending

    async def submit(self, requests: Mapping[str, Mapping[str, object]]) -> str:
        self.sent.append(dict(requests))
        return f"batch-{len(self.sent)}"

    async def ended(self, batch_id: str) -> bool:
        if self.pending:
            raise BatchPendingError(f"batch {batch_id} is still running")
        return True

    async def results(self, batch_id: str) -> list[BatchOutcome]:
        sent = self.sent[int(batch_id.removeprefix("batch-")) - 1]
        return [BatchOutcome(cid, VERDICT, 3000, 400, None) for cid in sent]


class FakeJudgeRuntime:
    def __init__(self, port: FakePort | None = None) -> None:
        self.port = port or FakePort()
        self.aliases: list[str] = []

    @asynccontextmanager
    async def __call__(self, settings: Settings) -> AsyncIterator[JudgeRuntime]:
        assert isinstance(settings, Settings)
        yield JudgeRuntime(self.port, self.chunks)

    async def chunks(self, alias: str) -> list[IndexedChunk]:
        self.aliases.append(alias)
        return anchor_chunks()


def run_judge(eval_dir: Path, runtime: FakeJudgeRuntime, *args: str) -> int:
    return main(
        ["judge", "--eval-dir", str(eval_dir), *args],
        stdout=io.StringIO(),
        judge_runtime=runtime,
    )


def test_a_judge_smoke_judges_a_spread_of_answers_once_and_charges_the_ledger(
    eval_dir: Path, json_logs: LogReader
) -> None:
    run_ask(eval_dir, FakeRuntime(), "--run", "reference-1", "--limit", "6")
    runtime = FakeJudgeRuntime()

    code = run_judge(eval_dir, runtime, "--run", "reference-1", "--limit", "2")

    assert code == 0
    assert runtime.aliases == ["hpdocs_chunks"]
    [batch] = runtime.port.sent
    assert len(batch) == 2
    assert all(cid.endswith("--p1") for cid in batch)
    assert {params["model"] for params in batch.values()} == {OPUS}
    records = read_verdicts(eval_dir / RUNS_DIR / "reference-1" / "judge.jsonl")
    assert [r.verdict is not None for r in records] == [True, True]
    assert (eval_dir / JUDGE_CACHE_FILE).exists()
    judge_charges = [c for c in Ledger(eval_dir / LEDGER_FILE).charges() if c.kind is Kind.JUDGE]
    assert len(judge_charges) == 2
    (done,) = [r for r in json_logs() if r["message"] == "judge pass done"]
    assert (done["run"], done["submitted"], done["cached"], done["recorded"]) == (
        "reference-1",
        2,
        0,
        2,
    )


def test_a_full_judge_pass_judges_every_in_scope_answer_and_the_repeat_sample(
    eval_dir: Path,
) -> None:
    run_ask(eval_dir, FakeRuntime(), "--run", "reference-1")
    runtime = FakeJudgeRuntime()

    assert run_judge(eval_dir, runtime, "--run", "reference-1") == 0

    [batch] = runtime.port.sent
    first = [cid for cid in batch if cid.endswith("--p1")]
    second = [cid for cid in batch if cid.endswith("--p2")]
    assert len(first) == 72  # every in-scope dev item (the fake answers all of them)
    assert len(second) == 20  # judge_repeat in the matrix


def test_a_second_judge_call_reuses_recorded_verdicts_and_pays_nothing(eval_dir: Path) -> None:
    run_ask(eval_dir, FakeRuntime(), "--run", "reference-1", "--limit", "6")
    run_judge(eval_dir, FakeJudgeRuntime(), "--run", "reference-1", "--limit", "2")
    runtime = FakeJudgeRuntime()

    assert run_judge(eval_dir, runtime, "--run", "reference-1", "--limit", "2") == 0
    assert runtime.port.sent == []


def test_a_full_pass_on_an_unfinished_run_stops_with_one_before_any_batch(
    eval_dir: Path, json_logs: LogReader
) -> None:
    run_ask(eval_dir, FakeRuntime(), "--run", "reference-1", "--limit", "3")
    runtime = FakeJudgeRuntime()

    assert run_judge(eval_dir, runtime, "--run", "reference-1") == 1
    assert runtime.port.sent == []
    (stopped,) = [r for r in json_logs() if r["message"] == "answer run stopped"]
    assert stopped["error"] == "RunNotFinishedError"


def test_a_run_the_matrix_does_not_judge_is_refused_with_two(
    eval_dir: Path, json_logs: LogReader
) -> None:
    assert run_judge(eval_dir, FakeJudgeRuntime(), "--run", "reference-2") == 2
    assert any("not judged" in r["message"] for r in json_logs())


def test_the_heldout_run_is_judged_only_with_allow_heldout(eval_dir: Path) -> None:
    assert run_judge(eval_dir, FakeJudgeRuntime(), "--run", "heldout") == 2


def test_a_batch_still_running_stops_with_one_and_names_the_batch(
    eval_dir: Path, json_logs: LogReader
) -> None:
    run_ask(eval_dir, FakeRuntime(), "--run", "reference-1", "--limit", "6")

    code = run_judge(
        eval_dir, FakeJudgeRuntime(FakePort(pending=True)), "--run", "reference-1", "--limit", "1"
    )

    assert code == 1
    (stopped,) = [r for r in json_logs() if r["message"] == "answer run stopped"]
    assert stopped["error"] == "BatchPendingError"
    assert "batch-1" in stopped["detail"]
