"""The answer benchmark CLI (`python -m app.evaluation.answers`); here its `estimate`
command, which `make eval-estimate` runs on the host before and between runs.
"""

import io
import json
import runpy
from pathlib import Path

import pytest

from app.evaluation.answers import cli
from app.evaluation.answers.budget import Kind, Ledger, charge
from app.evaluation.answers.cli import LEDGER_FILE, RUNS_DIR, main
from tests.unit.conftest import LogReader
from tests.unit.test_eval_answers_budget import MATRIX, SONNET
from tests.unit.test_eval_answers_pin import EVAL_DIR

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
