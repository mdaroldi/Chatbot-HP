"""The host-side driver: restart the backend with the run's settings, check its env,
run `ask` in a harness container, and always restore the backend's defaults.
A fake command runner records each docker command and its environment.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from app.core.settings import LLMProviderName
from app.evaluation.answers.answer_matrix import parse_answer_matrix, stub_matrix
from app.evaluation.answers.config_check import (
    CHECKED_NAMES,
    ConfigMismatchError,
    backend_env,
    overrides,
)
from app.evaluation.answers.driver import (
    Completed,
    DriveRequest,
    DriverError,
    drive,
    subprocess_runner,
)
from tests.unit.test_eval_answers_matrix import BASE, HELDOUT, REFERENCE

K3 = """
[[runs]]
name = "k3"
items = "answers-dev"
env = { RETRIEVAL_TOP_K = "3" }
"""
MATRIX = parse_answer_matrix(BASE + REFERENCE + HELDOUT + K3)
RUN = MATRIX.run("k3")
EVAL_DIR = Path("/repo/eval")
ENVIRON = {"PATH": "/usr/bin", "RETRIEVAL_TOP_K": "9", "ANTHROPIC_API_KEY": "sk-test-value"}

UP = ("docker", "compose", "up", "--detach", "--no-deps", "--no-build", "--wait", "backend")
PRINTENV = ("docker", "compose", "exec", "-T", "backend", "printenv", *CHECKED_NAMES)


def printed(values: Mapping[str, str]) -> str:
    return "".join(f"{value}\n" for value in values.values())


class FakeRunner:
    def __init__(
        self,
        *,
        fail: Sequence[str] = (),
        printenv: str | None = None,
        ask_code: int = 0,
        fail_restore: bool = False,
    ) -> None:
        self.fail, self.ask_code, self.fail_restore = set(fail), ask_code, fail_restore
        self.printenv = printed(backend_env(MATRIX, RUN)) if printenv is None else printenv
        self.calls: list[tuple[tuple[str, ...], dict[str, str], bool]] = []

    def __call__(self, argv: Sequence[str], *, env: Mapping[str, str], capture: bool) -> Completed:
        call = tuple(argv)
        self.calls.append((call, dict(env), capture))
        if call == UP:
            restore = len([c for c in self.calls if c[0] == UP]) > 1
            failed = self.fail_restore if restore else "up" in self.fail
            return Completed(1 if failed else 0, "")
        if call == PRINTENV:
            return Completed(1 if "printenv" in self.fail else 0, self.printenv)
        return Completed(self.ask_code, "")

    def commands(self) -> list[str]:
        names = {UP: "up", PRINTENV: "printenv"}
        return [names.get(call, "ask") for call, _, _ in self.calls]


def request(*ask_args: str) -> DriveRequest:
    return DriveRequest(RUN, EVAL_DIR, ask_args, uid=501, gid=20)


def test_a_run_restarts_checks_asks_and_restores_in_that_order() -> None:
    runner = FakeRunner()

    code = drive(MATRIX, request("--limit", "10"), runner, ENVIRON)

    assert code == 0
    assert runner.commands() == ["up", "printenv", "ask", "up"]


def test_the_backend_runs_with_the_run_overrides_over_the_callers_env() -> None:
    runner = FakeRunner()

    drive(MATRIX, request(), runner, ENVIRON)

    (_, restart_env, _), (_, printenv_env, capture), *_ = runner.calls
    assert restart_env == ENVIRON | overrides(RUN)
    assert restart_env["RETRIEVAL_TOP_K"] == "3"
    assert restart_env["LLM_PROVIDER"] == "anthropic"
    assert printenv_env == restart_env
    assert capture


def test_the_defaults_are_restored_with_the_callers_env_alone() -> None:
    runner = FakeRunner()

    drive(MATRIX, request(), runner, ENVIRON)

    restore, restore_env, _ = runner.calls[-1]
    assert restore == UP
    assert restore_env == ENVIRON


def test_the_harness_container_gets_the_eval_dir_the_callers_user_and_no_key() -> None:
    runner = FakeRunner()

    drive(MATRIX, request("--limit", "10", "--resume"), runner, ENVIRON)

    ask, _, capture = runner.calls[2]
    assert ask == (
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "--user",
        "501:20",
        "--volume",
        "/repo/eval:/eval",
        "--env",
        "LLM_PROVIDER=stub",
        "--env",
        "ANTHROPIC_API_KEY=",
        "--entrypoint",
        "python",
        "backend",
        "-m",
        "app.evaluation.answers",
        "ask",
        "--run",
        "k3",
        "--limit",
        "10",
        "--resume",
    )
    assert not capture  # its logs stream to the terminal
    assert all("sk-test-value" not in part for part in ask)


def test_the_ask_exit_code_is_the_drivers() -> None:
    runner = FakeRunner(ask_code=1)
    assert drive(MATRIX, request(), runner, ENVIRON) == 1
    assert runner.commands()[-1] == "up"


def test_a_backend_env_that_differs_stops_before_asking_and_still_restores() -> None:
    wrong = backend_env(MATRIX, RUN) | {"LLM_ANSWER_MODEL": "claude-other"}
    runner = FakeRunner(printenv=printed(wrong))

    with pytest.raises(ConfigMismatchError, match="LLM_ANSWER_MODEL"):
        drive(MATRIX, request(), runner, ENVIRON)

    assert runner.commands() == ["up", "printenv", "up"]


@pytest.mark.parametrize(
    ("step", "done"), [("up", ["up", "up"]), ("printenv", ["up", "printenv", "up"])]
)
def test_a_failed_docker_step_stops_before_asking_and_still_restores(
    step: str, done: list[str]
) -> None:
    runner = FakeRunner(fail=[step])

    with pytest.raises(DriverError, match=step):
        drive(MATRIX, request(), runner, ENVIRON)

    assert runner.commands() == done


def test_a_failed_restore_after_a_run_is_an_error() -> None:
    runner = FakeRunner(fail_restore=True)

    with pytest.raises(DriverError, match="restore"):
        drive(MATRIX, request(), runner, ENVIRON)

    assert runner.commands() == ["up", "printenv", "ask", "up"]


def test_a_failed_restore_never_hides_the_error_that_stopped_the_run() -> None:
    runner = FakeRunner(fail=["printenv"], fail_restore=True)

    with pytest.raises(DriverError, match="printenv"):
        drive(MATRIX, request(), runner, ENVIRON)


# ------------------------------------------------------------------ the real runner


def test_the_subprocess_runner_captures_output_in_its_directory(tmp_path: Path) -> None:
    run = subprocess_runner(tmp_path)
    script = "import os; print(os.getcwd()); print(os.environ['ONLY'])"

    done = run([sys.executable, "-c", script], env={"ONLY": "this"}, capture=True)

    assert done.returncode == 0
    assert done.stdout.splitlines() == [os.path.realpath(tmp_path), "this"]


def test_the_subprocess_runner_passes_the_exit_code_through_uncaptured(tmp_path: Path) -> None:
    done = subprocess_runner(tmp_path)(
        [sys.executable, "-c", "raise SystemExit(3)"], env={}, capture=False
    )
    assert done == Completed(3, "")


def test_a_stub_drive_pins_the_whole_checked_env_and_asks_with_the_stub() -> None:
    stub = stub_matrix(MATRIX)
    expected = backend_env(stub, RUN, LLMProviderName.STUB)
    runner = FakeRunner(printenv=printed(expected))
    stub_request = DriveRequest(RUN, EVAL_DIR, ("--limit", "3"), 501, 20, LLMProviderName.STUB)

    assert drive(stub, stub_request, runner, ENVIRON) == 0

    (_, restart_env, _), _, (ask, _, _), (_, restore_env, _) = runner.calls
    assert restart_env == ENVIRON | expected
    assert restart_env["LLM_ANSWER_MODEL"] == "stub"
    assert ask[-5:] == ("k3", "--limit", "3", "--provider", "stub")
    assert restore_env == ENVIRON


def test_a_real_drive_never_passes_a_provider_to_ask() -> None:
    runner = FakeRunner()
    drive(MATRIX, request(), runner, ENVIRON)
    assert "--provider" not in runner.calls[2][0]
