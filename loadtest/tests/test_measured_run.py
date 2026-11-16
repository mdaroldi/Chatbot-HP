"""The measured-run wrapper (ADR 0013) holds four guarantees: the whole run is kept awake
with `caffeinate -i -s`, it starts from `make load-reset`, it records host snapshots
before the reset, at start and at end, and it fails when pmset logged a sleep, wake or
dark wake inside the run window. The script runs here against fake host tools on PATH,
so the guarantees are checked as behavior on any OS, not only on the macOS host."""

import os
import pathlib
import re
import stat
import subprocess  # nosec B404: runs the repo's own script against fakes, no user input

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "measured_run.sh"
MAKEFILE = ROOT.parent / "Makefile"

# Each fake appends one line per call to calls.log, so the tests can check the order.
FAKES = {
    "caffeinate": 'echo "caffeinate $1 $2" >> "$FAKE_DIR/calls.log"; shift 2; exec "$@"',
    "sysctl": 'echo "vm.swapusage: total = 3072.00M  used = 1065.00M  free = 2007.00M"',
    "memory_pressure": 'echo "System-wide memory free percentage: 40%"',
    "vm_stat": 'printf "Pageins: 10.\\nPageouts: 20.\\nSwapins: 30.\\nSwapouts: 40.\\n"',
    "make": (
        'echo "make $*" >> "$FAKE_DIR/calls.log"\n'
        'echo "fake make $1"\n'
        'if [ "$1" = load-reset ]; then exit "${FAKE_RESET_RC:-0}"; fi\n'
        'exit "${FAKE_LOAD_RC:-0}"'
    ),
    # A clock that moves one second per call. A full run reads it six times: window start,
    # the three snapshots, window end, so the window is 12:00:01 .. 12:00:05.
    "date": (
        'n=$(cat "$FAKE_DIR/clock" 2>/dev/null || echo 0); n=$((n + 1))\n'
        'echo "$n" > "$FAKE_DIR/clock"\n'
        'printf "2026-11-16 12:00:%02d\\n" "$n"'
    ),
    # Events one second before and after the window never count; FAKE_EVENT lands inside.
    "pmset": (
        'echo "pmset $*" >> "$FAKE_DIR/calls.log"\n'
        'echo "2026-11-16 12:00:00 +0100 Sleep               \tEntering Sleep (before)"\n'
        'echo "2026-11-16 12:00:03 +0100 Wake Requests       \t[process=powerd]"\n'
        'if [ -n "${FAKE_EVENT:-}" ]; then\n'
        '  echo "2026-11-16 12:00:03 +0100 $FAKE_EVENT  \tEntering $FAKE_EVENT state"\n'
        "fi\n"
        'echo "2026-11-16 12:00:06 +0100 Wake                \tWake from Deep Idle (after)"'
    ),
}


@pytest.fixture(scope="module")
def fake_bin(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Written once per module: a host scanner may hold each new executable for a second."""
    bin_dir = tmp_path_factory.mktemp("bin")
    for name, body in FAKES.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def run_script(
    bin_dir: pathlib.Path, tmp_path: pathlib.Path, *args: str, **env: str
) -> subprocess.CompletedProcess[str]:
    environment = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_DIR": str(tmp_path),
        "MAKE": str(bin_dir / "make"),
        **env,
    }
    return subprocess.run(  # noqa: S603  # nosec B603: fixed argv, the repo's own script
        ["/bin/bash", str(SCRIPT), *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def calls(tmp_path: pathlib.Path) -> list[str]:
    return (tmp_path / "calls.log").read_text().splitlines()


def runner_log(tmp_path: pathlib.Path, run: str = "r1") -> str:
    return (tmp_path / "loadtest" / "results" / f"{run}.runner.log").read_text()


# ---------------------------------------------------------------- caffeinate


def test_everything_runs_under_caffeinate_idle_and_system_sleep(
    fake_bin: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    done = run_script(fake_bin, tmp_path, "r1")

    assert done.returncode == 0, done.stderr
    assert calls(tmp_path)[0] == "caffeinate -i -s"
    assert len([line for line in calls(tmp_path) if line.startswith("caffeinate")]) == 1


# ---------------------------------------------------------------- reset, then the run


def test_the_run_starts_from_a_reset_and_passes_its_variables_to_make_load(
    fake_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    run_script(fake_bin, tmp_path, "r1", "LOAD_STEPS=2,4", "LOAD_STEP_SECONDS=300")

    makes = [line for line in calls(tmp_path) if line.startswith("make")]
    assert makes == [
        "make load-reset",
        "make load LOAD_RUN=r1 LOAD_STEPS=2,4 LOAD_STEP_SECONDS=300",
    ]


def test_a_failed_reset_stops_before_the_run(
    fake_bin: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    done = run_script(fake_bin, tmp_path, "r1", FAKE_RESET_RC="2")

    assert done.returncode == 1
    assert "make load LOAD_RUN=r1" not in calls(tmp_path)
    assert "reset failed" in runner_log(tmp_path)


def test_a_failed_load_run_keeps_its_exit_code_and_still_checks_the_host(
    fake_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    done = run_script(fake_bin, tmp_path, "r1", FAKE_LOAD_RC="4")

    assert done.returncode == 4
    assert "[end]" in runner_log(tmp_path)
    assert calls(tmp_path)[-1] == "pmset -g log"


# ---------------------------------------------------------------- host snapshots


def test_snapshots_are_taken_before_the_reset_at_start_and_at_end(
    fake_bin: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    run_script(fake_bin, tmp_path, "r1")

    log = runner_log(tmp_path)
    order = [
        log.index("[pre-reset]"),
        log.index("fake make load-reset"),
        log.index("[start]"),
        log.index("fake make load\n"),
        log.index("[end]"),
    ]
    assert order == sorted(order)


@pytest.mark.parametrize("label", ["pre-reset", "start", "end"])
def test_each_snapshot_records_time_swap_memory_and_paging(
    fake_bin: pathlib.Path, tmp_path: pathlib.Path, label: str
) -> None:
    run_script(fake_bin, tmp_path, "r1")

    block = runner_log(tmp_path).split(f"[{label}] ")[1]
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\n", block)
    for expected in ("vm.swapusage", "memory free percentage", "Pageouts", "Swapouts"):
        assert expected in block.split("\n[")[0], expected


# ---------------------------------------------------------------- sleep and wake check


@pytest.mark.parametrize("event", ["Sleep", "Wake", "DarkWake"])
def test_a_sleep_or_wake_inside_the_window_voids_the_run(
    fake_bin: pathlib.Path, tmp_path: pathlib.Path, event: str
) -> None:
    done = run_script(fake_bin, tmp_path, "r1", FAKE_EVENT=event)

    assert done.returncode == 3
    log = runner_log(tmp_path)
    assert f" {event} " in log.split("--- pmset")[1]
    assert "run void" in log


def test_events_outside_the_window_and_wake_requests_do_not_void_the_run(
    fake_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    done = run_script(fake_bin, tmp_path, "r1")

    assert done.returncode == 0, done.stderr
    assert "run void" not in runner_log(tmp_path)


# ---------------------------------------------------------------- usage and log


def test_the_runner_log_sits_next_to_the_run_and_is_never_overwritten(
    fake_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    run_script(fake_bin, tmp_path, "r1")
    first = runner_log(tmp_path)

    again = run_script(fake_bin, tmp_path, "r1")

    assert again.returncode == 2
    assert runner_log(tmp_path) == first


def test_a_run_name_is_required(fake_bin: pathlib.Path, tmp_path: pathlib.Path) -> None:
    done = run_script(fake_bin, tmp_path)

    assert done.returncode == 2
    assert not (tmp_path / "calls.log").exists()


# ---------------------------------------------------------------- Makefile target


def recipe(target: str) -> str:
    match = re.search(rf"^{target}:.*\n((?:\t.*\n)+)", MAKEFILE.read_text(), re.MULTILINE)
    assert match, f"target {target} missing"
    return match.group(1)


def test_load_measured_needs_a_run_name() -> None:
    assert 'test -n "$(LOAD_RUN)"' in recipe("load-measured")


def test_load_measured_runs_the_script_with_this_make_and_the_load_variables() -> None:
    body = recipe("load-measured")

    assert 'MAKE="$(MAKE)" $(LOADTEST)/measured_run.sh "$(LOAD_RUN)" $(LOAD_ENV)' in body


def test_the_script_is_executable() -> None:
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
