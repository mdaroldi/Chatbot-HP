"""The inference executor and the CPU allowance check (ADR 0008)."""

import logging
import threading
from pathlib import Path

import pytest

from app.infra.inference import (
    CpuAllowance,
    CpuSource,
    cpu_allowance,
    create_inference_executor,
    run_inference,
    warn_if_oversubscribed,
)


def cpu_max(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cpu.max"
    path.write_text(text)
    return path


# ------------------------------------------------------------------ CPU allowance


def test_a_cgroup_quota_sets_the_allowance(tmp_path: Path) -> None:
    allowance = cpu_allowance(cpu_max=cpu_max(tmp_path, "250000 100000\n"), cpu_count=lambda: 8)

    assert allowance == CpuAllowance(cpus=2.5, source=CpuSource.CGROUP)


@pytest.mark.parametrize("text", ["max 100000\n", "garbage", "100000", "0 100000", "1 0"])
def test_no_usable_quota_falls_back_to_the_cpu_count(tmp_path: Path, text: str) -> None:
    allowance = cpu_allowance(cpu_max=cpu_max(tmp_path, text), cpu_count=lambda: 8)

    assert allowance == CpuAllowance(cpus=8.0, source=CpuSource.CPU_COUNT)


def test_a_missing_cgroup_file_falls_back_to_the_cpu_count(tmp_path: Path) -> None:
    allowance = cpu_allowance(cpu_max=tmp_path / "absent", cpu_count=lambda: 4)

    assert allowance == CpuAllowance(cpus=4.0, source=CpuSource.CPU_COUNT)


def test_an_unknown_cpu_count_counts_as_one(tmp_path: Path) -> None:
    assert cpu_allowance(cpu_max=tmp_path / "absent", cpu_count=lambda: None).cpus == 1.0


# ------------------------------------------------------------------ warning


def test_more_threads_than_the_quota_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.infra.inference"):
        warned = warn_if_oversubscribed(2, 2, CpuAllowance(2.0, CpuSource.CGROUP))

    assert warned
    (record,) = caplog.records
    assert record.__dict__["inference_threads"] == 4
    assert record.__dict__["cpus"] == 2.0
    assert record.__dict__["cpu_source"] == "cgroup"
    assert "cgroup CPU quota" in record.getMessage()


def test_the_warning_says_when_the_cpu_count_was_the_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.infra.inference"):
        warn_if_oversubscribed(4, 4, CpuAllowance(8.0, CpuSource.CPU_COUNT))

    message = caplog.records[0].getMessage()
    assert "no cgroup CPU quota was readable" in message
    assert "CPU count" in message


def test_threads_within_the_allowance_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.infra.inference"):
        warned = warn_if_oversubscribed(2, 2, CpuAllowance(4.0, CpuSource.CGROUP))

    assert not warned
    assert caplog.records == []


# ------------------------------------------------------------------ executor


async def test_inference_runs_on_the_dedicated_named_threads() -> None:
    executor = create_inference_executor(2)
    try:
        name = await run_inference(executor, lambda: threading.current_thread().name)
    finally:
        executor.shutdown()

    assert name.startswith("inference")
    assert name != threading.current_thread().name


def test_the_executor_has_the_configured_worker_count() -> None:
    executor = create_inference_executor(3)
    try:
        assert executor._max_workers == 3
    finally:
        executor.shutdown()


async def test_an_exception_in_inference_reaches_the_caller() -> None:
    executor = create_inference_executor(1)

    def boom() -> None:
        raise ValueError("model failed")

    try:
        with pytest.raises(ValueError, match="model failed"):
            await run_inference(executor, boom)
    finally:
        executor.shutdown()
