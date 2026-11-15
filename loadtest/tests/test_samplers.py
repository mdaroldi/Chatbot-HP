"""Resource samples taken during a run: CPU and memory per load-stack container, the
host's memory and swap (8 GB machine), and the CPU of the Locust process itself."""

import json
import sys
from types import SimpleNamespace

import pytest

from hpdocs_load.samplers import (
    DOCKER_STATS_COMMAND,
    LOAD_PREFIX,
    ContainerSample,
    GeneratorSample,
    HostProbes,
    HostSample,
    parse_docker_stats,
    parse_size,
    run_command,
    take_samples,
)


def stats_line(name: str, cpu: str = "12.50%", mem: str = "512MiB / 2GiB") -> str:
    return json.dumps({"Name": name, "CPUPerc": cpu, "MemUsage": mem, "ID": "abc"})


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("0B", 0),
        ("512B", 512),
        ("1.5KiB", 1536),
        ("2MiB", 2 * 1024**2),
        ("1.25GiB", int(1.25 * 1024**3)),
        ("7.6GiB", int(7.6 * 1024**3)),
        ("3kB", 3000),
        ("4MB", 4_000_000),
        ("1GB", 1_000_000_000),
        (" 10MiB ", 10 * 1024**2),
    ],
)
def test_sizes_parse_in_docker_units(text: str, size: int) -> None:
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "MiB", "12XB", "1.2.3MiB", "--"])
def test_unknown_sizes_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="size"):
        parse_size(text)


def test_docker_stats_keep_load_stack_containers_only() -> None:
    output = "\n".join(
        [
            stats_line("hpdocs-load-backend-1", "312.40%", "1.2GiB / 2GiB"),
            stats_line("hpdocs-backend-1"),
            "",
            stats_line("hpdocs-load-qdrant-1", "8.00%", "300MiB / 7.6GiB"),
        ]
    )

    assert parse_docker_stats(output, at=50.0) == [
        ContainerSample(
            at=50.0,
            name="hpdocs-load-backend-1",
            cpu_percent=312.4,
            mem_bytes=int(1.2 * 1024**3),
            mem_limit_bytes=2 * 1024**3,
        ),
        ContainerSample(
            at=50.0,
            name="hpdocs-load-qdrant-1",
            cpu_percent=8.0,
            mem_bytes=300 * 1024**2,
            mem_limit_bytes=int(7.6 * 1024**3),
        ),
    ]
    assert LOAD_PREFIX == "hpdocs-load-"


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("not json", id="not-json"),
        pytest.param(json.dumps({"Name": "hpdocs-load-x"}), id="missing-fields"),
        pytest.param(stats_line("hpdocs-load-x", cpu="12.5"), id="cpu-without-percent"),
        pytest.param(stats_line("hpdocs-load-x", mem="512MiB"), id="mem-without-limit"),
    ],
)
def test_malformed_docker_stats_raise(line: str) -> None:
    with pytest.raises(ValueError, match="docker stats"):
        parse_docker_stats(line, at=1.0)


def test_the_docker_command_takes_one_sample_as_json() -> None:
    assert DOCKER_STATS_COMMAND == ("docker", "stats", "--no-stream", "--format", "{{json .}}")


def test_run_command_returns_standard_output() -> None:
    assert run_command((sys.executable, "-c", "print('ok')")) == "ok\n"


def test_run_command_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="exit status 3"):
        run_command((sys.executable, "-c", "raise SystemExit(3)"))


def test_one_round_samples_host_generator_and_containers() -> None:
    vm = SimpleNamespace(percent=71.5, available=2 * 1024**3)
    swap = SimpleNamespace(used=512 * 1024**2, percent=25.0, sin=7 * 16384, sout=3 * 16384)
    seen: list[tuple[str, ...]] = []

    def docker(argv: tuple[str, ...]) -> str:
        seen.append(argv)
        return stats_line("hpdocs-load-postgres-1", "1.00%", "64MiB / 7.6GiB")

    samples = take_samples(
        at=9.0,
        docker=docker,
        host=HostProbes(memory=lambda: vm, swap=lambda: swap, page_bytes=16384),
        generator_cpu=lambda: 87.0,
    )

    assert seen == [DOCKER_STATS_COMMAND]
    assert samples[:2] == [
        HostSample(
            at=9.0,
            mem_percent=71.5,
            mem_available_bytes=2 * 1024**3,
            swap_used_bytes=512 * 1024**2,
            swap_percent=25.0,
            pageins=7,
            pageouts=3,
        ),
        GeneratorSample(at=9.0, cpu_percent=87.0),
    ]
    assert [s.kind for s in samples] == ["host", "generator", "container"]
