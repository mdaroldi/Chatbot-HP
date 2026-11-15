"""Resource samples during a run (rules 4 and 10, and the 8 GB host's memory).

Per round: the host's memory and swap, the Locust process's CPU (a step with a
saturated generator does not count), and CPU and memory of each load-stack container
from one `docker stats` sample. The probes are injected, so the parsing is tested
without Docker or psutil. subprocess runs only the fixed `docker stats` argv, never
through a shell.
"""

import json
import re
import subprocess  # nosec B404
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

LOAD_PREFIX = "hpdocs-load-"
DOCKER_STATS_COMMAND = ("docker", "stats", "--no-stream", "--format", "{{json .}}")
_COMMAND_TIMEOUT_SECONDS = 30
_UNITS = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}
_SIZE = re.compile(r"(\d+(?:\.\d+)?)([A-Za-z]+)")


@dataclass(frozen=True)
class ContainerSample:
    at: float
    name: str
    cpu_percent: float
    mem_bytes: int
    mem_limit_bytes: int
    kind: str = field(default="container", init=False)


@dataclass(frozen=True)
class HostSample:
    at: float
    mem_percent: float
    mem_available_bytes: int
    swap_used_bytes: int
    swap_percent: float
    pageins: int  # cumulative since boot, as vm_stat counts them
    pageouts: int
    kind: str = field(default="host", init=False)


@dataclass(frozen=True)
class GeneratorSample:
    """Percent of one core: Locust runs its users on one core."""

    at: float
    cpu_percent: float
    kind: str = field(default="generator", init=False)


Sample = ContainerSample | HostSample | GeneratorSample


class Memory(Protocol):
    @property
    def percent(self) -> float: ...

    @property
    def available(self) -> int: ...


class Swap(Protocol):
    @property
    def used(self) -> int: ...

    @property
    def percent(self) -> float: ...

    @property
    def sin(self) -> int: ...

    @property
    def sout(self) -> int: ...


def parse_size(text: str) -> int:
    match = _SIZE.fullmatch(text.strip())
    if match is None or match.group(2) not in _UNITS:
        raise ValueError("not a docker size")
    return int(float(match.group(1)) * _UNITS[match.group(2)])


def _percent(text: str) -> float:
    if not text.endswith("%"):
        raise ValueError("not a percentage")
    return float(text.removesuffix("%"))


def _container(line: str, at: float) -> ContainerSample:
    try:
        row = json.loads(line)
        used, limit = row["MemUsage"].split(" / ")
        return ContainerSample(
            at=at,
            name=row["Name"],
            cpu_percent=_percent(row["CPUPerc"]),
            mem_bytes=parse_size(used),
            mem_limit_bytes=parse_size(limit),
        )
    except (ValueError, KeyError):
        raise ValueError("unexpected docker stats line") from None


def parse_docker_stats(output: str, *, at: float) -> list[ContainerSample]:
    """Only the load stack's containers: the dev stack may run next to it."""
    samples = [_container(line, at) for line in output.splitlines() if line.strip()]
    return [s for s in samples if s.name.startswith(LOAD_PREFIX)]


def run_command(argv: Sequence[str]) -> str:
    try:
        # S603/B603: argv is a fixed tuple from this module, run without a shell.
        done = subprocess.run(  # noqa: S603 # nosec B603
            list(argv), capture_output=True, text=True, check=True, timeout=_COMMAND_TIMEOUT_SECONDS
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{argv[0]} failed with exit status {error.returncode}") from None
    return done.stdout


@dataclass(frozen=True)
class HostProbes:
    """psutil reports vm_stat's page-ins and page-outs in bytes; samples store pages."""

    memory: Callable[[], Memory]
    swap: Callable[[], Swap]
    page_bytes: int


def take_samples(
    *,
    at: float,
    docker: Callable[[tuple[str, ...]], str],
    host: HostProbes,
    generator_cpu: Callable[[], float],
) -> list[Sample]:
    vm, sw, page_bytes = host.memory(), host.swap(), host.page_bytes
    return [
        HostSample(
            at=at,
            mem_percent=vm.percent,
            mem_available_bytes=vm.available,
            swap_used_bytes=sw.used,
            swap_percent=sw.percent,
            pageins=sw.sin // page_bytes,
            pageouts=sw.sout // page_bytes,
        ),
        GeneratorSample(at=at, cpu_percent=generator_cpu()),
        *parse_docker_stats(docker(DOCKER_STATS_COMMAND), at=at),
    ]
