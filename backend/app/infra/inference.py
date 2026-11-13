"""The inference executor and the CPU allowance it should fit (ADR 0008).

Embedding and reranking are CPU-bound ONNX calls. They run on a dedicated thread
pool, never on the event loop, with the ONNX intra-op thread count set explicitly, so
that workers times threads can be kept within the CPUs the container may use.
fastembed's `parallel` option (process pools) is never used in the request path.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


class CpuSource(StrEnum):
    CGROUP = "cgroup"
    CPU_COUNT = "cpu_count"


@dataclass(frozen=True)
class CpuAllowance:
    """CPUs this process may use, and where the number came from."""

    cpus: float
    source: CpuSource


def _cgroup_quota(cpu_max: Path) -> float | None:
    """cgroup v2 `cpu.max` reads "<quota> <period>" or "max <period>" (no limit)."""
    try:
        quota, period = cpu_max.read_text(encoding="ascii").split()
        cpus = int(quota) / int(period)
    except (OSError, ValueError, ZeroDivisionError):
        return None
    return cpus if cpus > 0 else None


def cpu_allowance(
    *,
    cpu_max: Path = CGROUP_CPU_MAX,
    cpu_count: Callable[[], int | None] = os.cpu_count,
) -> CpuAllowance:
    """The container's CPU quota when one is readable, else the machine's CPU count,
    which can be more than a container may use. The eval report header reads it too.
    """
    quota = _cgroup_quota(cpu_max)
    if quota is not None:
        return CpuAllowance(quota, CpuSource.CGROUP)
    return CpuAllowance(float(cpu_count() or 1), CpuSource.CPU_COUNT)


def warn_if_oversubscribed(workers: int, threads: int, allowance: CpuAllowance) -> bool:
    """Log a warning when the inference threads exceed the allowance; True if so."""
    total = workers * threads
    if total <= allowance.cpus:
        return False
    if allowance.source is CpuSource.CGROUP:
        basis = "the cgroup CPU quota"
    else:
        basis = "the machine's CPU count (no cgroup CPU quota was readable)"
    logger.warning(
        "inference workers times ONNX threads exceed %s; calls will contend for CPU",
        basis,
        extra={
            "inference_workers": workers,
            "onnx_threads": threads,
            "inference_threads": total,
            "cpus": allowance.cpus,
            "cpu_source": allowance.source.value,
        },
    )
    return True


def create_inference_executor(workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inference")


async def run_inference[T](executor: Executor | None, call: Callable[[], T]) -> T:
    """Run a blocking model call on the executor (the default one when None)."""
    return await asyncio.get_running_loop().run_in_executor(executor, call)
