"""Run settings from the environment `make load` passes on (empty means default)."""

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from hpdocs_load.ramp import StepPlan, parse_steps

_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_DEFAULTS = {
    "LOAD_SEED": "1",
    "LOAD_SAMPLE_SECONDS": "5",
    "LOAD_STEPS": "1",
    "LOAD_STEP_SECONDS": "300",
    "LOAD_WARMUP_SECONDS": "60",
    "LOAD_SPAWN_RATE": "1",
    "LOAD_BURST": "0",
    "LOAD_BUDGET": "0",
}
_SWITCH = {"0": False, "1": True}


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    sample_seconds: float
    plan: StepPlan
    # The rate-limits run's burst client (hpdocs_load.burst); never on in a capacity run.
    burst: bool = False
    # Profile B's spend guard (hpdocs_load.budget); `make load` turns it on for anthropic.
    budget: bool = False


def _positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError
    return value


def _switch(text: str) -> bool:
    """Only 0 or 1: a typo must stop the run, not decide whether the burst client runs."""
    if text not in _SWITCH:
        raise ValueError
    return _SWITCH[text]


def config_from_env(env: Mapping[str, str]) -> RunConfig:
    """The run name becomes a directory under results/, so only safe names pass."""
    name = env.get("LOAD_RUN", "")
    if not _RUN_NAME.fullmatch(name):
        raise ValueError("LOAD_RUN must be 1-64 of a-z, 0-9 and '-', not starting with '-'")

    def value[T](key: str, parse: Callable[[str], T]) -> T:
        try:
            return parse(env.get(key) or _DEFAULTS[key])
        except ValueError as error:
            raise ValueError(f"{key} has an invalid value {error}".rstrip()) from None

    return RunConfig(
        name=name,
        seed=value("LOAD_SEED", int),
        sample_seconds=value("LOAD_SAMPLE_SECONDS", _positive),
        plan=StepPlan(
            users=value("LOAD_STEPS", parse_steps),
            step_seconds=value("LOAD_STEP_SECONDS", _positive),
            warmup_seconds=value("LOAD_WARMUP_SECONDS", float),  # StepPlan bounds it
            spawn_rate=value("LOAD_SPAWN_RATE", _positive),
        ),
        burst=value("LOAD_BURST", _switch),
        budget=value("LOAD_BUDGET", _switch),
    )
