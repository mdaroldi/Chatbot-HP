"""Step ramp: a fixed user count per step; each step is measured after its warm-up.

The grid is exploratory: steps are added until the SLO clearly breaks, so the list is
open-ended. Rule 2 (at least 3 measured minutes) is checked by the report, not here, so
short smoke runs stay possible.
"""

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class StepPlan:
    users: tuple[int, ...]
    step_seconds: float
    warmup_seconds: float
    spawn_rate: float

    def __post_init__(self) -> None:
        if not self.users:
            raise ValueError("a plan needs at least one of its steps")
        if self.step_seconds <= 0:
            raise ValueError("a step needs a positive length")
        if not 0 <= self.warmup_seconds < self.step_seconds:
            raise ValueError("the warm-up must be shorter than a step and not negative")
        if self.spawn_rate <= 0:
            raise ValueError("the spawn rate must be positive")

    @property
    def total_seconds(self) -> float:
        return self.step_seconds * len(self.users)


@dataclass(frozen=True)
class Window:
    users: int
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


def parse_steps(text: str) -> tuple[int, ...]:
    """Strictly increasing positive user counts, e.g. `1,5,10`."""
    try:
        steps = tuple(int(item) for item in text.split(","))
    except ValueError:
        raise ValueError("steps must be a comma list of whole numbers") from None
    if steps[0] <= 0 or any(b <= a for a, b in itertools.pairwise(steps)):
        raise ValueError("steps must be positive and strictly increasing")
    return steps


def tick(plan: StepPlan, elapsed: float) -> tuple[int, float] | None:
    """Locust's shape callback: the users to hold now, or None to end the run."""
    index = int(elapsed // plan.step_seconds)
    if index >= len(plan.users):
        return None
    return plan.users[index], plan.spawn_rate


def measured_windows(plan: StepPlan, *, run_start: float) -> list[Window]:
    return [
        Window(
            users=users,
            start=run_start + i * plan.step_seconds + plan.warmup_seconds,
            end=run_start + (i + 1) * plan.step_seconds,
        )
        for i, users in enumerate(plan.users)
    ]
