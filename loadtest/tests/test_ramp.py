"""Step ramp maths: which user count runs when, and which part of a step is measured
(after its warm-up, to the step's end)."""

import pytest

from hpdocs_load.ramp import StepPlan, Window, measured_windows, parse_steps, tick

PLAN = StepPlan(users=(1, 5, 10), step_seconds=300.0, warmup_seconds=60.0, spawn_rate=2.0)


# ---------------------------------------------------------------- parse_steps


def test_steps_parse_from_a_comma_list() -> None:
    assert parse_steps("1, 5,10") == (1, 5, 10)


def test_a_single_step_is_a_valid_grid() -> None:
    assert parse_steps("1") == (1,)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("1,,5", id="blank-item"),
        pytest.param("1,x", id="not-a-number"),
        pytest.param("0,5", id="zero"),
        pytest.param("-1,5", id="negative"),
        pytest.param("5,5", id="repeated"),
        pytest.param("10,5", id="decreasing"),
        pytest.param("1.5", id="fraction"),
    ],
)
def test_bad_step_lists_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="steps"):
        parse_steps(text)


# ---------------------------------------------------------------- StepPlan


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"users": ()}, "steps", id="no-steps"),
        pytest.param({"step_seconds": 0.0}, "step", id="zero-step"),
        pytest.param({"warmup_seconds": -1.0}, "warm-up", id="negative-warm-up"),
        pytest.param({"warmup_seconds": 300.0}, "warm-up", id="warm-up-fills-the-step"),
        pytest.param({"spawn_rate": 0.0}, "spawn", id="zero-spawn-rate"),
    ],
)
def test_impossible_plans_are_rejected(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {
        "users": (1,),
        "step_seconds": 300.0,
        "warmup_seconds": 60.0,
        "spawn_rate": 1.0,
    }

    with pytest.raises(ValueError, match=match):
        StepPlan(**{**base, **kwargs})  # type: ignore[arg-type]


def test_the_plan_lasts_its_steps() -> None:
    assert PLAN.total_seconds == 900.0


# ---------------------------------------------------------------- tick


@pytest.mark.parametrize(
    ("elapsed", "users"),
    [(0.0, 1), (299.9, 1), (300.0, 5), (599.0, 5), (600.0, 10), (899.9, 10)],
)
def test_tick_runs_the_step_that_holds_the_elapsed_time(elapsed: float, users: int) -> None:
    assert tick(PLAN, elapsed) == (users, 2.0)


def test_tick_stops_the_run_after_the_last_step() -> None:
    assert tick(PLAN, 900.0) is None
    assert tick(PLAN, 5000.0) is None


# ---------------------------------------------------------------- measured windows


def test_each_step_is_measured_from_its_warm_up_end_to_its_end() -> None:
    assert measured_windows(PLAN, run_start=1000.0) == [
        Window(users=1, start=1060.0, end=1300.0),
        Window(users=5, start=1360.0, end=1600.0),
        Window(users=10, start=1660.0, end=1900.0),
    ]


def test_a_window_knows_its_length() -> None:
    assert Window(users=1, start=10.0, end=250.0).seconds == 240.0
