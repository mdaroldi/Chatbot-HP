"""A run is configured from the environment `make load` passes on; bad values stop the
run before any user starts."""

import pytest

from hpdocs_load.config import RunConfig, config_from_env
from hpdocs_load.ramp import StepPlan


def test_defaults_need_only_a_run_name() -> None:
    assert config_from_env({"LOAD_RUN": "smoke-1"}) == RunConfig(
        name="smoke-1",
        seed=1,
        sample_seconds=5.0,
        plan=StepPlan(users=(1,), step_seconds=300.0, warmup_seconds=60.0, spawn_rate=1.0),
    )


def test_every_value_can_be_set() -> None:
    config = config_from_env(
        {
            "LOAD_RUN": "ramp-1",
            "LOAD_SEED": "42",
            "LOAD_SAMPLE_SECONDS": "2.5",
            "LOAD_STEPS": "4,8,16",
            "LOAD_STEP_SECONDS": "240",
            "LOAD_WARMUP_SECONDS": "45",
            "LOAD_SPAWN_RATE": "4",
        }
    )

    assert config == RunConfig(
        name="ramp-1",
        seed=42,
        sample_seconds=2.5,
        plan=StepPlan(users=(4, 8, 16), step_seconds=240.0, warmup_seconds=45.0, spawn_rate=4.0),
    )


def test_a_run_name_is_required() -> None:
    with pytest.raises(ValueError, match="LOAD_RUN"):
        config_from_env({})


@pytest.mark.parametrize("name", ["", "Has-Caps", "-leading", "a/b", "../up", "a b", "x" * 65])
def test_run_names_are_safe_directory_names(name: str) -> None:
    with pytest.raises(ValueError, match="LOAD_RUN"):
        config_from_env({"LOAD_RUN": name})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LOAD_SEED", "x"),
        ("LOAD_SAMPLE_SECONDS", "0"),
        ("LOAD_SAMPLE_SECONDS", "nan"),
        ("LOAD_STEP_SECONDS", "inf"),
        ("LOAD_WARMUP_SECONDS", "abc"),
    ],
)
def test_bad_numbers_are_rejected_by_name(key: str, value: str) -> None:
    with pytest.raises(ValueError, match=key):
        config_from_env({"LOAD_RUN": "r", key: value})


def test_empty_values_mean_the_default() -> None:
    # make passes every variable on, set or not.
    env = dict.fromkeys(
        (
            "LOAD_SEED",
            "LOAD_SAMPLE_SECONDS",
            "LOAD_STEPS",
            "LOAD_STEP_SECONDS",
            "LOAD_WARMUP_SECONDS",
            "LOAD_SPAWN_RATE",
        ),
        "",
    )

    assert config_from_env({"LOAD_RUN": "r", **env}) == config_from_env({"LOAD_RUN": "r"})


def test_the_burst_client_is_off_unless_enabled() -> None:
    assert config_from_env({"LOAD_RUN": "r"}).burst is False
    assert config_from_env({"LOAD_RUN": "r", "LOAD_BURST": ""}).burst is False
    assert config_from_env({"LOAD_RUN": "r", "LOAD_BURST": "0"}).burst is False


def test_the_burst_client_is_enabled_by_one() -> None:
    assert config_from_env({"LOAD_RUN": "r", "LOAD_BURST": "1"}).burst is True


@pytest.mark.parametrize("value", ["yes", "true", "on", "2", " 1", "1 "])
def test_only_zero_or_one_enable_the_burst_client(value: str) -> None:
    # A typo must stop the run, not silently run a capacity test with a burst client.
    with pytest.raises(ValueError, match="LOAD_BURST"):
        config_from_env({"LOAD_RUN": "r", "LOAD_BURST": value})
