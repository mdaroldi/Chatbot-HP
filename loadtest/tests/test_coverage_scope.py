"""The coverage gate leaves out exactly one production file: the Locust glue, which has
no branching of its own (ADR 0013). Anything else must stay measured."""

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_only_the_locustfile_is_omitted_besides_the_tests() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["coverage"]

    assert config["run"]["omit"] == ["tests/*", "locustfile.py"]
    assert config["run"]["branch"] is True
    assert config["report"]["fail_under"] == 90


def test_the_omitted_glue_exists_at_the_project_root() -> None:
    assert (ROOT / "locustfile.py").is_file()
