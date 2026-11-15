"""A run directory holds the run's metadata, one JSON line per request and one per
resource sample; it is never reused, so runs never mix."""

import json
import pathlib

import pytest

from hpdocs_load.metrics import RequestRecord, read_records
from hpdocs_load.run_files import RunFiles
from hpdocs_load.samplers import GeneratorSample, HostSample

RECORD = RequestRecord("list", 1.0, 20.0, 200, None, False, None)


def test_a_new_run_gets_its_own_directory(tmp_path: pathlib.Path) -> None:
    with RunFiles.create(tmp_path / "results", "smoke-1") as files:
        assert files.directory == tmp_path / "results" / "smoke-1"
        assert files.directory.is_dir()


def test_an_existing_run_is_never_overwritten(tmp_path: pathlib.Path) -> None:
    RunFiles.create(tmp_path, "smoke-1").close()

    with pytest.raises(FileExistsError):
        RunFiles.create(tmp_path, "smoke-1")


def test_metadata_is_rewritten_whole(tmp_path: pathlib.Path) -> None:
    with RunFiles.create(tmp_path, "r") as files:
        files.write_meta({"started_at": 1.0})
        files.write_meta({"started_at": 1.0, "ended_at": 2.0})

    meta = json.loads((files.directory / "run.json").read_text())
    assert meta == {"started_at": 1.0, "ended_at": 2.0}


def test_requests_are_appended_and_readable_while_running(tmp_path: pathlib.Path) -> None:
    with RunFiles.create(tmp_path, "r") as files:
        files.add_request(RECORD)
        files.add_request(RECORD)

        assert read_records(files.directory / "requests.jsonl") == [RECORD, RECORD]


def test_samples_carry_their_kind(tmp_path: pathlib.Path) -> None:
    with RunFiles.create(tmp_path, "r") as files:
        files.add_samples([HostSample(1.0, 50.0, 100, 0, 0.0), GeneratorSample(1.0, 12.0)])
        lines = (files.directory / "samples.jsonl").read_text().splitlines()

    assert [json.loads(line) for line in lines] == [
        {
            "kind": "host",
            "at": 1.0,
            "mem_percent": 50.0,
            "mem_available_bytes": 100,
            "swap_used_bytes": 0,
            "swap_percent": 0.0,
        },
        {"kind": "generator", "at": 1.0, "cpu_percent": 12.0},
    ]


def test_closing_twice_is_harmless(tmp_path: pathlib.Path) -> None:
    files = RunFiles.create(tmp_path, "r")

    files.close()
    files.close()

    with pytest.raises(ValueError, match="closed"):
        files.add_request(RECORD)
