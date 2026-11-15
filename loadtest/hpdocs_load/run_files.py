"""One directory per run under the git-ignored results/ (rule 8): run.json, plus
requests.jsonl and samples.jsonl, flushed per line so a cut run keeps what it measured,
and stages.jsonl from the backend's logs at the end."""

import json
import pathlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from types import TracebackType
from typing import TextIO

from hpdocs_load.metrics import RequestRecord, write_record
from hpdocs_load.samplers import Sample
from hpdocs_load.stages import StageRow, stage_line


class RunFiles:
    def __init__(self, directory: pathlib.Path) -> None:
        self.directory = directory
        self._requests: TextIO = (directory / "requests.jsonl").open("x")
        self._samples: TextIO = (directory / "samples.jsonl").open("x")

    @classmethod
    def create(cls, root: pathlib.Path, name: str) -> "RunFiles":
        """Fails when the run exists: a run's numbers are never mixed with another's."""
        directory = root / name
        root.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        return cls(directory)

    def write_meta(self, meta: Mapping[str, object]) -> None:
        (self.directory / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    def add_request(self, record: RequestRecord) -> None:
        write_record(self._requests, record)
        self._requests.flush()

    def add_samples(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self._samples.write(json.dumps({"kind": sample.kind, **asdict(sample)}) + "\n")
        self._samples.flush()

    def __enter__(self) -> "RunFiles":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    def add_stages(self, rows: Iterable[StageRow]) -> None:
        """Written once at the end of a run, from the backend's logs."""
        with (self.directory / "stages.jsonl").open("x") as out:
            out.writelines(stage_line(row) + "\n" for row in rows)

    def close(self) -> None:
        self._requests.close()
        self._samples.close()
