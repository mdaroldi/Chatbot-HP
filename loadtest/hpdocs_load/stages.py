"""Per-stage timings under load, from the backend's own JSON logs (no backend change).

Only the timing fields of `retrieval done` and `chat answer` are kept, with the request
ID that joins them: no content, chunk IDs or conversation IDs reach the run directory.
A timing that is present but not a number raises, so a log contract change is noticed.
"""

import json
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime

BACKEND_CONTAINER = "hpdocs-load-backend-1"
_STAGES = {
    "retrieval done": (
        "retrieval",
        ("dense_embed_ms", "sparse_embed_ms", "search_ms", "rerank_ms", "total_ms"),
    ),
    "chat answer": ("chat", ("ttft_ms", "generation_ms", "total_ms")),
}


@dataclass(frozen=True)
class StageRow:
    at: float
    request_id: str
    kind: str
    timings: dict[str, float]


def _timings(row: dict[str, object], fields: tuple[str, ...]) -> dict[str, float]:
    kept: dict[str, float] = {}
    for name in fields:
        if name not in row:
            raise ValueError(f"log line lacks {name}")
        value = row[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} is not a number")  # noqa: TRY004 - one error for any contract change
        kept[name] = float(value)
    return kept


def parse_stage_logs(text: str) -> list[StageRow]:
    rows: list[StageRow] = []
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        stage = _STAGES.get(row.get("message"))
        if stage is None:
            continue
        kind, fields = stage
        rows.append(
            StageRow(
                at=datetime.fromisoformat(row["timestamp"]).timestamp(),
                request_id=str(row["request_id"]),
                kind=kind,
                timings=_timings(row, fields),
            )
        )
    return rows


def logs_command(since: float) -> tuple[str, ...]:
    return ("docker", "logs", "--since", str(int(since)), BACKEND_CONTAINER)


def stage_line(row: StageRow) -> str:
    return json.dumps(asdict(row), sort_keys=True)


def read_stages(path: pathlib.Path) -> list[StageRow]:
    with path.open() as lines:
        return [StageRow(**json.loads(line)) for line in lines if line.strip()]
