"""Append-only JSON-lines files of the answer harness (results journal, budget ledger).

One object per line, appended and fsynced before the caller goes on, so a crash loses
at most the line in flight. Reading is strict: a partial last line or a line that is
not a JSON object raises the caller's error naming the line number, never its content.
"""

import json
import os
from pathlib import Path
from typing import cast

# A decoded JSON object; values are checked one by one before use.
Json = dict[str, object]


def append_object(path: Path, obj: Json) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_objects(path: Path, error: type[Exception]) -> list[tuple[int, Json]]:
    """(line number, object) pairs in file order; a missing file has none."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = text.split("\n")
    if lines[-1]:
        # The last write did not finish its line: no guess at what it held.
        raise error(f"line {len(lines)}: partial line")
    objects: list[tuple[int, Json]] = []
    for number, line in enumerate(lines[:-1], start=1):
        try:
            decoded: object = json.loads(line)
        except json.JSONDecodeError:
            decoded = None
        if not isinstance(decoded, dict):
            raise error(f"line {number}: not a JSON object")
        objects.append((number, cast(Json, decoded)))
    return objects
