"""Reading the answer stream as a client does, to time it the way a user waits for it.

The contract is ADR 0010's: `event: <name>`, one `data: <JSON object>` line, a blank
line; `:` lines are keepalives. A stream that breaks the contract raises, so a backend
change can never be counted as capacity by mistake.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

TOKEN, SOURCES, DONE, ERROR = "token", "sources", "done", "error"
NO_DONE = "no_done"
_KNOWN = {TOKEN, SOURCES, DONE, ERROR}


class SseProtocolError(ValueError):
    """The stream does not follow the backend's SSE contract."""


@dataclass(frozen=True)
class SseEvent:
    name: str
    data: dict[str, object]


@dataclass(frozen=True)
class StreamResult:
    """What the client saw: first token and end times from the request start, in ms."""

    ttft_ms: float | None
    total_ms: float | None
    tokens: int
    outcome: str | None
    error_code: str | None

    @property
    def reached_done(self) -> bool:
        """The unit of capacity: the stream ended with `done` (completed or refused)."""
        return self.outcome is not None


def _data(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise SseProtocolError("event data is not JSON") from None
    if not isinstance(value, dict):
        raise SseProtocolError("event data is not a JSON object")
    return value


def parse_events(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Yield each complete event; the text itself is never logged by callers."""
    name: str | None = None
    data: dict[str, object] | None = None
    for line in lines:
        if line.startswith(":"):
            continue
        if line == "":
            if name is None and data is None:
                continue
            if name is None or data is None:
                raise SseProtocolError("an event needs one event line and one data line")
            yield SseEvent(name, data)
            name, data = None, None
        elif line.startswith("event: ") and name is None:
            name = line.removeprefix("event: ")
        elif line.startswith("data: ") and data is None and name is not None:
            data = _data(line.removeprefix("data: "))
        else:
            raise SseProtocolError("unexpected line in the event stream")
    if name is not None or data is not None:
        raise SseProtocolError("the stream ended inside an event")


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise SseProtocolError(f"`{key}` is missing or not a string")
    return value


def summarize_stream(start: float, events: Iterable[tuple[float, SseEvent]]) -> StreamResult:
    """Fold timed events (monotonic seconds) into what the load report needs."""
    first: float | None = None
    last: float | None = None
    tokens = 0
    outcome: str | None = None
    error_code: str | None = None
    for at, event in events:
        if outcome is not None or error_code is not None:
            raise SseProtocolError("an event arrived after the stream ended")
        if event.name not in _KNOWN:
            raise SseProtocolError("unknown event")
        last = at
        if event.name == TOKEN:
            tokens += 1
            first = at if first is None else first
        elif event.name == DONE:
            outcome = _string(event.data, "outcome")
        elif event.name == ERROR:
            error_code = _string(event.data, "code")
    if outcome is None and error_code is None:
        error_code = NO_DONE

    def _ms(at: float | None) -> float | None:
        return None if at is None else (at - start) * 1000

    return StreamResult(_ms(first), _ms(last), tokens, outcome, error_code)


def timed_events(lines: Iterable[tuple[float, str]]) -> Iterator[tuple[float, SseEvent]]:
    """Each event with the arrival time of the blank line that completes it."""
    clock = [0.0]

    def untimed() -> Iterator[str]:
        for at, line in lines:
            clock[0] = at
            yield line

    for event in parse_events(untimed()):
        yield clock[0], event
