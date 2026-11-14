"""Server-sent events from the answer stream, parsed incrementally.

Network chunks split lines anywhere, including between the CR and LF of one line
end, so the parser buffers text and emits an event only at its blank line. It
follows the WHATWG event-stream rules, with one deliberate difference: a last frame
missing its blank line is delivered at close() rather than dropped, so a truncated
`done` reaches the client and fails there as an unreadable response instead of
vanishing. Data stays text; the API client decodes the JSON.

A frame may not grow past MAX_PENDING_FRAME_BYTES before its blank line: a stream
that never terminates a frame would otherwise grow the buffer without bound. The
limit is per pending frame, so a long answer of many small frames never meets it.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

_LINE_END = re.compile(r"\r\n|\r|\n")
_DEFAULT_EVENT = "message"
# Far above any real frame: a done event holds one answer of a few thousand tokens.
MAX_PENDING_FRAME_BYTES = 1024 * 1024


class SseFrameTooLargeError(Exception):
    """A frame passed MAX_PENDING_FRAME_BYTES before its blank line."""


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: str


@dataclass
class SseParser:
    _buffer: str = ""
    _event: str = _DEFAULT_EVENT
    _data: list[str] = field(default_factory=list)
    _frame_bytes: int = 0

    def feed(self, chunk: str) -> list[SseEvent]:
        """Events completed by this chunk, in order."""
        self._buffer += chunk
        events: list[SseEvent] = []
        start = 0
        for match in _LINE_END.finditer(self._buffer):
            if match.group() == "\r" and match.end() == len(self._buffer):
                break  # an LF may follow in the next chunk
            line = self._buffer[start : match.start()]
            if line:
                self._frame_bytes += len(line.encode()) + len(match.group())
                self._check_size(0)
            events.extend(self._line(line))
            start = match.end()
        self._buffer = self._buffer[start:]
        self._check_size(len(self._buffer.encode()))
        return events

    def close(self) -> list[SseEvent]:
        """The end of the stream: a pending last line and frame are delivered."""
        events = self.feed("\n") if self._buffer.rstrip("\r") else []
        self._buffer = ""
        events.extend(self._line(""))
        return events

    def _check_size(self, unterminated: int) -> None:
        if self._frame_bytes + unterminated > MAX_PENDING_FRAME_BYTES:
            raise SseFrameTooLargeError

    def _line(self, line: str) -> list[SseEvent]:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return []
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            self._event = value
        elif name == "data":
            self._data.append(value)
        return []

    def _dispatch(self) -> list[SseEvent]:
        events = [SseEvent(self._event, "\n".join(self._data))] if self._data else []
        self._event = _DEFAULT_EVENT
        self._data = []
        self._frame_bytes = 0
        return events


def iter_events(chunks: Iterable[str]) -> Iterator[SseEvent]:
    """Events as their chunks arrive, then whatever close() delivers."""
    parser = SseParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.close()
