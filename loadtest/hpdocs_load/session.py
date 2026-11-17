"""One simulated browser: runs plans over a transport and records every request.

The transport is Locust's HTTP client in the locustfile and a fake in the tests. A
record holds times, the status and an error code, never text (rule 9). Times are taken
from a monotonic clock; the wall clock only stamps when a request started, so records
line up with the resource samples.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from hpdocs_load.budget import admit_all
from hpdocs_load.metrics import MESSAGE, RequestRecord
from hpdocs_load.sse import SseProtocolError, summarize_stream, timed_events
from hpdocs_load.workload import CREATE, DELETE, LIST, OPEN, PREVIOUS, READY, THINK, Action

CONNECTION_ERROR, PROTOCOL_ERROR = "connection_error", "protocol_error"
_CONVERSATIONS = "/v1/conversations"
_OK = 200


@dataclass(frozen=True)
class Reply:
    status: int
    text: str


@dataclass(frozen=True)
class StreamReply:
    """Lines with the monotonic time each one arrived; status 0 means no connection."""

    status: int
    lines: Iterable[tuple[float, str]]


class Transport(Protocol):
    def call(self, method: str, path: str, *, name: str) -> Reply: ...

    def stream(self, path: str, *, content: str) -> StreamReply: ...


class Clock(Protocol):
    def wall(self) -> float: ...

    def mono(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


def error_code_of(status: int, text: str) -> str | None:
    """The backend's unified error code when it sent one, else a code from the status."""
    if 200 <= status < 300:
        return None
    if status == 0:
        return CONNECTION_ERROR
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else f"http_{status}"


class Session:
    def __init__(
        self,
        transport: Transport,
        clock: Clock,
        record: Callable[[RequestRecord], None],
        *,
        admit: Callable[..., bool] = admit_all,
    ) -> None:
        self._transport = transport
        self._admit = admit
        self._clock = clock
        self._record = record
        self._kept: list[str] = []

    def run(self, actions: Iterable[Action]) -> None:
        """A failed create ends the plan: there is no conversation to ask in. So does a
        question the spend guard refuses: the run is stopping."""
        for action in actions:
            if action.kind == MESSAGE and not self._admit(follow_up=action.follow_up):
                return
            if action.kind == THINK:
                self._clock.sleep(action.seconds)
            elif action.kind == CREATE and not self._create():
                return
            elif action.kind == MESSAGE:
                self._message(str(action.text), follow_up=action.follow_up)
            elif action.kind == DELETE:
                self._call("DELETE", f"{_CONVERSATIONS}/{self._kept.pop()}", DELETE)
            elif action.kind == OPEN:
                self._call("GET", f"{_CONVERSATIONS}/{self._target(action.target)}", OPEN)
            elif action.kind in {LIST, READY}:
                self._call("GET", "/ready" if action.kind == READY else _CONVERSATIONS, action.kind)

    def _target(self, target: str) -> str:
        """Selecting picks the conversation before the current one, if there is one."""
        if target == PREVIOUS and len(self._kept) > 1:
            return self._kept[-2]
        return self._kept[-1]

    def _create(self) -> bool:
        reply = self._call("POST", _CONVERSATIONS, CREATE)
        if error_code_of(reply.status, reply.text) is not None:
            return False
        self._kept.append(str(json.loads(reply.text)["id"]))
        return True

    def _call(self, method: str, path: str, name: str) -> Reply:
        wall, start = self._clock.wall(), self._clock.mono()
        reply = self._transport.call(method, path, name=name)
        self._record(
            RequestRecord(
                name=name,
                start=wall,
                duration_ms=(self._clock.mono() - start) * 1000,
                status=reply.status,
                ttft_ms=None,
                reached_done=False,
                error_code=error_code_of(reply.status, reply.text),
            )
        )
        return reply

    def _message(self, content: str, *, follow_up: bool) -> None:
        wall, start = self._clock.wall(), self._clock.mono()
        path = f"{_CONVERSATIONS}/{self._kept[-1]}/messages"
        reply = self._transport.stream(path, content=content)
        ttft_ms: float | None = None
        reached_done = False
        try:
            if reply.status != _OK:
                lines = [] if reply.status == 0 else [line for _, line in reply.lines]
                code = error_code_of(reply.status, "\n".join(lines))
            else:
                result = summarize_stream(start, timed_events(reply.lines))
                ttft_ms, reached_done, code = result.ttft_ms, result.reached_done, result.error_code
        except SseProtocolError:
            code = PROTOCOL_ERROR
        except OSError:
            code = CONNECTION_ERROR
        self._record(
            RequestRecord(
                name=MESSAGE,
                start=wall,
                duration_ms=(self._clock.mono() - start) * 1000,
                status=reply.status,
                ttft_ms=ttft_ms,
                reached_done=reached_done,
                error_code=code,
                follow_up=follow_up,
            )
        )
