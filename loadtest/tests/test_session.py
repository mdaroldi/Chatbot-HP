"""One simulated browser runs its plan over a transport and leaves one record per
request: times, status and error code, never text."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from hpdocs_load.metrics import RequestRecord
from hpdocs_load.session import Reply, Session, StreamReply, error_code_of
from hpdocs_load.workload import (
    CREATE,
    DELETE,
    LIST,
    MESSAGE,
    OPEN,
    PREVIOUS,
    READY,
    THINK,
    Action,
)


@dataclass
class FakeClock:
    now: float = 1000.0

    def wall(self) -> float:
        return self.now + 5_000_000.0

    def mono(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def sse(*events: tuple[str, str]) -> list[str]:
    lines: list[str] = []
    for name, data in events:
        lines += [f"event: {name}", f"data: {data}", ""]
    return lines


ANSWER = sse(
    ("token", '{"text": "a"}'),
    ("token", '{"text": "b"}'),
    ("sources", '{"sources": []}'),
    ("done", '{"outcome": "completed"}'),
)


@dataclass
class FakeTransport:
    clock: FakeClock
    replies: dict[str, Reply] = field(default_factory=dict)
    stream_status: int = 200
    stream_lines: list[str] = field(default_factory=lambda: list(ANSWER))
    break_after: int | None = None
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)
    created: int = 0

    def call(self, method: str, path: str, *, name: str) -> Reply:
        self.calls.append((method, path, name))
        self.clock.sleep(0.05)
        if name == "create":
            self.created += 1
            return self.replies.get(name, Reply(201, f'{{"id": "c{self.created}"}}'))
        return self.replies.get(name, Reply(204 if method == "DELETE" else 200, "{}"))

    def stream(self, path: str, *, content: str) -> StreamReply:
        self.calls.append(("POST", path, "message"))
        self.contents.append(content)
        return StreamReply(self.stream_status, self._lines())

    def _lines(self) -> Iterator[tuple[float, str]]:
        for i, line in enumerate(self.stream_lines):
            if self.break_after is not None and i == self.break_after:
                raise ConnectionError("peer closed")
            if line.startswith("event: "):
                self.clock.sleep(0.1)  # a frame's lines arrive together, frames 100 ms apart
            yield self.clock.mono(), line


def rig(**kwargs: object) -> tuple[Session, FakeTransport, FakeClock, list[RequestRecord]]:
    clock = FakeClock()
    transport = FakeTransport(clock, **kwargs)  # type: ignore[arg-type]
    records: list[RequestRecord] = []
    return Session(transport, clock, records.append), transport, clock, records


CONVERSATION = (
    Action(THINK, seconds=12.0),
    Action(CREATE),
    Action(MESSAGE, text="question"),
    Action(LIST),
)


# ---------------------------------------------------------------- paths and calls


def test_a_question_creates_streams_then_lists() -> None:
    session, transport, _, records = rig()

    session.run(CONVERSATION)

    assert transport.calls == [
        ("POST", "/v1/conversations", "create"),
        ("POST", "/v1/conversations/c1/messages", "message"),
        ("GET", "/v1/conversations", "list"),
    ]
    assert [r.name for r in records] == ["create", "message", "list"]


def test_ready_open_and_delete_use_their_routes() -> None:
    session, transport, _, _ = rig()

    session.run((Action(READY), Action(CREATE), Action(OPEN), Action(DELETE)))

    assert transport.calls[0] == ("GET", "/ready", "ready")
    assert transport.calls[2] == ("GET", "/v1/conversations/c1", "open")
    assert transport.calls[3] == ("DELETE", "/v1/conversations/c1", "delete")


def test_selecting_opens_the_conversation_before_the_current_one() -> None:
    session, transport, _, _ = rig()

    session.run((Action(CREATE),))
    session.run((Action(CREATE), Action(OPEN, target=PREVIOUS)))

    assert transport.calls[-1] == ("GET", "/v1/conversations/c1", "open")


def test_selecting_with_no_earlier_conversation_opens_the_current_one() -> None:
    session, transport, _, _ = rig()

    session.run((Action(CREATE), Action(OPEN, target=PREVIOUS)))

    assert transport.calls[-1] == ("GET", "/v1/conversations/c1", "open")


def test_a_deleted_conversation_is_never_selected_again() -> None:
    session, transport, _, _ = rig()

    session.run((Action(CREATE),))
    session.run((Action(CREATE), Action(DELETE)))
    session.run((Action(CREATE), Action(OPEN, target=PREVIOUS)))

    assert transport.calls[-1] == ("GET", "/v1/conversations/c1", "open")


def test_think_sleeps_and_sends_nothing() -> None:
    session, transport, clock, records = rig()

    session.run((Action(THINK, seconds=12.0),))

    assert clock.now == pytest.approx(1012.0)
    assert transport.calls == []
    assert records == []


def test_a_failed_create_ends_the_conversation() -> None:
    session, transport, _, records = rig(
        replies={"create": Reply(503, '{"error": {"code": "retrieval_unavailable"}}')}
    )

    session.run(CONVERSATION)

    assert [c[2] for c in transport.calls] == ["create"]
    assert records[0].error_code == "retrieval_unavailable"


# ---------------------------------------------------------------- records


def test_a_completed_stream_is_recorded_with_its_times() -> None:
    session, _, _, records = rig()

    session.run((Action(CREATE), Action(MESSAGE, text="q")))
    message = records[1]

    # create took 50 ms; the stream starts at 1000.05, frames 100 ms apart.
    assert message.start == pytest.approx(5_001_000.05)
    assert message.status == 200
    assert message.ttft_ms == pytest.approx(100.0)
    assert message.duration_ms == pytest.approx(400.0)
    assert message.reached_done
    assert message.error_code is None


def test_plain_calls_are_recorded_with_status_and_duration() -> None:
    session, _, _, records = rig()

    session.run((Action(LIST),))

    assert records == [
        RequestRecord(
            name="list",
            start=5_001_000.0,
            duration_ms=pytest.approx(50.0),  # type: ignore[arg-type]
            status=200,
            ttft_ms=None,
            reached_done=False,
            error_code=None,
        )
    ]


def test_a_rejected_message_keeps_the_backend_code() -> None:
    session, _, _, records = rig(
        stream_status=429, stream_lines=['{"error": {"code": "rate_limited",', '"message": "x"}}']
    )

    session.run((Action(CREATE), Action(MESSAGE, text="q")))

    assert records[1].status == 429
    assert records[1].error_code == "rate_limited"
    assert records[1].ttft_ms is None
    assert not records[1].reached_done


def test_a_connection_failure_is_recorded_without_reading_the_stream() -> None:
    session, _, _, records = rig(stream_status=0, break_after=0)

    session.run((Action(CREATE), Action(MESSAGE, text="q")))

    assert records[1].error_code == "connection_error"


def test_a_stream_cut_mid_answer_is_a_connection_error() -> None:
    session, _, _, records = rig(break_after=4)

    session.run((Action(CREATE), Action(MESSAGE, text="q")))

    assert records[1].error_code == "connection_error"
    assert not records[1].reached_done
    assert records[1].ttft_ms is None


def test_a_stream_breaking_the_contract_is_a_protocol_error() -> None:
    session, _, _, records = rig(stream_lines=["event: token", "data: nope", ""])

    session.run((Action(CREATE), Action(MESSAGE, text="q")))

    assert records[1].error_code == "protocol_error"
    assert not records[1].reached_done


def test_a_mid_stream_error_event_keeps_its_code() -> None:
    session, _, _, records = rig(
        stream_lines=sse(("token", '{"text": "a"}'), ("error", '{"code": "llm_timeout"}'))
    )

    session.run((Action(CREATE), Action(MESSAGE, text="q")))

    assert records[1].error_code == "llm_timeout"
    assert records[1].ttft_ms == pytest.approx(100.0)


def test_the_follow_up_goes_to_the_same_conversation() -> None:
    session, transport, _, _ = rig()

    session.run((Action(CREATE), Action(MESSAGE, text="a"), Action(MESSAGE, text="b")))

    paths = [c[1] for c in transport.calls if c[2] == "message"]
    assert paths == ["/v1/conversations/c1/messages"] * 2
    assert transport.contents == ["a", "b"]


# ---------------------------------------------------------------- error codes


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (200, "{}", None),
        (201, "", None),
        (204, "", None),
        (0, "", "connection_error"),
        (409, '{"error": {"code": "conversation_busy"}}', "conversation_busy"),
        (502, "<html>bad gateway</html>", "http_502"),
        (500, '{"error": {"code": 7}}', "http_500"),
        (500, '{"error": "x"}', "http_500"),
        (500, "[1]", "http_500"),
        (404, "", "http_404"),
    ],
)
def test_error_code_of(status: int, body: str, code: str | None) -> None:
    assert error_code_of(status, body) == code
