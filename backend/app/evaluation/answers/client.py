"""HTTP client of the answer harness: one conversation per item, earlier turns replayed,
the last turn read as SSE (Phase 5 rule 3).

Retries: only a 429 or 503 status, which the API returns before the first token (it
primes the answer before the response starts), at most `MAX_RETRIES` times, after the
Retry-After seconds. Nothing else is retried: another status, a transport failure or
an `error` event after text is the item's result, because a retry there would change
what the benchmark measures. Latencies are taken on the attempt that answered, so a
retry's wait never counts as answer time.

Deadlines: `read` between bytes (keepalives arrive every few seconds, so a silent
minute is a hung stream) and `total` per attempt, the backend's own longest answer.
Failure codes are the server's `error.code`, or a `HarnessCode`; nothing from a body
other than a validated code is kept.
"""

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import httpx

from app.domain.answer_dataset import EvalItem
from app.evaluation.answers.sse import (
    Done,
    FrameParser,
    MalformedEventError,
    parse_done,
    parse_error,
)

MAX_RETRIES = 2
RETRYABLE = frozenset({429, 503})
DEFAULT_RETRY_AFTER_SECONDS = 1.0
# Beyond this the stack is not briefly busy but stuck; waiting longer only hides it.
MAX_RETRY_AFTER_SECONDS = 60.0
_CODE = re.compile(r"[a-z][a-z_]*")


class HarnessCode(StrEnum):
    """Failures the harness names itself, beside the server's error codes."""

    SETUP_TURN_FAILED = "setup_turn_failed"
    READ_TIMEOUT = "read_timeout"
    TOTAL_TIMEOUT = "total_timeout"
    TRANSPORT_ERROR = "transport_error"
    MALFORMED_EVENT = "malformed_event"
    INCOMPLETE_STREAM = "incomplete_stream"


@dataclass(frozen=True)
class Timeouts:
    read: float = 60.0
    # The backend's longest answer: rewrite + total LLM deadlines + one minute (ADR 0010).
    total: float = 185.0


@dataclass(frozen=True)
class Created:
    conversation_id: str
    retries: int


@dataclass(frozen=True)
class Answered:
    done: Done
    retries: int
    ttft_ms: float | None  # None when no token arrived (a refusal is sent whole)
    total_ms: float


@dataclass(frozen=True)
class Failed:
    code: str
    http_status: int | None
    retries: int


@dataclass(frozen=True)
class ItemResult:
    """One item's outcome; `error` set means it is not scored and counts against the run."""

    item_id: str
    client_id: str
    conversation_id: str | None
    retries: int
    answer: Answered | None
    error: str | None = None
    cause: str | None = None  # the earlier turn's code, for setup_turn_failed
    http_status: int | None = None
    setup_refused: int = 0  # earlier turns that were refused (kept, as a user would)

    @property
    def ok(self) -> bool:
        return self.error is None


class _RetryableError(Exception):
    def __init__(self, failure: Failed, seconds: float) -> None:
        super().__init__(failure.code)
        self.failure, self.seconds = failure, seconds


def retry_after_seconds(response: httpx.Response) -> float:
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return DEFAULT_RETRY_AFTER_SECONDS
    if not math.isfinite(seconds) or seconds < 0:
        return DEFAULT_RETRY_AFTER_SECONDS
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def _field(value: object, key: str) -> object:
    return cast(dict[str, object], value).get(key) if isinstance(value, dict) else None


def _body_field(response: httpx.Response, key: str) -> object:
    try:
        body: object = response.json()
    except ValueError:
        return None
    return _field(body, key)


def _error_code(response: httpx.Response) -> str:
    """The unified error body's code, or `http_<status>` when there is none to trust."""
    code = _field(_body_field(response, "error"), "code")
    if isinstance(code, str) and _CODE.fullmatch(code):
        return code
    return f"http_{response.status_code}"


def _status_failure(response: httpx.Response) -> Failed:
    failure = Failed(_error_code(response), response.status_code, 0)
    if response.status_code in RETRYABLE:
        raise _RetryableError(failure, retry_after_seconds(response))
    return failure


class ChatClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        timeouts: Timeouts,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._http, self._timeouts, self._clock, self._sleep = http, timeouts, clock, sleep
        self._timeout = httpx.Timeout(timeouts.read)

    async def _with_retries[T](
        self, attempt: Callable[[], Awaitable[T | Failed]]
    ) -> tuple[T | Failed, int]:
        retries = 0
        while True:
            try:
                return await attempt(), retries
            except _RetryableError as retry:
                if retries == MAX_RETRIES:
                    return retry.failure, retries
                retries += 1
                await self._sleep(retry.seconds)

    async def create_conversation(self, client_id: str) -> Created | Failed:
        result, retries = await self._with_retries(lambda: self._create(client_id))
        if isinstance(result, Failed):
            return Failed(result.code, result.http_status, retries)
        return Created(result, retries)

    async def _create(self, client_id: str) -> str | Failed:
        try:
            response = await self._http.post(
                "/v1/conversations", headers={"X-Client-ID": client_id}, timeout=self._timeout
            )
        except httpx.TransportError as exc:
            return _transport_failure(exc, None)
        if response.status_code != httpx.codes.CREATED:
            return _status_failure(response)
        conversation_id = _body_field(response, "id")
        if not isinstance(conversation_id, str) or not conversation_id:
            return Failed(HarnessCode.MALFORMED_EVENT, response.status_code, 0)
        return conversation_id

    async def delete_conversation(self, conversation_id: str, client_id: str) -> Failed | None:
        """None once deleted; the run keeps a failure so leftovers are visible."""
        try:
            response = await self._http.delete(
                f"/v1/conversations/{conversation_id}",
                headers={"X-Client-ID": client_id},
                timeout=self._timeout,
            )
        except httpx.TransportError as exc:
            return _transport_failure(exc, None)
        if response.status_code != httpx.codes.NO_CONTENT:
            return Failed(_error_code(response), response.status_code, 0)
        return None

    async def send_turn(
        self, conversation_id: str, client_id: str, content: str
    ) -> Answered | Failed:
        result, retries = await self._with_retries(
            lambda: self._attempt(conversation_id, client_id, content)
        )
        if isinstance(result, Answered):
            return Answered(result.done, retries, result.ttft_ms, result.total_ms)
        return Failed(result.code, result.http_status, retries)

    async def _attempt(
        self, conversation_id: str, client_id: str, content: str
    ) -> Answered | Failed:
        status: int | None = None
        # Before the request: the API primes the answer (retrieval and the first model
        # token) before it sends the headers, and that wait is part of the latency.
        start = self._clock()
        try:
            async with asyncio.timeout(self._timeouts.total):
                async with self._http.stream(
                    "POST",
                    f"/v1/conversations/{conversation_id}/messages",
                    json={"content": content},
                    headers={"X-Client-ID": client_id},
                    timeout=self._timeout,
                ) as response:
                    status = response.status_code
                    if status != httpx.codes.OK:
                        await response.aread()
                        return _status_failure(response)
                    return await self._read(response, conversation_id, start)
        except TimeoutError:
            return Failed(HarnessCode.TOTAL_TIMEOUT, status, 0)
        except httpx.TransportError as exc:
            return _transport_failure(exc, status)

    async def _read(
        self, response: httpx.Response, conversation_id: str, start: float
    ) -> Answered | Failed:
        parser, first_token = FrameParser(), None
        try:
            async for line in response.aiter_lines():
                frame = parser.feed(line)
                if frame is None:
                    continue
                if frame.event == "token" and first_token is None:
                    first_token = self._clock()
                elif frame.event == "error":
                    return Failed(parse_error(frame.data).code, response.status_code, 0)
                elif frame.event == "done":
                    return self._answered(frame.data, conversation_id, start, first_token)
        except MalformedEventError:
            return Failed(HarnessCode.MALFORMED_EVENT, response.status_code, 0)
        return Failed(HarnessCode.INCOMPLETE_STREAM, response.status_code, 0)

    def _answered(
        self, data: str, conversation_id: str, start: float, first_token: float | None
    ) -> Answered:
        done = parse_done(data)
        if done.conversation_id != conversation_id:
            raise MalformedEventError("done: conversation_id is not this conversation")
        end = self._clock()
        ttft = None if first_token is None else (first_token - start) * 1000
        return Answered(done, 0, ttft, (end - start) * 1000)


def _transport_failure(exc: httpx.TransportError, status: int | None) -> Failed:
    if isinstance(exc, httpx.ReadTimeout):
        return Failed(HarnessCode.READ_TIMEOUT, status, 0)
    return Failed(HarnessCode.TRANSPORT_ERROR, status, 0)


async def run_item(chat: ChatClient, item: EvalItem, client_id: str) -> ItemResult:
    """Create a conversation, replay the earlier turns, then ask the last one.

    A refused earlier turn stays in the history, as it would for a user; a failed one
    ends the item as `setup_turn_failed`, because the last turn would then be asked
    with a history the dataset did not write.
    """
    created = await chat.create_conversation(client_id)
    retries = created.retries
    if isinstance(created, Failed):
        return ItemResult(
            item.id, client_id, None, retries, None, created.code, None, created.http_status
        )
    conversation = created.conversation_id
    refused = 0
    *earlier, last = item.turns
    for turn in earlier:
        result = await chat.send_turn(conversation, client_id, turn)
        retries += result.retries
        if isinstance(result, Failed):
            return ItemResult(
                item.id,
                client_id,
                conversation,
                retries,
                None,
                HarnessCode.SETUP_TURN_FAILED,
                result.code,
                result.http_status,
                refused,
            )
        refused += result.done.refusal_reason is not None
    result = await chat.send_turn(conversation, client_id, last)
    retries += result.retries
    if isinstance(result, Failed):
        return ItemResult(
            item.id,
            client_id,
            conversation,
            retries,
            None,
            result.code,
            None,
            result.http_status,
            refused,
        )
    return ItemResult(item.id, client_id, conversation, retries, result, setup_refused=refused)
