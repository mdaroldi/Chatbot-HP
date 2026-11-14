"""ChatApi over HTTP: the backend's /v1 conversation endpoints and its answer stream.

Every failure leaves as an ApiError carrying a code, never response text. Parsing
errors are raised `from None`: a Pydantic or JSON error message quotes the input,
and a chained exception would carry it into any traceback that gets logged.

Only reads are retried, once, after a transport failure; a POST is never repeated,
because a repeated question would be answered and stored twice.
"""

import logging
import re
import time
import uuid
from collections.abc import Callable, Generator, Iterator
from contextlib import closing, contextmanager

import httpx
from pydantic import BaseModel, ValidationError

from hpdocs_ui.api import (
    Conversation,
    ConversationDetail,
    ConversationPage,
    Done,
    SourcesEvent,
    StreamEvent,
    Token,
)
from hpdocs_ui.errors import (
    BACKEND_UNREACHABLE,
    BAD_RESPONSE,
    RATE_LIMITED,
    STREAM_TIMEOUT,
    ApiError,
    parse_retry_after,
)
from hpdocs_ui.identifiers import trusted_identifier
from hpdocs_ui.settings import FrontendSettings
from hpdocs_ui.sse import SseFrameTooLargeError, SseParser

_logger = logging.getLogger(__name__)

_CONNECT_SECONDS = 2.0
_WRITE_SECONDS = 5.0
_POOL_SECONDS = 2.0
# Lists and threads come from Postgres; ten seconds means the backend is stuck.
_PLAIN_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_SECONDS, read=10.0, write=_WRITE_SECONDS, pool=_POOL_SECONDS
)
# A code is a key into the wording table and may be logged, so it must look like one.
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_EVENT_MODELS: dict[str, type[Token | SourcesEvent | Done]] = {
    "token": Token,
    "sources": SourcesEvent,
    "done": Done,
}


class _ErrorInfo(BaseModel):
    code: str
    request_id: str | None = None


class _ErrorBody(BaseModel):
    error: _ErrorInfo


class HttpChatApi:
    def __init__(
        self,
        settings: FrontendSettings,
        client_id: str,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if trusted_identifier(client_id) is None:
            raise ValueError("the client ID is not a valid identifier")
        self._base_url = str(settings.backend_url).rstrip("/")
        self._headers = {"X-Client-ID": client_id}
        self._stream_timeout = httpx.Timeout(
            connect=_CONNECT_SECONDS,
            read=settings.ui_stream_read_timeout_seconds,
            write=_WRITE_SECONDS,
            pool=_POOL_SECONDS,
        )
        self._stream_total_timeout = settings.ui_stream_total_timeout_seconds
        self._transport = transport
        self._clock = clock

    def create_conversation(self) -> Conversation:
        response = self._send("POST", "/v1/conversations", retry=False)
        return _parse(response, Conversation, expected=201)

    def list_conversations(self, cursor: str | None = None) -> ConversationPage:
        params = {} if cursor is None else {"cursor": cursor}
        response = self._send("GET", "/v1/conversations", retry=True, params=params)
        return _parse(response, ConversationPage, expected=200)

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        path = _conversation_path(conversation_id)
        return _parse(self._send("GET", path, retry=True), ConversationDetail, expected=200)

    def delete_conversation(self, conversation_id: str) -> None:
        response = self._send("DELETE", _conversation_path(conversation_id), retry=False)
        _raise_unless(response, expected=204)

    @contextmanager
    def stream_answer(self, conversation_id: str, content: str) -> Iterator[Iterator[StreamEvent]]:
        path = _conversation_path(conversation_id) + "/messages"
        # closing(), not the garbage collector: Streamlit stops a script by raising
        # through this frame, and the backend must see the disconnect right away.
        with closing(self._stream(path, content)) as events:
            yield events

    def _client(self, timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers=self._headers,
            transport=self._transport or httpx.HTTPTransport(retries=0),
        )

    def _send(
        self, method: str, path: str, *, retry: bool, params: dict[str, str] | None = None
    ) -> httpx.Response:
        try:
            return self._request(method, path, params)
        except httpx.TransportError as exc:
            if not retry:
                raise ApiError(BACKEND_UNREACHABLE) from None
            # Type only: messages can carry hosts or internal details.
            _logger.warning("chat API %s failed (%s); retrying once", method, type(exc).__name__)
        try:
            return self._request(method, path, params)
        except httpx.TransportError:
            raise ApiError(BACKEND_UNREACHABLE) from None

    def _request(self, method: str, path: str, params: dict[str, str] | None) -> httpx.Response:
        with self._client(_PLAIN_TIMEOUT) as client:
            return client.request(method, self._base_url + path, params=params)

    def _stream(self, path: str, content: str) -> Generator[StreamEvent]:
        deadline = self._clock() + self._stream_total_timeout
        status: int | None = None
        request_id: str | None = None
        try:
            with (
                self._client(self._stream_timeout) as client,
                client.stream(
                    "POST",
                    self._base_url + path,
                    json={"content": content},
                    headers={"Accept": "text/event-stream"},
                ) as response,
            ):
                status = response.status_code
                request_id = trusted_identifier(response.headers.get("x-request-id"))
                if status != 200 or not _is_event_stream(response):
                    response.read()
                    _raise_unless(response, expected=200)
                    raise ApiError(BAD_RESPONSE, status=status, request_id=request_id)
                yield from _answer_events(response, deadline, self._clock, request_id)
        except httpx.TransportError as exc:
            raise ApiError(_transport_code(exc), status=status, request_id=request_id) from None


def _answer_events(
    response: httpx.Response, deadline: float, clock: Callable[[], float], request_id: str | None
) -> Iterator[StreamEvent]:
    """Events up to the first done or error; reading stops there."""
    parser = SseParser()
    bad = ApiError(BAD_RESPONSE, status=200, request_id=request_id)
    try:
        for chunk in response.iter_text():
            if clock() > deadline:
                raise ApiError(STREAM_TIMEOUT, status=200, request_id=request_id)
            for event in parser.feed(chunk):
                if (yield from _dispatch(event.event, event.data, request_id)):
                    return
        for event in parser.close():
            if (yield from _dispatch(event.event, event.data, request_id)):
                return
    except SseFrameTooLargeError:
        raise bad from None
    # The stream ended without done or error: the answer's end was lost.
    raise bad


def _dispatch(name: str, data: str, request_id: str | None) -> Generator[StreamEvent, None, bool]:
    """Yields the event, if it is one the page knows; returns True if it was terminal."""
    if name == "error":
        error = _model(data, _ErrorInfo, request_id)
        raise ApiError(
            _code(error.code),
            status=200,
            request_id=trusted_identifier(error.request_id) or request_id,
        )
    model = _EVENT_MODELS.get(name)
    if model is None:
        return False
    event = _model(data, model, request_id)
    yield event
    return isinstance(event, Done)


def _model[M: BaseModel](data: str, model: type[M], request_id: str | None) -> M:
    try:
        return model.model_validate_json(data)
    except ValidationError:
        raise ApiError(BAD_RESPONSE, status=200, request_id=request_id) from None


def _parse[M: BaseModel](response: httpx.Response, model: type[M], *, expected: int) -> M:
    _raise_unless(response, expected=expected)
    try:
        return model.model_validate_json(response.content)
    except ValidationError:
        raise _bad_response(response) from None


def _raise_unless(response: httpx.Response, *, expected: int) -> None:
    """Raise the response's ApiError unless it has the expected status."""
    status = response.status_code
    if status == expected:
        return
    if status < 400:
        raise _bad_response(response)
    try:
        info = _ErrorBody.model_validate_json(response.content).error
    except ValidationError:
        raise _bad_response(response) from None
    retry_after = parse_retry_after(response.headers.get("retry-after"))
    raise ApiError(
        _code(info.code),
        status=status,
        request_id=_request_id(response) or trusted_identifier(info.request_id),
        retry_after_seconds=retry_after if info.code == RATE_LIMITED else None,
    )


def _bad_response(response: httpx.Response) -> ApiError:
    return ApiError(BAD_RESPONSE, status=response.status_code, request_id=_request_id(response))


def _request_id(response: httpx.Response) -> str | None:
    return trusted_identifier(response.headers.get("x-request-id"))


def _code(raw: str) -> str:
    return raw if _ERROR_CODE.fullmatch(raw) else BAD_RESPONSE


def _is_event_stream(response: httpx.Response) -> bool:
    content_type: str = response.headers.get("content-type", "")
    return content_type.startswith("text/event-stream")


def _transport_code(exc: httpx.TransportError) -> str:
    if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
        return STREAM_TIMEOUT
    return BACKEND_UNREACHABLE


def _conversation_path(conversation_id: str) -> str:
    """The path for a conversation; only a UUID, rewritten canonically, reaches a URL."""
    try:
        canonical = str(uuid.UUID(conversation_id))
    except ValueError:
        raise ValueError("the conversation ID is not a UUID") from None
    return f"/v1/conversations/{canonical}"
