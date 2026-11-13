"""POST /v1/conversations/{id}/messages: one question, answered as server-sent events.

Priming: the answer runs up to its first event (a token, or the refusal decision)
before the response starts, so a failure before the first token is an HTTP error in
the unified shape and nothing is stored. While priming, a watcher reads the ASGI
`receive` channel: the body is already read, so the next message is the client
leaving, and the answer is then cancelled (the service removes the turn) and no
response is sent. The watcher is stopped before the response starts, because the
streaming response then listens for the disconnect itself.

Streaming: `token` events, then `sources` and `done`; or, if the answer fails after
text was shown, one `error` event. The next event is awaited as one task that a
heartbeat timeout never cancels (cancelling it would cancel the answer); every
`SSE_HEARTBEAT_SECONDS` of silence sends a `: keepalive` comment instead. However
the response ends, the response itself closes the answer, in a shielded task: an
answer the client left is stored `cancelled` before the request finishes.
"""

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import Receive, Scope, Send

from app.api.conversations import SourceOut
from app.api.dependencies import get_app_settings, get_chat_service, require_client_id
from app.api.rate_limits import limit_messages
from app.core.request_context import get_request_id
from app.core.settings import Settings
from app.core.tasks import shielded
from app.services.chat import AnswerEvent, AnswerFailed, AnswerToken, ChatService

logger = logging.getLogger(__name__)

KEEPALIVE = b": keepalive\n\n"
# The same fixed text the request-ID middleware gives any unhandled error.
INTERNAL_ERROR = ("internal_error", "Internal server error.")
_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

Events = AsyncGenerator[AnswerEvent, None]


class MessageContent(BaseModel):
    """The request fields; the per-app subclass adds this app's length limit."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str


def message_request_model(settings: Settings) -> type[MessageContent]:
    class MessageRequestBody(MessageContent):
        content: str = Field(min_length=1, max_length=settings.chat_message_max_chars)

    return MessageRequestBody


def frame(event: str, data: object) -> bytes:
    """One SSE event; JSON escapes newlines, so the data is always a single line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def event_frames(event: AnswerEvent) -> list[bytes]:
    if isinstance(event, AnswerToken):
        return [frame("token", {"text": event.text})]
    if isinstance(event, AnswerFailed):
        return [error_frame(event.code, event.message, event.turn.assistant.id)]
    answer, turn = event.answer, event.turn
    sources = [SourceOut.of(source).model_dump() for source in answer.sources]
    done = {
        "conversation_id": turn.assistant.conversation_id,
        "user_message_id": turn.user.id,
        "message_id": turn.assistant.id,
        "outcome": answer.outcome.value,
        "refusal": answer.refusal_reason is not None,
        "refusal_reason": None if answer.refusal_reason is None else answer.refusal_reason.value,
        "content": answer.content,
        "citations": list(answer.citations),
        "flags": dataclasses.asdict(answer.flags),
    }
    return [frame("sources", {"sources": sources}), frame("done", done)]


def error_frame(code: str, message: str, message_id: str | None) -> bytes:
    data = {"code": code, "message": message, "request_id": get_request_id()}
    return frame("error", {**data, "message_id": message_id})


async def _until_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


async def _next_event(events: Events) -> AnswerEvent:
    return await anext(events)


async def prime(events: Events, receive: Receive) -> AnswerEvent | None:
    """The first event, or None if the client left first (the answer is cancelled).

    Failures of the answer are raised as they are, for the error handlers.
    """
    first = asyncio.ensure_future(_next_event(events))
    watcher = asyncio.ensure_future(_until_disconnect(receive))
    try:
        await asyncio.wait({first, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Whatever happened, neither task outlives priming.
        for task in (watcher, first):
            if not task.done():
                task.cancel()
        await asyncio.wait({watcher, first})
    if first.cancelled():
        return None
    return first.result()


class AnswerStream:
    """The SSE body for one answer, and the explicit close of what it reads."""

    def __init__(self, events: Events, first: AnswerEvent, heartbeat_seconds: float) -> None:
        self._events, self._first, self._heartbeat = events, first, heartbeat_seconds
        self._pending: asyncio.Future[AnswerEvent] | None = None
        self.body = self._frames()

    async def _frames(self) -> AsyncGenerator[bytes, None]:
        event = self._first
        while True:
            for chunk in event_frames(event):
                yield chunk
            if not isinstance(event, AnswerToken):
                return
            pending = self._pending = asyncio.ensure_future(_next_event(self._events))
            while not (await asyncio.wait({pending}, timeout=self._heartbeat))[0]:
                yield KEEPALIVE
            self._pending = None
            try:
                event = pending.result()
            except Exception:
                # The service already stored the answer as failed; the client gets a
                # fixed error, the log the exception type.
                logger.exception("answer stream failed")
                yield error_frame(*INTERNAL_ERROR, None)
                return

    async def close(self) -> None:
        """Cancel the wait for an event (the service stores what was shown), then close
        the frames and the events, so nothing is left to the garbage collector.
        """
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.cancel()
            await asyncio.wait({pending})
        await self.body.aclose()
        await self._events.aclose()


class AnswerStreamResponse(StreamingResponse):
    def __init__(self, stream: AnswerStream) -> None:
        super().__init__(stream.body, media_type="text/event-stream", headers=_STREAM_HEADERS)
        self._stream = stream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette stops reading the body when the client leaves but never closes
            # it; closing here runs outside its cancelled scope, and in its own task so
            # a second cancellation cannot interrupt it.
            await shielded(self._stream.close())


class NoResponse(Response):
    """The client left during priming: there is no one to answer."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:  # noqa: ARG002
        return


def build_messages_router(settings: Settings) -> APIRouter:
    body_model = message_request_model(settings)
    router = APIRouter(
        prefix="/v1/conversations", tags=["messages"], dependencies=[Depends(limit_messages)]
    )

    @router.post(
        "/{conversation_id}/messages",
        response_class=StreamingResponse,
        responses={
            200: {"content": {"text/event-stream": {}}, "description": "The answer as SSE."},
            400: {"description": "`client_id_required`."},
            404: {"description": "`conversation_not_found`."},
            409: {"description": "`conversation_busy`."},
            422: {"description": "Invalid request (input not echoed)."},
            429: {"description": "`rate_limited`, with Retry-After."},
            500: {"description": "`unknown_document` or `internal_error`."},
            502: {"description": "`llm_error`."},
            503: {
                "description": "`llm_unavailable`, `llm_timeout`, `retrieval_unavailable` "
                "or `models_not_ready`."
            },
        },
    )
    async def post_message(  # noqa: PLR0913, PLR0917 - FastAPI injects each by name
        conversation_id: uuid.UUID,
        body: body_model,  # type: ignore[valid-type]  # built per app from settings
        request: Request,
        client_id: Annotated[str, Depends(require_client_id)],
        service: Annotated[ChatService, Depends(get_chat_service)],
        app_settings: Annotated[Settings, Depends(get_app_settings)],
    ) -> Response:
        """Answer one question from the manuals, streamed as server-sent events."""
        content = cast(MessageContent, body).content
        events = service.answer(client_id, str(conversation_id), content)
        first = await prime(events, request.receive)
        if first is None:
            return NoResponse()
        return AnswerStreamResponse(AnswerStream(events, first, app_settings.sse_heartbeat_seconds))

    return router
