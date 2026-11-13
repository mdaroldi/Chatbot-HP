"""/v1/conversations: create, list, read and delete one client's conversations.

Every route requires X-Client-ID and passes it to the store, which applies it inside
each statement; another client's conversation is indistinguishable from a missing
one (404). Responses show what the chat UI needs; the stored trace (tokens, timings,
chunk IDs, model) stays internal.
"""

import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from app.api.cursor import decode_cursor, encode_cursor
from app.api.dependencies import get_chat_store, require_client_id
from app.api.rate_limits import limit_conversations
from app.domain.chat import Conversation, Message, Source
from app.domain.chat_store import ChatStore, ConversationNotFoundError

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_CURSOR_CHARS = 256

ClientId = Annotated[str, Depends(require_client_id)]
Store = Annotated[ChatStore, Depends(get_chat_store)]

router = APIRouter(
    prefix="/v1/conversations", tags=["conversations"], dependencies=[Depends(limit_conversations)]
)


class ConversationOut(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, conversation: Conversation) -> "ConversationOut":
        return cls(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class ConversationList(BaseModel):
    items: list[ConversationOut]
    next_cursor: str | None


class PagesOut(BaseModel):
    pdf_start: int
    pdf_end: int
    printed_start: int | None
    printed_end: int | None


class SourceOut(BaseModel):
    tag: str
    document_id: str
    document_title: str
    section_path: list[str]
    pages: PagesOut
    has_warning: bool
    snippet: str

    @classmethod
    def of(cls, source: Source) -> "SourceOut":
        pages = source.pages
        return cls(
            tag=source.tag,
            document_id=source.doc_id,
            document_title=source.document_title,
            section_path=list(source.section_path),
            pages=PagesOut(
                pdf_start=pages.pdf_start,
                pdf_end=pages.pdf_end,
                printed_start=pages.printed_start,
                printed_end=pages.printed_end,
            ),
            has_warning=source.has_warning,
            snippet=source.snippet,
        )


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    outcome: Literal["pending", "completed", "refused", "failed", "cancelled"] | None
    refusal: bool
    refusal_reason: Literal["below_threshold", "model_refused", "provider_refused"] | None
    citations: list[str]
    sources: list[SourceOut]

    @classmethod
    def of(cls, message: Message) -> "MessageOut":
        return cls(
            id=message.id,
            role=message.role.value,
            content=message.content,
            created_at=message.created_at,
            outcome=None if message.outcome is None else message.outcome.value,
            refusal=message.refusal,
            refusal_reason=None if message.refusal_reason is None else message.refusal_reason.value,
            citations=list(message.citations),
            sources=[SourceOut.of(s) for s in message.sources],
        )


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


@router.post("", status_code=HTTPStatus.CREATED)
async def create_conversation(client_id: ClientId, store: Store) -> ConversationOut:
    return ConversationOut.of(await store.create_conversation(client_id))


@router.get("")
async def list_conversations(
    client_id: ClientId,
    store: Store,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_CHARS)] = None,
) -> ConversationList:
    """Newest first by last activity; pass next_cursor back to get the next page."""
    after = None if cursor is None else decode_cursor(cursor)
    page = await store.list_conversations(client_id, limit=limit, after=after)
    return ConversationList(
        items=[ConversationOut.of(c) for c in page.items],
        next_cursor=None if page.next_cursor is None else encode_cursor(page.next_cursor),
    )


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID, client_id: ClientId, store: Store
) -> ConversationDetail:
    thread = await store.get_thread(client_id, str(conversation_id))
    if thread is None:
        raise ConversationNotFoundError
    summary = ConversationOut.of(thread.conversation)
    return ConversationDetail(
        **summary.model_dump(), messages=[MessageOut.of(m) for m in thread.messages]
    )


@router.delete("/{conversation_id}", status_code=HTTPStatus.NO_CONTENT, response_class=Response)
async def delete_conversation(
    conversation_id: uuid.UUID, client_id: ClientId, store: Store
) -> Response:
    """Deletes the conversation and all its messages at once (personal data)."""
    if not await store.delete_conversation(client_id, str(conversation_id)):
        raise ConversationNotFoundError
    return Response(status_code=HTTPStatus.NO_CONTENT)
