"""The page's URL parameters: who the browser is, and which conversation is open.

The URL is the only place an anonymous browser keeps its client ID, so a reload or a
shared link opens the same conversations (the page warns not to share it). Values are
validated, never repaired: an invalid client ID is replaced by a new one, and the
conversation is dropped with it, because it belongs to the old ID.
"""

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from hpdocs_ui.identifiers import new_client_id, trusted_identifier

CLIENT_PARAM = "cid"
CONVERSATION_PARAM = "conv"


@dataclass(frozen=True)
class UrlState:
    client_id: str
    conversation_id: str | None


def resolve(params: Mapping[str, str], *, new_id: Callable[[], str] = new_client_id) -> UrlState:
    client_id = trusted_identifier(params.get(CLIENT_PARAM))
    if client_id is None:
        return UrlState(new_id(), None)
    return UrlState(client_id, _conversation_id(params.get(CONVERSATION_PARAM)))


def query_params(state: UrlState) -> dict[str, str]:
    params = {CLIENT_PARAM: state.client_id}
    if state.conversation_id is not None:
        params[CONVERSATION_PARAM] = state.conversation_id
    return params


def _conversation_id(raw: str | None) -> str | None:
    """The canonical UUID, so the same conversation always has the same URL."""
    try:
        return str(uuid.UUID(raw or ""))
    except ValueError:
        return None
