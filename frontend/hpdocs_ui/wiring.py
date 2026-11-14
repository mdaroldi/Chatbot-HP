"""The page's one seam to the backend; tests replace build_api with a fake."""

import logging

from hpdocs_ui.api import ChatApi
from hpdocs_ui.http_api import HttpChatApi
from hpdocs_ui.settings import FrontendSettings

# Their INFO and DEBUG lines carry request URLs, which hold conversation IDs.
_HTTP_LOGGERS = ("httpx", "httpcore")


def build_api(settings: FrontendSettings, client_id: str) -> ChatApi:
    return HttpChatApi(settings, client_id)


def quiet_http_logs() -> None:
    for name in _HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
