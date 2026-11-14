"""Frontend settings, loaded only from environment variables."""

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """The UI talks to the backend only over HTTP.

    The stream timeouts follow the backend's deadlines (ADR 0011): no byte arrives
    before the first token, so the read timeout must outlast the rewrite, retrieval
    and first-token deadlines and the heartbeat; the total must outlast the longest
    answer. Tests on .env.example check those sums.
    """

    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, frozen=True)

    # Default matches the docker-compose service name and port.
    backend_url: HttpUrl = HttpUrl("http://backend:8000")
    ui_stream_read_timeout_seconds: float = Field(default=60.0, gt=0, le=600, allow_inf_nan=False)
    ui_stream_total_timeout_seconds: float = Field(
        default=185.0, gt=0, le=3600, allow_inf_nan=False
    )
    # The backend's CHAT_MESSAGE_MAX_CHARS; longer input is refused before sending.
    ui_message_max_chars: int = Field(default=1000, ge=1, le=4000)

    @model_validator(mode="after")
    def _total_covers_read(self) -> "FrontendSettings":
        if self.ui_stream_total_timeout_seconds < self.ui_stream_read_timeout_seconds:
            raise ValueError(
                "ui_stream_total_timeout_seconds must not be shorter than "
                "ui_stream_read_timeout_seconds"
            )
        return self
