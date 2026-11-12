"""Frontend settings, loaded only from environment variables."""

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """The UI talks to the backend only over HTTP; this is the one thing it needs."""

    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, frozen=True)

    # Default matches the docker-compose service name and port.
    backend_url: HttpUrl = HttpUrl("http://backend:8000")
