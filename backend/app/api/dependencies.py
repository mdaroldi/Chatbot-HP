"""FastAPI dependencies: how routes obtain settings and infrastructure adapters."""

from collections.abc import Sequence

from fastapi import Request

from app.core.settings import Settings
from app.domain.health import DependencyProbe


def get_app_settings(request: Request) -> Settings:
    """Settings the app was built with (not re-read from the environment)."""
    settings: Settings = request.app.state.settings
    return settings


def get_readiness_probes(request: Request) -> Sequence[DependencyProbe]:
    """Probes registered on the app; tests override this with fakes."""
    probes: Sequence[DependencyProbe] = request.app.state.readiness_probes
    return probes
