"""Liveness and readiness endpoints."""

from collections.abc import Sequence
from http import HTTPStatus
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies import get_app_settings, get_readiness_probes
from app.core.settings import LLMProviderName, Settings
from app.domain.health import DependencyProbe, FailureReason, ProbeStatus, ReadinessReport
from app.services.readiness import check_readiness

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class CheckResponse(BaseModel):
    status: ProbeStatus
    detail: FailureReason | None = None
    info: dict[str, str] | None = None  # what a ready dependency serves (models, index)


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckResponse]
    llm: LLMProviderName  # which model answers; the key is never shown

    @classmethod
    def from_report(cls, report: ReadinessReport, llm: LLMProviderName) -> "ReadinessResponse":
        return cls(
            status="ready" if report.ready else "not_ready",
            llm=llm,
            checks={
                result.name: CheckResponse(
                    status=result.status,
                    detail=result.detail,
                    info=dict(result.info) if result.info is not None else None,
                )
                for result in report.results
            },
        )


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness only: answers while the process can serve HTTP, without touching
    dependencies, so an outage of Postgres or Qdrant never gets the API restarted.
    """
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={HTTPStatus.SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(
    settings: Annotated[Settings, Depends(get_app_settings)],
    probes: Annotated[Sequence[DependencyProbe], Depends(get_readiness_probes)],
) -> JSONResponse:
    """Readiness: 503 with a per-dependency status document when anything is down.

    The body is a status document, not the error shape, because orchestrators and
    operators read it as data; it still gets X-Request-ID from the middleware.
    """
    report = await check_readiness(probes, settings.readiness_timeout_seconds)
    body = ReadinessResponse.from_report(report, settings.llm_provider)
    status = HTTPStatus.OK if report.ready else HTTPStatus.SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status, content=body.model_dump(mode="json", exclude_none=True))
