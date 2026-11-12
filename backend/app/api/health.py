"""Liveness endpoint."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness only: answers while the process can serve HTTP, without touching
    dependencies, so an outage of Postgres or Qdrant never gets the API restarted.
    """
    return HealthResponse()
