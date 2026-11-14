import logging
from collections.abc import Callable

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.api.deps import DbSession, Qdrant, RedisClient
from app.schemas.health import HealthResponse, ServiceState

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


def _probe(name: str, check: Callable[[], object]) -> ServiceState:
    """Run one dependency check. A health probe must report failure, never raise,
    so this is the one place where catching a broad Exception is deliberate."""
    try:
        check()
    except Exception:
        logger.exception("health check failed for %s", name)
        return "down"
    return "up"


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    db: DbSession,
    redis_client: RedisClient,
    qdrant: Qdrant,
) -> HealthResponse:
    services: dict[str, ServiceState] = {
        "postgres": _probe("postgres", lambda: db.execute(text("SELECT 1"))),
        "redis": _probe("redis", redis_client.ping),
        "qdrant": _probe("qdrant", qdrant.get_collections),
    }
    all_up = all(state == "up" for state in services.values())
    if not all_up:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if all_up else "degraded", services=services)