import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient
from redis import Redis

from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients once at startup; close them at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)

    engine = create_db_engine(settings)
    app.state.settings = settings
    app.state.session_factory = create_session_factory(engine)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.qdrant = QdrantClient(url=settings.qdrant_url)
    logger.info("application started")

    yield

    app.state.qdrant.close()
    app.state.redis.close()
    engine.dispose()
    logger.info("application stopped")


app = FastAPI(title="Palm Mind RAG API", lifespan=lifespan)
app.include_router(health.router)