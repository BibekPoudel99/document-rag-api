import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient
from redis import Redis

from app.api.error_handlers import register_exception_handlers
from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients once at startup; close them at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)

    engine = create_db_engine(settings)
    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = EmbeddingService.from_model_name(
        settings.embedding_model, settings.embedding_batch_size
    )
    vector_store = VectorStore(qdrant, settings.qdrant_collection, embedder.dimension)
    vector_store.ensure_collection()  # fail fast if Qdrant is down or mismatched

    app.state.settings = settings
    app.state.session_factory = create_session_factory(engine)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.qdrant = qdrant
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    logger.info("application started")

    yield

    qdrant.close()
    app.state.redis.close()
    engine.dispose()
    logger.info("application stopped")


app = FastAPI(title="Palm Mind RAG API", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(health.router)