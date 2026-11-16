import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from qdrant_client import QdrantClient
from redis import Redis

from app.api.error_handlers import register_exception_handlers
from app.api.routes import chat, documents, health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory, init_db
from app.services.chat_memory import ChatMemory
from app.services.embeddings import EmbeddingService
from app.services.llm import OllamaClient
from app.services.rag import RagService, Retriever
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients once at startup; close them at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)

    engine = create_db_engine(settings)
    init_db(engine)  # fail fast if Postgres is down; creates missing tables

    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = EmbeddingService.from_model_name(
        settings.embedding_model, settings.embedding_batch_size
    )
    vector_store = VectorStore(qdrant, settings.qdrant_collection, embedder.dimension)
    vector_store.ensure_collection()  # fail fast if Qdrant is down or mismatched

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)

    llm_http = httpx.Client(
        base_url=settings.llm_base_url,
        timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0),
    )
    llm = OllamaClient(
        llm_http,
        model=settings.llm_model,
        context_tokens=settings.llm_context_tokens,
        default_max_tokens=settings.llm_max_answer_tokens,
    )
    retriever = Retriever(
        embedder,
        vector_store,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )

    app.state.settings = settings
    app.state.session_factory = create_session_factory(engine)
    app.state.redis = redis_client
    app.state.qdrant = qdrant
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.rag_service = RagService(
        llm,
        retriever,
        history_messages=settings.rag_history_messages,
        max_context_chars=settings.rag_max_context_chars,
    )
    app.state.chat_memory = ChatMemory(
        redis_client,
        max_messages=settings.chat_memory_max_messages,
        ttl_seconds=settings.chat_memory_ttl_seconds,
    )
    logger.info("application started (llm model: %s)", settings.llm_model)

    yield

    llm_http.close()
    qdrant.close()
    redis_client.close()
    engine.dispose()
    logger.info("application stopped")


app = FastAPI(title="Palm Mind RAG API", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)