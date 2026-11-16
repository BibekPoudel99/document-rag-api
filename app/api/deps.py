from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session

from app.repositories.documents import DocumentRepository
from app.services.chat import ChatService
from app.services.ingestion import IngestionService


def get_db(request: Request) -> Iterator[Session]:
    """One SQLAlchemy session per request, always closed afterwards."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_qdrant(request: Request) -> QdrantClient:
    return request.app.state.qdrant


DbSession = Annotated[Session, Depends(get_db)]
RedisClient = Annotated[Redis, Depends(get_redis)]
Qdrant = Annotated[QdrantClient, Depends(get_qdrant)]


def get_ingestion_service(
    request: Request, db: DbSession, redis_client: RedisClient
) -> IngestionService:
    """Shared, expensive objects come from app.state; the DB session is per request."""
    state = request.app.state
    settings = state.settings
    return IngestionService(
        DocumentRepository(db),
        state.embedder,
        state.vector_store,
        redis_client,
        max_upload_bytes=settings.max_upload_bytes,
        default_chunk_size=settings.default_chunk_size,
        default_chunk_overlap=settings.default_chunk_overlap,
    )


def get_chat_service(request: Request, redis_client: RedisClient) -> ChatService:
    """A cheap per-request wrapper around the shared RAG service and chat memory."""
    state = request.app.state
    return ChatService(state.chat_memory, state.rag_service, redis_client)


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]