from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session

def get_db(request: Request) -> Iterator[Session]:
    """One session per request, always closed afterwards"""
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