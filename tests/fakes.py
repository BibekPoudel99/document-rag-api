"""Test doubles for the ingestion pipeline's external systems."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.orm import Session

from app.repositories.documents import DocumentRepository
from app.services.ingestion import IngestionService
from app.services.vector_store import make_point_id


class FakeEmbedder:
    dimension = 2

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding model exploded")
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class FakeVectorStore:
    """Keeps chunks in a dict; can simulate failures, including a partial write."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, list[str]] = {}
        self.fail_on_upsert = False
        self.fail_on_delete = False
        self.upsert_calls = 0

    def upsert_chunks(
        self,
        document_id: uuid.UUID,
        filename: str,
        chunking_strategy: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> list[uuid.UUID]:
        self.upsert_calls += 1
        if self.fail_on_upsert:
            self.documents[document_id] = list(chunks[:1])  # a partial write...
            raise RuntimeError("qdrant down")  # ...then the failure
        self.documents[document_id] = list(chunks)
        return [make_point_id(document_id, index) for index in range(len(chunks))]

    def delete_document(self, document_id: uuid.UUID) -> None:
        if self.fail_on_delete:
            raise RuntimeError("qdrant still down")
        self.documents.pop(document_id, None)


class FakeLock:
    def __init__(self, redis: "FakeRedis", name: str) -> None:
        self._redis = redis
        self._name = name

    def acquire(self, blocking: bool = False) -> bool:
        if self._redis.unavailable:
            raise RedisConnectionError("redis is down")
        if self._name in self._redis.held:
            return False
        self._redis.held.add(self._name)
        return True

    def release(self) -> None:
        self._redis.held.discard(self._name)


class FakeRedis:
    def __init__(self) -> None:
        self.held: set[str] = set()
        self.unavailable = False

    def lock(self, name: str, timeout: float | None = None, blocking: bool = True) -> FakeLock:
        return FakeLock(self, name)


@dataclass
class IngestionHarness:
    service: IngestionService
    embedder: FakeEmbedder
    vector_store: FakeVectorStore
    redis: FakeRedis
    session: Session


def make_harness(session: Session, max_upload_bytes: int = 1_000_000) -> IngestionHarness:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    redis_client = FakeRedis()
    service = IngestionService(
        DocumentRepository(session),
        embedder,  # type: ignore[arg-type]
        vector_store,  # type: ignore[arg-type]
        redis_client,  # type: ignore[arg-type]
        max_upload_bytes=max_upload_bytes,
        default_chunk_size=800,
        default_chunk_overlap=100,
    )
    return IngestionHarness(service, embedder, vector_store, redis_client, session)