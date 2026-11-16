"""Test doubles for the ingestion and chat pipelines' external systems."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fakeredis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.orm import Session

from app.core.exceptions import LLMUnavailableError
from app.repositories.documents import DocumentRepository
from app.services.chat import ChatService
from app.services.chat_memory import ChatMemory
from app.services.ingestion import IngestionService
from app.services.llm import LLMMessage
from app.services.rag import RagService, Retriever
from app.services.vector_store import RetrievedChunk, make_point_id

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


class FakeLLM:
    """Returns scripted replies in order (then a default); records every prompt."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[list[LLMMessage]] = []
        self.fail = False

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if self.fail:
            raise LLMUnavailableError("fake llm is down")
        return self._replies.pop(0) if self._replies else "fake answer"


class FakeQueryEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text))]


class FakeSearcher:
    """Returns the configured hits (best first) and records how it was called."""

    def __init__(self, hits: Sequence[RetrievedChunk] = ()) -> None:
        self.hits = list(hits)
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"top_k": top_k, "document_ids": document_ids, "score_threshold": score_threshold}
        )
        return self.hits[:top_k]



DEFAULT_DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def make_chunk(
    text: str = "Machine learning learns patterns from data.",
    index: int = 0,
    score: float = 0.8,
    filename: str = "notes.txt",
) -> RetrievedChunk:
    return RetrievedChunk(str(DEFAULT_DOC_ID), filename, index, text, score)


@dataclass
class ChatHarness:
    service: ChatService
    llm: FakeLLM
    embedder: FakeQueryEmbedder
    searcher: FakeSearcher
    memory: ChatMemory
    locks: FakeRedis


def make_chat_harness(
    hits: Sequence[RetrievedChunk] = (),
    replies: Sequence[str] = (),
    memory_redis: Any = None,
) -> ChatHarness:
    """A real ChatService/RagService/ChatMemory wired to fakes.

    Memory uses fakeredis (real list semantics); locks use FakeRedis because
    real redis-py locks need Lua scripting, which fakeredis may not provide.
    """
    llm = FakeLLM(*replies)
    embedder = FakeQueryEmbedder()
    searcher = FakeSearcher(hits)
    retriever = Retriever(embedder, searcher, top_k=4, score_threshold=0.3)
    rag = RagService(llm, retriever, history_messages=6, max_context_chars=6000)
    memory = ChatMemory(
        memory_redis or fakeredis.FakeRedis(decode_responses=True),
        max_messages=20,
        ttl_seconds=3600,
    )
    locks = FakeRedis()
    service = ChatService(memory, rag, locks)  # type: ignore[arg-type]
    return ChatHarness(service, llm, embedder, searcher, memory, locks)