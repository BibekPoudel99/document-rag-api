"""Ingestion pipeline: validate -> extract -> chunk -> embed -> store.

Consistency model: Postgres and Qdrant cannot share a transaction, so the order
of steps is chosen so that the LAST step (the Postgres commit that marks the
document 'completed') is the commit point. Vectors are written first; if
anything fails we delete them and mark the document 'failed'.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import LockError, RedisError
from redis.lock import Lock

from app.core.exceptions import (
    DocumentBusyError,
    IngestionFailedError,
    InvalidChunkingParametersError,
)
from app.db.models import Document, DocumentStatus
from app.repositories.documents import ChunkRecord, DocumentRepository
from app.services.chunking import ChunkingStrategy, chunk_text
from app.services.embeddings import EmbeddingService
from app.services.extraction import extract_text
from app.services.file_validation import ValidatedUpload, validate_upload
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_DOCUMENT_NAMESPACE = uuid.UUID("b3a1f0c4-7d52-4e8a-9a3e-5c1d2f6e8b07")
_LOCK_SECONDS = 600  # a crashed worker's lock disappears after this long


def make_document_id(
    content_hash: str, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int
) -> uuid.UUID:
    """Same content + same chunking settings -> same ID, so uploads are idempotent."""
    key = f"{content_hash}:{strategy.value}:{chunk_size}:{chunk_overlap}"
    return uuid.uuid5(_DOCUMENT_NAMESPACE, key)


def ingestion_lock_name(document_id: uuid.UUID) -> str:
    return f"ingest:{document_id}"


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    created: bool  # False when an identical, completed document was reused


class IngestionService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedder: EmbeddingService,
        vector_store: VectorStore,
        redis_client: Redis,
        *,
        max_upload_bytes: int,
        default_chunk_size: int,
        default_chunk_overlap: int,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._redis = redis_client
        self._max_upload_bytes = max_upload_bytes
        self._default_chunk_size = default_chunk_size
        self._default_chunk_overlap = default_chunk_overlap

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    def ingest(
        self,
        filename: str | None,
        content: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        # Cheap checks first: nothing is written until the input is known to be good.
        upload = validate_upload(filename, content, self._max_upload_bytes)
        size, overlap = self._resolve_chunking(chunk_size, chunk_overlap)
        text = extract_text(content, upload.file_type)
        chunks = self._chunk(text, strategy, size, overlap)

        content_hash = hashlib.sha256(content).hexdigest()
        document_id = make_document_id(content_hash, strategy, size, overlap)

        lock = self._redis.lock(
            ingestion_lock_name(document_id), timeout=_LOCK_SECONDS, blocking=False
        )
        try:
            acquired = lock.acquire(blocking=False)
        except RedisError as exc:
            logger.error("could not reach Redis for the ingestion lock: %s", exc)
            raise IngestionFailedError(
                "Document processing could not start; a required service is unavailable."
            ) from exc
        if not acquired:
            raise DocumentBusyError(
                "This document is already being processed. Try again shortly."
            )
        try:
            return self._ingest_locked(
                document_id, upload, content, content_hash, strategy, size, overlap, chunks
            )
        finally:
            self._release(lock)

    def _resolve_chunking(
        self, chunk_size: int | None, chunk_overlap: int | None
    ) -> tuple[int, int]:
        size = self._default_chunk_size if chunk_size is None else chunk_size
        if chunk_overlap is None:
            # Keep the default overlap proportionate when the caller picks a small size.
            overlap = min(self._default_chunk_overlap, size // 5)
        else:
            overlap = chunk_overlap
        return size, overlap

    @staticmethod
    def _chunk(
        text: str, strategy: ChunkingStrategy, size: int, overlap: int
    ) -> list[str]:
        try:
            return chunk_text(text, strategy, size, overlap)
        except ValueError as exc:  # our own chunker's messages are safe to show
            raise InvalidChunkingParametersError(str(exc)) from exc

    def _ingest_locked(
        self,
        document_id: uuid.UUID,
        upload: ValidatedUpload,
        content: bytes,
        content_hash: str,
        strategy: ChunkingStrategy,
        size: int,
        overlap: int,
        chunks: list[str],
    ) -> IngestionResult:
        existing = self._repository.get(document_id)
        if existing is not None and existing.status == DocumentStatus.COMPLETED:
            logger.info("document %s already ingested; reusing", document_id)
            return IngestionResult(document=existing, created=False)

        if existing is None:
            document = self._repository.create_processing(
                document_id=document_id,
                filename=upload.filename,
                file_type=upload.file_type.value,
                file_size=len(content),
                content_hash=content_hash,
                chunking_strategy=strategy.value,
                chunk_size=size,
                chunk_overlap=overlap,
            )
        else:  # a failed or interrupted earlier attempt: try again
            document = existing
            self._repository.mark_processing(document, upload.filename)

        self._embed_and_store(document, upload, strategy, chunks)
        return IngestionResult(document=document, created=True)

    def _embed_and_store(
        self,
        document: Document,
        upload: ValidatedUpload,
        strategy: ChunkingStrategy,
        chunks: list[str],
    ) -> None:
        document_id = document.id  # read now; the object may be expired after a rollback
        started = time.perf_counter()
        # Broad except on purpose: whatever fails (model, Qdrant, Postgres), the
        # result must be the same - compensate, then report one clean error.
        try:
            vectors = self._embedder.embed_texts(chunks)
            point_ids = self._vector_store.upsert_chunks(
                document_id, upload.filename, strategy.value, chunks, vectors
            )
            records = [
                ChunkRecord(chunk_index=index, vector_id=point_id, char_count=len(text))
                for index, (point_id, text) in enumerate(zip(point_ids, chunks))
            ]
            self._repository.complete(document, records)  # the commit point
        except Exception as exc:
            logger.exception("ingestion failed for document %s", document_id)
            self._compensate(document, document_id)
            raise IngestionFailedError(
                "Document processing failed. It is safe to retry the upload."
            ) from exc
        logger.info(
            "ingested document %s: %d chunks in %.1fs",
            document_id,
            len(chunks),
            time.perf_counter() - started,
        )

    def _compensate(self, document: Document, document_id: uuid.UUID) -> None:
        """Undo partial work. Each step is attempted even if the previous one fails,
        and neither may mask the original error."""
        try:
            self._vector_store.delete_document(document_id)
        except Exception:
            logger.exception("could not remove vectors for document %s", document_id)
        try:
            self._repository.mark_failed(document)
        except Exception:
            logger.exception("could not mark document %s as failed", document_id)

    @staticmethod
    def _release(lock: Lock) -> None:
        try:
            lock.release()
        except LockError:
            logger.warning("ingestion lock had already expired before release")