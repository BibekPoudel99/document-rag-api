import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import (
    DocumentBusyError,
    EmptyFileError,
    IngestionFailedError,
    InvalidChunkingParametersError,
    UnsupportedFileTypeError,
)
from app.db.models import Document, DocumentChunk, DocumentStatus
from app.services.chunking import ChunkingStrategy
from app.services.ingestion import ingestion_lock_name, make_document_id
from app.services.vector_store import make_point_id
from tests.fakes import IngestionHarness
from tests.pdf_factory import make_text_pdf

RECURSIVE = ChunkingStrategy.RECURSIVE
FIXED = ChunkingStrategy.FIXED

TEXT = "\n\n".join(
    f"Paragraph {i}. " + "lorem ipsum dolor " * 20 for i in range(10)
).encode("utf-8")


def _documents(session: Session) -> list[Document]:
    return list(session.scalars(select(Document)))


def _chunks(session: Session) -> list[DocumentChunk]:
    return sorted(session.scalars(select(DocumentChunk)), key=lambda c: c.chunk_index)


# ---------- the happy path ----------

def test_ingest_txt_persists_metadata_chunks_and_vectors(harness: IngestionHarness) -> None:
    result = harness.service.ingest("notes.txt", TEXT, RECURSIVE, 300, 30)
    document = result.document

    assert result.created is True
    assert document.status == DocumentStatus.COMPLETED
    assert document.num_chunks > 1
    assert document.file_type == "txt"
    assert document.chunking_strategy == "recursive"
    assert (document.chunk_size, document.chunk_overlap) == (300, 30)
    assert document.content_hash == hashlib.sha256(TEXT).hexdigest()
    assert document.file_size == len(TEXT)

    rows = _chunks(harness.session)
    assert len(rows) == document.num_chunks
    assert [row.vector_id for row in rows] == [
        make_point_id(document.id, index) for index in range(document.num_chunks)
    ]
    assert len(harness.vector_store.documents[document.id]) == document.num_chunks


def test_ingest_pdf(harness: IngestionHarness) -> None:
    pdf = make_text_pdf("Vector databases store embeddings for similarity search")
    result = harness.service.ingest("report.pdf", pdf, RECURSIVE)
    assert result.document.file_type == "pdf"
    assert result.document.num_chunks == 1


def test_each_strategy_gets_its_own_document(harness: IngestionHarness) -> None:
    fixed = harness.service.ingest("a.txt", TEXT, FIXED, 300, 30)
    recursive = harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)
    assert fixed.document.id != recursive.document.id
    strategies = {doc.chunking_strategy for doc in _documents(harness.session)}
    assert strategies == {"fixed", "recursive"}


# ---------- idempotency ----------

def test_identical_upload_is_reused_without_reprocessing(harness: IngestionHarness) -> None:
    first = harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)
    second = harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    assert first.created is True
    assert second.created is False
    assert second.document.id == first.document.id
    assert harness.embedder.calls == 1
    assert harness.vector_store.upsert_calls == 1
    assert len(_documents(harness.session)) == 1


def test_same_content_with_different_chunk_settings_is_a_new_document(
    harness: IngestionHarness,
) -> None:
    first = harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)
    second = harness.service.ingest("a.txt", TEXT, RECURSIVE, 400, 30)
    assert first.document.id != second.document.id
    assert len(_documents(harness.session)) == 2


def test_document_id_is_deterministic_and_parameter_sensitive() -> None:
    base = make_document_id("abc", FIXED, 300, 30)
    assert base == make_document_id("abc", FIXED, 300, 30)
    assert base != make_document_id("abd", FIXED, 300, 30)
    assert base != make_document_id("abc", RECURSIVE, 300, 30)
    assert base != make_document_id("abc", FIXED, 400, 30)
    assert base != make_document_id("abc", FIXED, 300, 40)


# ---------- failure handling and consistency ----------

def test_embedding_failure_leaves_nothing_behind(harness: IngestionHarness) -> None:
    harness.embedder.fail = True
    with pytest.raises(IngestionFailedError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    [document] = _documents(harness.session)
    assert document.status == DocumentStatus.FAILED
    assert document.num_chunks == 0
    assert _chunks(harness.session) == []
    assert harness.vector_store.documents == {}
    assert harness.redis.held == set()  # the lock was released


def test_vector_store_failure_removes_partial_vectors(harness: IngestionHarness) -> None:
    harness.vector_store.fail_on_upsert = True
    with pytest.raises(IngestionFailedError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    [document] = _documents(harness.session)
    assert document.status == DocumentStatus.FAILED
    assert harness.vector_store.documents == {}  # the partial write was cleaned up
    assert _chunks(harness.session) == []


def test_failed_document_can_be_retried(harness: IngestionHarness) -> None:
    harness.vector_store.fail_on_upsert = True
    with pytest.raises(IngestionFailedError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    harness.vector_store.fail_on_upsert = False
    result = harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    assert result.created is True
    assert result.document.status == DocumentStatus.COMPLETED
    assert len(_documents(harness.session)) == 1  # same document, not a duplicate
    assert len(_chunks(harness.session)) == result.document.num_chunks
    assert harness.embedder.calls == 2
    assert result.document.id in harness.vector_store.documents


def test_cleanup_failure_does_not_mask_the_original_error(harness: IngestionHarness) -> None:
    harness.vector_store.fail_on_upsert = True
    harness.vector_store.fail_on_delete = True
    with pytest.raises(IngestionFailedError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    [document] = _documents(harness.session)
    assert document.status == DocumentStatus.FAILED
    assert harness.redis.held == set()


# ---------- concurrency lock ----------

def test_busy_document_is_rejected_without_side_effects(harness: IngestionHarness) -> None:
    document_id = make_document_id(hashlib.sha256(TEXT).hexdigest(), RECURSIVE, 300, 30)
    harness.redis.held.add(ingestion_lock_name(document_id))  # someone else is working

    with pytest.raises(DocumentBusyError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)

    assert _documents(harness.session) == []
    assert harness.embedder.calls == 0


def test_lock_is_released_after_success(harness: IngestionHarness) -> None:
    harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)
    assert harness.redis.held == set()


def test_redis_failure_is_reported_cleanly(harness: IngestionHarness) -> None:
    harness.redis.unavailable = True
    with pytest.raises(IngestionFailedError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 300, 30)
    assert _documents(harness.session) == []


# ---------- input problems are caught before any work ----------

def test_overlap_not_smaller_than_size_is_rejected_before_any_work(
    harness: IngestionHarness,
) -> None:
    with pytest.raises(InvalidChunkingParametersError):
        harness.service.ingest("a.txt", TEXT, FIXED, 300, 300)
    assert _documents(harness.session) == []
    assert harness.embedder.calls == 0


def test_recursive_overlap_without_room_is_rejected(harness: IngestionHarness) -> None:
    with pytest.raises(InvalidChunkingParametersError):
        harness.service.ingest("a.txt", TEXT, RECURSIVE, 100, 99)


def test_default_overlap_adapts_to_small_chunk_size(harness: IngestionHarness) -> None:
    result = harness.service.ingest("a.txt", TEXT, RECURSIVE, chunk_size=100)
    assert result.document.chunk_overlap == 20  # min(default 100, 100 // 5)
    assert result.document.status == DocumentStatus.COMPLETED


def test_invalid_uploads_have_no_side_effects(harness: IngestionHarness) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        harness.service.ingest("a.docx", TEXT, RECURSIVE)
    with pytest.raises(EmptyFileError):
        harness.service.ingest("a.txt", b"", RECURSIVE)
    assert _documents(harness.session) == []
    assert harness.embedder.calls == 0