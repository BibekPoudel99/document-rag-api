import hashlib
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
from app.services.chunking import ChunkingStrategy
from app.services.ingestion import ingestion_lock_name, make_document_id
from tests.fakes import IngestionHarness, make_harness
from tests.pdf_factory import make_text_pdf

URL = "/api/v1/documents/upload"

TEXT = "\n\n".join(
    f"Paragraph {i}. " + "lorem ipsum dolor " * 20 for i in range(10)
).encode("utf-8")


@pytest.fixture
def client(harness: IngestionHarness) -> Iterator[TestClient]:
    app.dependency_overrides[deps.get_ingestion_service] = lambda: harness.service
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _upload(
    client: TestClient,
    name: str = "notes.txt",
    content: bytes = TEXT,
    **form: str,
):
    return client.post(
        URL,
        files={"file": (name, content, "application/octet-stream")},
        data=form,
    )


def test_upload_txt_returns_201_with_expected_fields(client: TestClient) -> None:
    response = _upload(client, chunking_strategy="fixed", chunk_size="300", chunk_overlap="30")

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["document_id"])  # a valid UUID
    assert body["filename"] == "notes.txt"
    assert body["file_type"] == "txt"
    assert body["chunking_strategy"] == "fixed"
    assert (body["chunk_size"], body["chunk_overlap"]) == (300, 30)
    assert body["num_chunks"] > 1
    assert body["status"] == "completed"
    assert body["already_ingested"] is False


def test_defaults_are_used_when_options_are_omitted(client: TestClient) -> None:
    body = _upload(client).json()
    assert body["chunking_strategy"] == "recursive"
    assert (body["chunk_size"], body["chunk_overlap"]) == (800, 100)


def test_repeating_an_upload_returns_200_and_the_same_document(client: TestClient) -> None:
    first = _upload(client, chunk_size="300", chunk_overlap="30")
    second = _upload(client, chunk_size="300", chunk_overlap="30")

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["already_ingested"] is True
    assert second.json()["document_id"] == first.json()["document_id"]


def test_upload_pdf(client: TestClient) -> None:
    pdf = make_text_pdf("Vector databases store embeddings for similarity search")
    response = _upload(client, "report.pdf", pdf)
    assert response.status_code == 201
    assert response.json()["file_type"] == "pdf"


def test_unsupported_file_type_returns_415(client: TestClient) -> None:
    response = _upload(client, "notes.docx")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_empty_file_returns_422(client: TestClient) -> None:
    response = _upload(client, content=b"")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_file"


def test_invalid_pdf_returns_422(client: TestClient) -> None:
    response = _upload(client, "broken.pdf", b"this is not really a pdf")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_file"


def test_missing_file_returns_422(client: TestClient) -> None:
    response = client.post(URL, data={"chunking_strategy": "fixed"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "file" in response.json()["error"]["message"]


def test_unknown_strategy_returns_422(client: TestClient) -> None:
    response = _upload(client, chunking_strategy="semantic")
    assert response.status_code == 422
    assert "chunking_strategy" in response.json()["error"]["message"]


def test_chunk_size_out_of_range_returns_422(client: TestClient) -> None:
    response = _upload(client, chunk_size="10")
    assert response.status_code == 422
    assert "chunk_size" in response.json()["error"]["message"]


def test_overlap_not_smaller_than_size_returns_422(client: TestClient) -> None:
    response = _upload(client, chunk_size="200", chunk_overlap="200")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_chunking_parameters"


def test_file_over_the_size_limit_returns_413(
    client: TestClient, harness: IngestionHarness
) -> None:
    small = make_harness(harness.session, max_upload_bytes=10)
    app.dependency_overrides[deps.get_ingestion_service] = lambda: small.service

    response = _upload(client, content=b"x" * 1000)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_ingestion_failure_returns_503_without_leaking_details(
    client: TestClient, harness: IngestionHarness
) -> None:
    harness.vector_store.fail_on_upsert = True
    response = _upload(client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ingestion_failed"
    assert "qdrant" not in response.text.lower()


def test_document_being_processed_returns_409(
    client: TestClient, harness: IngestionHarness
) -> None:
    document_id = make_document_id(
        hashlib.sha256(TEXT).hexdigest(), ChunkingStrategy.RECURSIVE, 300, 30
    )
    harness.redis.held.add(ingestion_lock_name(document_id))

    response = _upload(client, chunk_size="300", chunk_overlap="30")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "document_busy"