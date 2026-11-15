import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from app.services.vector_store import (
    RetrievedChunk,
    VectorDimensionMismatchError,
    VectorStore,
    make_point_id,
)

DOC = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOC = uuid.UUID("22222222-2222-2222-2222-222222222222")


class FakeQdrant:
    """Records calls; stands in for QdrantClient so no server is needed."""

    def __init__(
        self,
        existing: models.VectorParams | None = None,
        points: list[Any] | None = None,
    ) -> None:
        self.existing = existing
        self.points_to_return = points or []
        self.created: list[dict[str, Any]] = []
        self.indexes: list[str] = []
        self.upserts: list[list[models.PointStruct]] = []
        self.deletes: list[Any] = []
        self.queries: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.existing is not None

    def get_collection(self, collection_name: str) -> Any:
        params = SimpleNamespace(vectors=self.existing)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.created.append({"name": collection_name, "config": vectors_config})

    def create_payload_index(self, collection_name: str, field_name: str, field_schema: Any) -> None:
        self.indexes.append(field_name)

    def upsert(self, collection_name: str, points: list[models.PointStruct], wait: bool) -> None:
        assert wait is True
        self.upserts.append(list(points))

    def delete(self, collection_name: str, points_selector: Any, wait: bool) -> None:
        assert wait is True
        self.deletes.append(points_selector)

    def query_points(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return SimpleNamespace(points=self.points_to_return)


def make_store(client: FakeQdrant, dimension: int = 384) -> VectorStore:
    return VectorStore(client, collection="docs", dimension=dimension)  # type: ignore[arg-type]


# ---------- point IDs ----------

def test_point_id_is_deterministic() -> None:
    assert make_point_id(DOC, 3) == make_point_id(DOC, 3)


def test_point_id_differs_by_chunk_and_by_document() -> None:
    assert make_point_id(DOC, 0) != make_point_id(DOC, 1)
    assert make_point_id(DOC, 0) != make_point_id(OTHER_DOC, 0)


# ---------- collection setup ----------

def test_ensure_collection_creates_cosine_collection_and_document_index() -> None:
    client = FakeQdrant(existing=None)
    make_store(client).ensure_collection()
    config = client.created[0]["config"]
    assert client.created[0]["name"] == "docs"
    assert config.size == 384
    assert config.distance == models.Distance.COSINE
    assert client.indexes == ["document_id"]


def test_ensure_collection_leaves_matching_collection_alone() -> None:
    client = FakeQdrant(existing=models.VectorParams(size=384, distance=models.Distance.COSINE))
    make_store(client).ensure_collection()
    assert client.created == []


def test_ensure_collection_rejects_wrong_dimension() -> None:
    client = FakeQdrant(existing=models.VectorParams(size=768, distance=models.Distance.COSINE))
    with pytest.raises(VectorDimensionMismatchError):
        make_store(client).ensure_collection()


def test_ensure_collection_rejects_wrong_distance() -> None:
    client = FakeQdrant(existing=models.VectorParams(size=384, distance=models.Distance.DOT))
    with pytest.raises(VectorDimensionMismatchError):
        make_store(client).ensure_collection()


# ---------- upsert ----------

def test_upsert_builds_payload_and_returns_deterministic_ids() -> None:
    client = FakeQdrant()
    ids = make_store(client).upsert_chunks(
        DOC, "notes.txt", "fixed", ["first", "second"], [[0.1, 0.2], [0.3, 0.4]]
    )
    assert ids == [make_point_id(DOC, 0), make_point_id(DOC, 1)]
    points = client.upserts[0]
    assert points[1].id == str(ids[1])
    assert points[1].vector == [0.3, 0.4]
    assert points[1].payload == {
        "document_id": str(DOC),
        "filename": "notes.txt",
        "chunk_index": 1,
        "text": "second",
        "chunking_strategy": "fixed",
    }


def test_upsert_sends_large_documents_in_batches() -> None:
    client = FakeQdrant()
    chunks = [f"chunk {i}" for i in range(250)]
    make_store(client).upsert_chunks(DOC, "big.txt", "fixed", chunks, [[0.0]] * 250)
    assert [len(batch) for batch in client.upserts] == [100, 100, 50]


def test_upsert_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        make_store(FakeQdrant()).upsert_chunks(DOC, "a.txt", "fixed", ["a", "b"], [[0.1]])


# ---------- search ----------

def _scored_point(score: float, index: int, text: str) -> Any:
    payload = {
        "document_id": str(DOC),
        "filename": "notes.txt",
        "chunk_index": index,
        "text": text,
        "chunking_strategy": "fixed",
    }
    return SimpleNamespace(score=score, payload=payload)


def test_search_maps_results_and_passes_parameters() -> None:
    client = FakeQdrant(points=[_scored_point(0.9, 4, "relevant text")])
    results = make_store(client).search([0.1, 0.2], top_k=3, score_threshold=0.4)
    assert results == [RetrievedChunk(str(DOC), "notes.txt", 4, "relevant text", 0.9)]
    query = client.queries[0]
    assert query["limit"] == 3
    assert query["score_threshold"] == 0.4
    assert query["query"] == [0.1, 0.2]
    assert query["query_filter"] is None


def test_search_filters_by_document_ids() -> None:
    client = FakeQdrant()
    make_store(client).search([0.1], top_k=5, document_ids=[DOC, OTHER_DOC])
    condition = client.queries[0]["query_filter"].must[0]
    assert condition.key == "document_id"
    assert condition.match.any == [str(DOC), str(OTHER_DOC)]


def test_search_treats_empty_document_list_as_no_filter() -> None:
    client = FakeQdrant()
    make_store(client).search([0.1], top_k=5, document_ids=[])
    assert client.queries[0]["query_filter"] is None


# ---------- delete ----------

def test_delete_document_selects_by_document_id() -> None:
    client = FakeQdrant()
    make_store(client).delete_document(DOC)
    condition = client.deletes[0].filter.must[0]
    assert condition.key == "document_id"
    assert condition.match.any == [str(DOC)]