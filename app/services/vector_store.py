"""Qdrant access: collection setup, upsert, search, delete. No business logic."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

# Fixed namespace so point IDs are stable across runs and machines.
_POINT_NAMESPACE = uuid.UUID("6f0d9c52-3a1e-4c57-9d0b-2b8e7a4f1c10")
_UPSERT_BATCH_SIZE = 100


class VectorDimensionMismatchError(RuntimeError):
    """The existing collection does not match the embedding model's configuration."""


def make_point_id(document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    """Deterministic ID: re-ingesting the same chunk overwrites instead of duplicating."""
    return uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{chunk_index}")


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


def _document_filter(document_ids: Sequence[uuid.UUID]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=[str(d) for d in document_ids]),
            )
        ]
    )


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        """Create the collection if missing; refuse to run against a mismatched one."""
        if self._client.collection_exists(self._collection):
            self._check_existing_collection()
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )
        # Speeds up "only these documents" filters and delete-by-document.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def _check_existing_collection(self) -> None:
        vectors = self._client.get_collection(self._collection).config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorDimensionMismatchError(
                f"Collection '{self._collection}' uses named vectors, which are not supported."
            )
        if vectors.size != self._dimension:
            raise VectorDimensionMismatchError(
                f"Collection '{self._collection}' has dimension {vectors.size}, "
                f"but the embedding model produces {self._dimension}."
            )
        if vectors.distance != models.Distance.COSINE:
            raise VectorDimensionMismatchError(
                f"Collection '{self._collection}' does not use cosine distance."
            )

    def upsert_chunks(
        self,
        document_id: uuid.UUID,
        filename: str,
        chunking_strategy: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> list[uuid.UUID]:
        """Store chunk vectors with their text; returns the point IDs in chunk order."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        point_ids = [make_point_id(document_id, index) for index in range(len(chunks))]
        points = [
            models.PointStruct(
                id=str(point_id),
                vector=list(vector),
                payload={
                    "document_id": str(document_id),
                    "filename": filename,
                    "chunk_index": index,
                    "text": text,
                    "chunking_strategy": chunking_strategy,
                },
            )
            for index, (point_id, vector, text) in enumerate(
                zip(point_ids, vectors, chunks)
            )
        ]
        # Batches keep each request small; wait=True returns only once the points
        # are durably stored, so later searches (and cleanup) see them.
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
            )
        return point_ids

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks by cosine similarity. No/empty document_ids = search everything."""
        response = self._client.query_points(
            collection_name=self._collection,
            query=list(query_vector),
            query_filter=_document_filter(document_ids) if document_ids else None,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    document_id=str(payload["document_id"]),
                    filename=str(payload["filename"]),
                    chunk_index=int(payload["chunk_index"]),
                    text=str(payload["text"]),
                    score=float(point.score),
                )
            )
        return results

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every vector belonging to a document (used for failure cleanup)."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=_document_filter([document_id])),
            wait=True,
        )