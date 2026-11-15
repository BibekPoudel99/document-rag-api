"""Manual check of the whole ingestion pipeline against the REAL services.

Run from the project root:  python -m scripts.smoke_ingest
Uses the real 'documents' collection. Safe to run repeatedly: ingestion is idempotent.
"""

from pathlib import Path

from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk
from app.db.session import create_db_engine, create_session_factory, init_db
from app.repositories.documents import DocumentRepository
from app.services.chunking import ChunkingStrategy
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStore

SAMPLE = Path("samples/ml_overview.txt")
QUESTION = "What is the second approach to machine learning?"


def main() -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    init_db(engine)
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = EmbeddingService.from_model_name(
        settings.embedding_model, settings.embedding_batch_size
    )
    store = VectorStore(qdrant, settings.qdrant_collection, embedder.dimension)
    store.ensure_collection()

    with create_session_factory(engine)() as session:
        service = IngestionService(
            DocumentRepository(session),
            embedder,
            store,
            redis_client,
            max_upload_bytes=settings.max_upload_bytes,
            default_chunk_size=settings.default_chunk_size,
            default_chunk_overlap=settings.default_chunk_overlap,
        )
        content = SAMPLE.read_bytes()

        first = service.ingest(SAMPLE.name, content, ChunkingStrategy.RECURSIVE, 500, 50)
        print(f"1) recursive: id={first.document.id} chunks={first.document.num_chunks} "
              f"status={first.document.status} created={first.created}")

        again = service.ingest(SAMPLE.name, content, ChunkingStrategy.RECURSIVE, 500, 50)
        print(f"2) same upload again: id={again.document.id} created={again.created}  "
              f"(same id, created=False means it was reused)")

        fixed = service.ingest(SAMPLE.name, content, ChunkingStrategy.FIXED, 500, 50)
        print(f"3) fixed: id={fixed.document.id} chunks={fixed.document.num_chunks} "
              f"created={fixed.created}")

        documents = session.scalar(select(func.count()).select_from(Document))
        chunk_rows = session.scalar(select(func.count()).select_from(DocumentChunk))
        points = qdrant.count(collection_name=settings.qdrant_collection, exact=True).count
        print(f"\nPostgres: {documents} documents, {chunk_rows} chunk rows | Qdrant: {points} points")

        print(f"\nSearch (recursive document only): {QUESTION}")
        hits = store.search(
            embedder.embed_query(QUESTION), top_k=3, document_ids=[first.document.id]
        )
        for hit in hits:
            preview = hit.text[:70].replace("\n", " ")
            print(f"  {hit.score:.3f}  chunk {hit.chunk_index}: {preview}")


if __name__ == "__main__":
    main()