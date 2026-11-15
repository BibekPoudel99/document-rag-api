"""Database access for documents and their chunks. No business rules here."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, DocumentStatus


@dataclass(frozen=True)
class ChunkRecord:
    chunk_index: int
    vector_id: uuid.UUID
    char_count: int


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self._session.get(Document, document_id)

    def create_processing(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        file_type: str,
        file_size: int,
        content_hash: str,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        document = Document(
            id=document_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            content_hash=content_hash,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=0,
            status=DocumentStatus.PROCESSING.value,
        )
        self._session.add(document)
        self._session.commit()
        return document

    def mark_processing(self, document: Document, filename: str) -> None:
        """Re-open a failed (or interrupted) document for another attempt."""
        document.filename = filename
        document.num_chunks = 0
        document.status = DocumentStatus.PROCESSING.value
        self._session.commit()

    def complete(self, document: Document, chunks: Sequence[ChunkRecord]) -> None:
        """Insert all chunk rows and flip the status in ONE transaction."""
        self._session.add_all(
            [
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    vector_id=chunk.vector_id,
                    char_count=chunk.char_count,
                )
                for chunk in chunks
            ]
        )
        document.num_chunks = len(chunks)
        document.status = DocumentStatus.COMPLETED.value
        self._session.commit()

    def mark_failed(self, document: Document) -> None:
        # The session may be in a failed transaction; discard it first.
        self._session.rollback()
        document.num_chunks = 0
        document.status = DocumentStatus.FAILED.value
        self._session.commit()