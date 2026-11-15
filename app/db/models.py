import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    chunking_strategy: Mapped[str] = mapped_column(String(32))
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    num_chunks: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]
    vector_id: Mapped[uuid.UUID]
    char_count: Mapped[int]
    token_count: Mapped[int | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )