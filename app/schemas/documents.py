import uuid

from pydantic import BaseModel

from app.db.models import DocumentStatus
from app.services.chunking import ChunkingStrategy
from app.services.file_validation import FileType


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    file_type: FileType
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    status: DocumentStatus
    already_ingested: bool