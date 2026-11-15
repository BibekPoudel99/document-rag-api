from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile, status

from app.api.deps import IngestionServiceDep
from app.db.models import DocumentStatus
from app.schemas.documents import DocumentUploadResponse
from app.services.chunking import ChunkingStrategy
from app.services.file_validation import FileType

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_ERROR_RESPONSES: dict[int | str, dict[str, str]] = {
    status.HTTP_409_CONFLICT: {"description": "The same document is already being processed."},
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "File exceeds the upload size limit."},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Only .pdf and .txt files are supported."},
    status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid file or invalid parameters."},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Processing failed; safe to retry."},
}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    responses={
        status.HTTP_200_OK: {
            "model": DocumentUploadResponse,
            "description": "An identical document was already ingested; nothing was reprocessed.",
        },
        **_ERROR_RESPONSES,
    },
)
def upload_document(
    response: Response,
    service: IngestionServiceDep,
    file: Annotated[UploadFile, File(description="A .pdf or .txt file.")],
    chunking_strategy: Annotated[
        ChunkingStrategy, Form(description="How to split the text into chunks.")
    ] = ChunkingStrategy.RECURSIVE,
    chunk_size: Annotated[
        int | None,
        Form(ge=100, le=2000, description="Characters per chunk. Omit for the server default."),
    ] = None,
    chunk_overlap: Annotated[
        int | None,
        Form(ge=0, le=500, description="Characters repeated between chunks. Omit for the default."),
    ] = None,
) -> DocumentUploadResponse:
    # Read one byte past the limit: enough to detect "too large" without loading
    # an unbounded upload into memory.
    content = file.file.read(service.max_upload_bytes + 1)

    result = service.ingest(
        filename=file.filename,
        content=content,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not result.created:
        response.status_code = status.HTTP_200_OK

    document = result.document
    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        file_type=FileType(document.file_type),
        chunking_strategy=ChunkingStrategy(document.chunking_strategy),
        chunk_size=document.chunk_size,
        chunk_overlap=document.chunk_overlap,
        num_chunks=document.num_chunks,
        status=DocumentStatus(document.status),
        already_ingested=not result.created,
    )