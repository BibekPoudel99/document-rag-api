import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Message = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "What are the main approaches to machine learning?"},
            ]
        }
    )

    message: Message = Field(description="The user's message (1-2000 characters).")
    conversation_id: uuid.UUID | None = Field(
        default=None,
        description="Omit to start a new conversation; send the returned ID to continue it.",
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        max_length=20,
        description="Restrict retrieval to these documents. Omit to search all documents.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="How many chunks to retrieve. Omit for the server default.",
    )


class SourceOut(BaseModel):
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    response: str
    sources: list[SourceOut]
    grounded: bool = Field(
        description="False when no relevant document text was found and the model was not asked."
    )
    search_query: str = Field(
        description="The query actually used for retrieval, after resolving follow-up references."
    )