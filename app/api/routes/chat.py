from fastapi import APIRouter

from app.api.deps import ChatServiceDep
from app.schemas.chat import ChatRequest, ChatResponse, SourceOut

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message in a conversation",
    responses={
        409: {"description": "Another message in this conversation is still being processed."},
        422: {"description": "Invalid request."},
        503: {"description": "The language model or conversation memory is unavailable."},
    },
)
def chat(body: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    turn = service.respond(
        message=body.message,
        conversation_id=body.conversation_id,
        document_ids=body.document_ids,
        top_k=body.top_k,
    )
    answer = turn.answer
    return ChatResponse(
        conversation_id=turn.conversation_id,
        response=answer.text,
        sources=[
            SourceOut(
                document_id=source.document_id,
                filename=source.filename,
                chunk_index=source.chunk_index,
                score=source.score,
                snippet=source.snippet,
            )
            for source in answer.sources
        ],
        grounded=answer.grounded,
        search_query=answer.search_query,
    )