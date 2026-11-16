"""Chat orchestration: one conversation turn = load memory -> answer -> save memory.

Phase 5 (interview booking) plugs into respond(): it decides, per message,
whether this is a document question or part of a booking conversation.
"""

import logging
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import LockError, RedisError

from app.core.exceptions import ChatMemoryUnavailableError, ConversationBusyError
from app.services.chat_memory import ChatMemory, ChatMessage, new_conversation_id
from app.services.rag import RagAnswer, RagService

logger = logging.getLogger(__name__)

# Longer than the worst case of two LLM calls (rewrite + answer) at the client timeout.
_LOCK_SECONDS = 300


def conversation_lock_name(conversation_id: uuid.UUID) -> str:
    return f"chat:{conversation_id}:lock"


@dataclass(frozen=True)
class ChatTurn:
    conversation_id: uuid.UUID
    answer: RagAnswer


class ChatService:
    def __init__(self, memory: ChatMemory, rag: RagService, redis_client: Redis) -> None:
        self._memory = memory
        self._rag = rag
        self._redis = redis_client

    def respond(
        self,
        message: str,
        conversation_id: uuid.UUID | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
        top_k: int | None = None,
    ) -> ChatTurn:
        # An unknown or expired ID is simply a conversation with no history yet.
        conversation = new_conversation_id() if conversation_id is None else conversation_id

        with self._turn_lock(conversation):
            history = self._load_history(conversation)
            answer = self._rag.answer(message, history, document_ids, top_k)
            self._save_turn(conversation, message, answer.text)
        return ChatTurn(conversation_id=conversation, answer=answer)

    @contextmanager
    def _turn_lock(self, conversation_id: uuid.UUID) -> Iterator[None]:
        """One message at a time per conversation, so two requests can't interleave
        their reads and writes of the same history."""
        lock = self._redis.lock(
            conversation_lock_name(conversation_id), timeout=_LOCK_SECONDS, blocking=False
        )
        try:
            acquired = lock.acquire(blocking=False)
        except RedisError as exc:
            logger.error("could not reach Redis for the conversation lock: %s", exc)
            raise ChatMemoryUnavailableError("Conversation memory is unavailable.") from exc
        if not acquired:
            raise ConversationBusyError(
                "A previous message in this conversation is still being processed. "
                "Please wait for it to finish."
            )
        try:
            yield
        finally:
            try:
                lock.release()
            except LockError:
                logger.warning("conversation lock had already expired before release")

    def _load_history(self, conversation_id: uuid.UUID) -> list[ChatMessage]:
        try:
            return self._memory.get_history(conversation_id)
        except RedisError as exc:
            logger.error("could not read conversation memory: %s", exc)
            raise ChatMemoryUnavailableError("Conversation memory is unavailable.") from exc

    def _save_turn(self, conversation_id: uuid.UUID, message: str, reply: str) -> None:
        # If saving fails we report an error rather than a success that silently
        # loses context: the next message would be answered without this turn.
        try:
            self._memory.append_exchange(conversation_id, message, reply)
        except RedisError as exc:
            logger.error("could not save conversation memory: %s", exc)
            raise ChatMemoryUnavailableError("Conversation memory is unavailable.") from exc