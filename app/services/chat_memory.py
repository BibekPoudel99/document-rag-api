"""Conversation memory stored in Redis.

One Redis LIST per conversation (key: chat:<conversation_id>:messages). Each
element is a JSON object: {"role", "content", "created_at"}.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from redis import Redis

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]
_ROLES: tuple[str, ...] = ("user", "assistant")


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    created_at: str  # ISO-8601, UTC


def new_conversation_id() -> uuid.UUID:
    return uuid.uuid4()


class ChatMemory:
    def __init__(self, redis_client: Redis, *, max_messages: int, ttl_seconds: int) -> None:
        if max_messages < 2 or max_messages % 2 != 0:
            raise ValueError("max_messages must be an even number >= 2 (user + assistant pairs)")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._redis = redis_client
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def key(conversation_id: uuid.UUID) -> str:
        # The ID is always a UUID (validated at the API), so a client can never
        # craft a key that lands in another namespace such as "booking:...".
        return f"chat:{conversation_id}:messages"

    def get_history(
        self, conversation_id: uuid.UUID, limit: int | None = None
    ) -> list[ChatMessage]:
        """Oldest-to-newest messages; `limit` keeps only the most recent ones.
        An unknown or expired conversation simply has an empty history."""
        if limit is not None and limit <= 0:
            return []
        start = 0 if limit is None else -limit
        raw_items = self._redis.lrange(self.key(conversation_id), start, -1)
        decoded = (self._decode(item) for item in raw_items)
        return [message for message in decoded if message is not None]

    def append_exchange(
        self, conversation_id: uuid.UUID, user_message: str, assistant_message: str
    ) -> None:
        """Store one completed turn. The three Redis commands run as one transaction,
        so the list can never be seen untrimmed or without an expiry."""
        key = self.key(conversation_id)
        now = datetime.now(UTC).isoformat()
        entries = [
            self._encode("user", user_message, now),
            self._encode("assistant", assistant_message, now),
        ]
        with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *entries)
            pipe.ltrim(key, -self._max_messages, -1)  # keep only the newest N
            pipe.expire(key, self._ttl_seconds)  # sliding expiry: every turn renews it
            pipe.execute()

    def clear(self, conversation_id: uuid.UUID) -> None:
        self._redis.delete(self.key(conversation_id))

    @staticmethod
    def _encode(role: Role, content: str, created_at: str) -> str:
        payload = {"role": role, "content": content, "created_at": created_at}
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _decode(raw: str) -> ChatMessage | None:
        try:
            data = json.loads(raw)
            role = data["role"]
            content = data["content"]
            created_at = data["created_at"]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("skipping unreadable chat memory entry")
            return None
        if role not in _ROLES or not isinstance(content, str):
            logger.warning("skipping malformed chat memory entry")
            return None
        return ChatMessage(role=cast(Role, role), content=content, created_at=str(created_at))