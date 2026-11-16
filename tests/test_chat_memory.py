import json
import uuid
from datetime import datetime

import fakeredis
import pytest

from app.services.chat_memory import ChatMemory, new_conversation_id


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def memory(redis_client: fakeredis.FakeRedis) -> ChatMemory:
    return ChatMemory(redis_client, max_messages=6, ttl_seconds=3600)


def test_new_conversation_has_an_empty_history(memory: ChatMemory) -> None:
    assert memory.get_history(new_conversation_id()) == []


def test_exchange_round_trip_keeps_order_and_roles(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "What is overfitting?", "It is when a model memorises noise.")

    history = memory.get_history(cid)
    assert [(m.role, m.content) for m in history] == [
        ("user", "What is overfitting?"),
        ("assistant", "It is when a model memorises noise."),
    ]


def test_non_ascii_text_survives_storage(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "नमस्ते, café?", "Résumé ✓")
    assert [m.content for m in memory.get_history(cid)] == ["नमस्ते, café?", "Résumé ✓"]


def test_multiple_turns_accumulate_in_order(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    memory.append_exchange(cid, "q2", "a2")
    assert [m.content for m in memory.get_history(cid)] == ["q1", "a1", "q2", "a2"]


def test_limit_returns_only_the_most_recent_messages(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    memory.append_exchange(cid, "q2", "a2")
    assert [m.content for m in memory.get_history(cid, limit=2)] == ["q2", "a2"]


def test_zero_or_negative_limit_returns_nothing(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    assert memory.get_history(cid, limit=0) == []
    assert memory.get_history(cid, limit=-3) == []


def test_oldest_messages_are_dropped_beyond_the_limit(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    for n in range(1, 5):  # 4 exchanges = 8 messages, but the limit is 6
        memory.append_exchange(cid, f"q{n}", f"a{n}")
    assert [m.content for m in memory.get_history(cid)] == [
        "q2", "a2", "q3", "a3", "q4", "a4",
    ]


def test_conversations_are_isolated(memory: ChatMemory) -> None:
    first, second = new_conversation_id(), new_conversation_id()
    memory.append_exchange(first, "only in first", "reply to first")
    assert memory.get_history(second) == []
    memory.append_exchange(second, "only in second", "reply to second")
    assert [m.content for m in memory.get_history(first)] == ["only in first", "reply to first"]


def test_expiry_is_set_and_renewed_on_every_turn(
    memory: ChatMemory, redis_client: fakeredis.FakeRedis
) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    assert 0 < redis_client.ttl(ChatMemory.key(cid)) <= 3600

    redis_client.expire(ChatMemory.key(cid), 5)  # pretend it is about to expire
    memory.append_exchange(cid, "q2", "a2")
    assert redis_client.ttl(ChatMemory.key(cid)) > 100  # renewed


def test_clear_removes_the_conversation(
    memory: ChatMemory, redis_client: fakeredis.FakeRedis
) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    memory.clear(cid)
    assert memory.get_history(cid) == []
    assert redis_client.exists(ChatMemory.key(cid)) == 0


def test_unreadable_entries_are_skipped_not_fatal(
    memory: ChatMemory, redis_client: fakeredis.FakeRedis
) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    bad_role = json.dumps({"role": "system", "content": "x", "created_at": "t"})
    redis_client.rpush(ChatMemory.key(cid), "not json at all", bad_role, "[1, 2]")

    assert [m.content for m in memory.get_history(cid)] == ["q1", "a1"]


def test_timestamps_are_timezone_aware_iso_strings(memory: ChatMemory) -> None:
    cid = new_conversation_id()
    memory.append_exchange(cid, "q1", "a1")
    for message in memory.get_history(cid):
        assert datetime.fromisoformat(message.created_at).tzinfo is not None


def test_key_is_namespaced_by_conversation() -> None:
    cid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert ChatMemory.key(cid) == "chat:12345678-1234-5678-1234-567812345678:messages"


@pytest.mark.parametrize(
    ("max_messages", "ttl_seconds"), [(0, 60), (1, 60), (3, 60), (4, 0)]
)
def test_invalid_settings_are_rejected(
    redis_client: fakeredis.FakeRedis, max_messages: int, ttl_seconds: int
) -> None:
    with pytest.raises(ValueError):
        ChatMemory(redis_client, max_messages=max_messages, ttl_seconds=ttl_seconds)