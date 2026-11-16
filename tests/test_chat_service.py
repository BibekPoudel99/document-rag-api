import uuid

import fakeredis
import pytest
import redis.exceptions as redis_errors

from app.core.exceptions import (
    ChatMemoryUnavailableError,
    ConversationBusyError,
    LLMUnavailableError,
)
from app.services.chat import conversation_lock_name
from app.services.rag import NO_ANSWER_TEXT
from tests.fakes import DEFAULT_DOC_ID, make_chat_harness, make_chunk

RELEVANT = [make_chunk("Unsupervised learning finds structure in unlabelled data.", 2, 0.8)]


class UnreadableRedis:
    """Memory backend whose reads fail."""

    def lrange(self, *args: object, **kwargs: object) -> list[str]:
        raise redis_errors.ConnectionError("redis is down")


class UnwritableRedis(fakeredis.FakeRedis):
    """Memory backend that reads fine but cannot write."""

    def pipeline(self, *args: object, **kwargs: object) -> object:
        raise redis_errors.ConnectionError("redis is down")


def _contents(harness, conversation_id: uuid.UUID) -> list[str]:
    return [m.content for m in harness.memory.get_history(conversation_id)]


def test_first_message_starts_a_conversation_and_is_remembered() -> None:
    h = make_chat_harness(RELEVANT, ["Learning from data."])

    turn = h.service.respond("What is ML?")

    assert isinstance(turn.conversation_id, uuid.UUID)
    assert turn.answer.text == "Learning from data."
    assert _contents(h, turn.conversation_id) == ["What is ML?", "Learning from data."]


def test_second_message_uses_the_stored_history_to_resolve_a_follow_up() -> None:
    h = make_chat_harness(RELEVANT, ["First answer", "rewritten query", "Second answer"])

    first = h.service.respond("Q1")
    second = h.service.respond("Q2", conversation_id=first.conversation_id)

    assert second.conversation_id == first.conversation_id
    assert h.embedder.queries == ["Q1", "rewritten query"]  # follow-up was rewritten
    assert _contents(h, first.conversation_id) == ["Q1", "First answer", "Q2", "Second answer"]


def test_conversations_do_not_share_history() -> None:
    h = make_chat_harness(RELEVANT)

    first = h.service.respond("question in conversation one")
    second = h.service.respond("question in conversation two")

    assert first.conversation_id != second.conversation_id
    assert len(h.llm.calls) == 2  # no rewrite call: the second conversation started empty
    assert _contents(h, second.conversation_id)[0] == "question in conversation two"
    assert len(_contents(h, second.conversation_id)) == 2


def test_an_unknown_conversation_id_is_treated_as_a_new_conversation() -> None:
    h = make_chat_harness(RELEVANT)
    never_seen = uuid.uuid4()

    turn = h.service.respond("hello", conversation_id=never_seen)

    assert turn.conversation_id == never_seen
    assert len(_contents(h, never_seen)) == 2


def test_an_unanswerable_question_is_still_stored_so_the_conversation_continues() -> None:
    h = make_chat_harness([make_chunk(score=0.05)])

    turn = h.service.respond("What is the capital of France?")

    assert turn.answer.grounded is False
    assert h.llm.calls == []
    assert _contents(h, turn.conversation_id) == [
        "What is the capital of France?",
        NO_ANSWER_TEXT,
    ]


def test_llm_failure_saves_nothing_and_releases_the_lock() -> None:
    h = make_chat_harness(RELEVANT)
    h.llm.fail = True
    conversation = uuid.uuid4()

    with pytest.raises(LLMUnavailableError):
        h.service.respond("hello", conversation_id=conversation)

    assert h.memory.get_history(conversation) == []  # a retry starts clean
    assert h.locks.held == set()


def test_lock_is_released_after_a_successful_turn() -> None:
    h = make_chat_harness(RELEVANT)
    h.service.respond("hello")
    assert h.locks.held == set()


def test_second_message_while_one_is_in_flight_is_rejected() -> None:
    h = make_chat_harness(RELEVANT)
    conversation = uuid.uuid4()
    h.locks.held.add(conversation_lock_name(conversation))  # a turn is already running

    with pytest.raises(ConversationBusyError):
        h.service.respond("hello", conversation_id=conversation)

    assert h.llm.calls == []
    assert h.memory.get_history(conversation) == []


def test_unreachable_lock_store_is_reported_as_memory_unavailable() -> None:
    h = make_chat_harness(RELEVANT)
    h.locks.unavailable = True
    with pytest.raises(ChatMemoryUnavailableError):
        h.service.respond("hello")


def test_unreadable_memory_is_reported_before_any_llm_work() -> None:
    h = make_chat_harness(RELEVANT, memory_redis=UnreadableRedis())
    with pytest.raises(ChatMemoryUnavailableError):
        h.service.respond("hello")
    assert h.llm.calls == []
    assert h.locks.held == set()


def test_failure_to_save_memory_is_an_error_not_a_silent_success() -> None:
    h = make_chat_harness(RELEVANT, memory_redis=UnwritableRedis(decode_responses=True))
    with pytest.raises(ChatMemoryUnavailableError):
        h.service.respond("hello")
    assert len(h.llm.calls) == 1  # the answer was generated, but we refuse to pretend it was saved
    assert h.locks.held == set()


def test_document_filter_and_top_k_are_passed_through() -> None:
    h = make_chat_harness(RELEVANT)
    h.service.respond("q", document_ids=[DEFAULT_DOC_ID], top_k=2)
    assert h.searcher.calls[0]["document_ids"] == [DEFAULT_DOC_ID]
    assert h.searcher.calls[0]["top_k"] == 2