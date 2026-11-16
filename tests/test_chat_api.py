import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
from app.services.chat import conversation_lock_name
from app.services.rag import NO_ANSWER_TEXT
from tests.fakes import DEFAULT_DOC_ID, ChatHarness, make_chat_harness, make_chunk

URL = "/api/v1/chat"
RELEVANT = [make_chunk("Unsupervised learning finds structure in unlabelled data.", 2, 0.8)]


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def client_for(harness: ChatHarness) -> TestClient:
    app.dependency_overrides[deps.get_chat_service] = lambda: harness.service
    return TestClient(app, raise_server_exceptions=False)


def test_first_message_returns_an_answer_sources_and_a_new_conversation_id() -> None:
    client = client_for(make_chat_harness(RELEVANT, ["Learning without labels."]))

    response = client.post(URL, json={"message": "What is unsupervised learning?"})

    assert response.status_code == 200
    body = response.json()
    uuid.UUID(body["conversation_id"])
    assert body["response"] == "Learning without labels."
    assert body["grounded"] is True
    assert body["search_query"] == "What is unsupervised learning?"
    assert body["sources"] == [
        {
            "document_id": str(DEFAULT_DOC_ID),
            "filename": "notes.txt",
            "chunk_index": 2,
            "score": 0.8,
            "snippet": "Unsupervised learning finds structure in unlabelled data.",
        }
    ]


def test_the_conversation_continues_with_the_same_id_and_uses_history() -> None:
    harness = make_chat_harness(RELEVANT, ["First answer", "rewritten query", "Second answer"])
    client = client_for(harness)

    first = client.post(URL, json={"message": "Q1"}).json()
    second = client.post(
        URL, json={"message": "Q2", "conversation_id": first["conversation_id"]}
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert second["search_query"] == "rewritten query"
    assert second["response"] == "Second answer"
    assert len(harness.llm.calls) == 3


def test_no_relevant_documents_gives_an_ungrounded_reply_with_no_sources() -> None:
    client = client_for(make_chat_harness([make_chunk(score=0.05)]))

    body = client.post(URL, json={"message": "What is the capital of France?"}).json()

    assert body["response"] == NO_ANSWER_TEXT
    assert body["grounded"] is False
    assert body["sources"] == []


def test_document_filter_and_top_k_reach_retrieval() -> None:
    harness = make_chat_harness(RELEVANT)
    client = client_for(harness)

    client.post(URL, json={"message": "q", "document_ids": [str(DEFAULT_DOC_ID)], "top_k": 2})

    assert harness.searcher.calls[0]["document_ids"] == [DEFAULT_DOC_ID]
    assert harness.searcher.calls[0]["top_k"] == 2


def test_blank_message_is_rejected() -> None:
    response = client_for(make_chat_harness()).post(URL, json={"message": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_missing_message_is_rejected() -> None:
    response = client_for(make_chat_harness()).post(URL, json={})
    assert response.status_code == 422
    assert "message" in response.json()["error"]["message"]


def test_overlong_message_is_rejected() -> None:
    response = client_for(make_chat_harness()).post(URL, json={"message": "x" * 2001})
    assert response.status_code == 422


def test_malformed_conversation_id_is_rejected() -> None:
    response = client_for(make_chat_harness()).post(
        URL, json={"message": "hi", "conversation_id": "../../etc/passwd"}
    )
    assert response.status_code == 422
    assert "conversation_id" in response.json()["error"]["message"]


@pytest.mark.parametrize("top_k", [0, 11])
def test_top_k_outside_the_allowed_range_is_rejected(top_k: int) -> None:
    response = client_for(make_chat_harness()).post(URL, json={"message": "hi", "top_k": top_k})
    assert response.status_code == 422
    assert "top_k" in response.json()["error"]["message"]


def test_llm_outage_returns_503_with_a_stable_error_code() -> None:
    harness = make_chat_harness(RELEVANT)
    harness.llm.fail = True

    response = client_for(harness).post(URL, json={"message": "hello"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


def test_concurrent_message_in_the_same_conversation_returns_409() -> None:
    harness = make_chat_harness(RELEVANT)
    conversation = uuid.uuid4()
    harness.locks.held.add(conversation_lock_name(conversation))

    response = client_for(harness).post(
        URL, json={"message": "hello", "conversation_id": str(conversation)}
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conversation_busy"


def test_memory_outage_returns_503() -> None:
    harness = make_chat_harness(RELEVANT)
    harness.locks.unavailable = True

    response = client_for(harness).post(URL, json={"message": "hello"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "memory_unavailable"