import json
from typing import Any

import httpx
import pytest

from app.core.exceptions import LLMUnavailableError
from app.services.llm import LLMMessage, OllamaClient


def make_client(handler: Any) -> OllamaClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return OllamaClient(http, model="test-model", context_tokens=4096, default_max_tokens=300)


def ok(content: Any = "hello") -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "done": True})


MESSAGES = [LLMMessage("system", "rules"), LLMMessage("user", "hi")]


def test_request_has_the_expected_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return ok()

    make_client(handler).chat(MESSAGES, temperature=0.1, max_tokens=50)

    body = seen["body"]
    assert seen["path"] == "/api/chat"
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    assert body["options"] == {"temperature": 0.1, "num_ctx": 4096, "num_predict": 50}
    assert "format" not in body


def test_default_max_tokens_is_used_when_not_given() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return ok()

    make_client(handler).chat(MESSAGES)
    assert seen["body"]["options"]["num_predict"] == 300


def test_json_mode_requests_json_output() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return ok('{"a": 1}')

    make_client(handler).chat(MESSAGES, json_mode=True)
    assert seen["body"]["format"] == "json"


def test_reply_text_is_stripped() -> None:
    assert make_client(lambda request: ok("  an answer \n")).chat(MESSAGES) == "an answer"


def _raise(exc: Exception) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def test_connection_failure_becomes_llm_unavailable() -> None:
    client = make_client(_raise(httpx.ConnectError("connection refused")))
    with pytest.raises(LLMUnavailableError):
        client.chat(MESSAGES)


def test_timeout_becomes_llm_unavailable() -> None:
    client = make_client(_raise(httpx.ReadTimeout("too slow")))
    with pytest.raises(LLMUnavailableError):
        client.chat(MESSAGES)


def test_server_error_becomes_llm_unavailable() -> None:
    client = make_client(lambda request: httpx.Response(500, text="internal failure"))
    with pytest.raises(LLMUnavailableError):
        client.chat(MESSAGES)


def test_unknown_model_becomes_llm_unavailable() -> None:
    client = make_client(lambda request: httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(LLMUnavailableError):
        client.chat(MESSAGES)


def test_non_json_body_becomes_llm_unavailable() -> None:
    client = make_client(lambda request: httpx.Response(200, content=b"not json"))
    with pytest.raises(LLMUnavailableError):
        client.chat(MESSAGES)


def test_missing_message_field_becomes_llm_unavailable() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"done": True}))
    with pytest.raises(LLMUnavailableError):
        client.chat(MESSAGES)


def test_empty_reply_becomes_llm_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        make_client(lambda request: ok("   ")).chat(MESSAGES)


def test_non_text_reply_becomes_llm_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        make_client(lambda request: ok(123)).chat(MESSAGES)