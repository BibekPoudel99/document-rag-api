"""LLM access. The rest of the app depends on the LLMClient protocol, not on Ollama,
so another provider can be added by writing one more small class."""

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from app.core.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...


class OllamaClient:
    """Talks to Ollama's /api/chat endpoint over HTTP."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        model: str,
        context_tokens: int,
        default_max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._context_tokens = context_tokens
        self._default_max_tokens = default_max_tokens

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                # Set explicitly: a prompt longer than the context window is
                # truncated by the server rather than rejected.
                "num_ctx": self._context_tokens,
                "num_predict": max_tokens or self._default_max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except httpx.HTTPError as exc:
            detail = exc.response.text[:200] if isinstance(exc, httpx.HTTPStatusError) else ""
            logger.error("LLM request failed: %s: %s %s", type(exc).__name__, exc, detail)
            raise LLMUnavailableError("The language model is currently unavailable.") from exc
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("LLM returned an unexpected response: %s: %s", type(exc).__name__, exc)
            raise LLMUnavailableError(
                "The language model returned an unexpected response."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            logger.error("LLM returned empty or non-text content")
            raise LLMUnavailableError("The language model returned an empty response.")
        return content.strip()