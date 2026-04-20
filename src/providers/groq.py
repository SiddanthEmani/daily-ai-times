"""Groq adapter."""
from __future__ import annotations

import os
import time
from typing import Any

from src.providers.base import LLMClient, LLMResult, Message, ModelRef


class GroqClient:
    """Wraps the official groq async SDK."""

    name = "groq"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._client: Any | None = None

    def healthy(self) -> bool:
        return bool(self._api_key)

    def _client_lazy(self) -> Any:
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    async def complete(
        self,
        model: ModelRef,
        messages: list[Message],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> LLMResult:
        client = self._client_lazy()
        kwargs: dict[str, Any] = {
            "model": model.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "timeout": timeout,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        started = time.perf_counter()
        response = await client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=choice.message.content or "",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency,
            raw=response,
        )


assert isinstance(GroqClient(api_key="x"), LLMClient)
