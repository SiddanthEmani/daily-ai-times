"""Google Gemini adapter using the google-genai SDK."""
from __future__ import annotations

import os
import time
from typing import Any

from src.providers.base import LLMClient, LLMResult, Message, ModelRef


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY", "")
        )
        self._client: Any | None = None

    def healthy(self) -> bool:
        return bool(self._api_key)

    def _client_lazy(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
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
        system_parts = [m.content for m in messages if m.role == "system"]
        user_parts = [m.content for m in messages if m.role != "system"]
        config: dict[str, Any] = {"temperature": temperature}
        if system_parts:
            config["system_instruction"] = "\n\n".join(system_parts)
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if response_format and response_format.get("type") == "json_schema":
            config["response_mime_type"] = "application/json"
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                config["response_schema"] = schema
        started = time.perf_counter()
        response = await client.aio.models.generate_content(
            model=model.model,
            contents="\n\n".join(user_parts),
            config=config,
        )
        latency = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            text=response.text or "",
            model=model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            latency_ms=latency,
            raw=response,
        )


assert isinstance(GeminiClient(api_key="x"), LLMClient)
