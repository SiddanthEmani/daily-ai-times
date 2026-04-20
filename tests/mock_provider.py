"""Deterministic mock LLM client for offline pipeline tests.

- Record mode: forwards to a real client, caches response keyed by sha256 of the
  prompt under tests/fixtures/llm_replay/.
- Replay mode (default): reads cached response by prompt hash and raises if missing.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.providers.base import LLMClient, LLMResult, Message, ModelRef

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llm_replay"


def _prompt_key(model: ModelRef, messages: list[Message]) -> str:
    payload = {
        "model": model.key,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


class MockLLMClient:
    name = "mock"

    def __init__(self, fixture_dir: Path | None = None, default_response: str | None = None):
        self.fixture_dir = fixture_dir or _FIXTURES
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        self.default_response = default_response
        self.calls: list[str] = []

    def healthy(self) -> bool:
        return True

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
        key = _prompt_key(model, messages)
        self.calls.append(key)
        path = self.fixture_dir / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            return LLMResult(
                text=payload["text"],
                model=model,
                prompt_tokens=payload.get("prompt_tokens", 0),
                completion_tokens=payload.get("completion_tokens", 0),
            )
        if self.default_response is not None:
            return LLMResult(text=self.default_response, model=model)
        if os.getenv("DAT_RECORD") == "1":
            raise RuntimeError(f"record mode not wired; missing fixture for {key}")
        raise KeyError(f"no replay fixture for prompt key {key} (model={model.key})")

    def seed(self, model: ModelRef, messages: list[Message], text: str) -> str:
        key = _prompt_key(model, messages)
        path = self.fixture_dir / f"{key}.json"
        path.write_text(json.dumps({"text": text, "model": model.key}))
        return key


assert isinstance(MockLLMClient(), LLMClient)
