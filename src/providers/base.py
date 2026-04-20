"""Provider-neutral LLM client protocol and shared types.

All provider adapters (Groq, Cerebras, OpenAI, Gemini) implement the same
`LLMClient` protocol so tools and subagents can swap providers via ModelRef
without touching call sites.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Reference to a concrete model hosted by a provider."""

    provider: str
    model: str
    tpm: int = 0
    tier: str = "bulk"

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class LLMResult:
    text: str
    model: ModelRef
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol every provider adapter must satisfy."""

    name: str

    async def complete(
        self,
        model: ModelRef,
        messages: list[Message],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> LLMResult: ...

    def healthy(self) -> bool: ...


class TPMBucket:
    """Per-minute token bucket. Non-blocking reserve + async wait."""

    __slots__ = ("limit", "_used", "_window_start", "_lock")

    def __init__(self, limit: int) -> None:
        self.limit = max(limit, 0)
        self._used = 0
        self._window_start = time.monotonic()
        self._lock = asyncio.Lock()

    async def reserve(self, tokens: int) -> None:
        if self.limit == 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                if now - self._window_start >= 60.0:
                    self._used = 0
                    self._window_start = now
                if self._used + tokens <= self.limit:
                    self._used += tokens
                    return
                wait_for = 60.0 - (now - self._window_start)
            await asyncio.sleep(max(wait_for, 0.05))


@dataclass(slots=True)
class ProviderSpec:
    name: str
    factory: Callable[[], LLMClient]
    models: list[ModelRef] = field(default_factory=list)
    env_key: str = ""


RetryPredicate = Callable[[BaseException], Awaitable[bool] | bool]
