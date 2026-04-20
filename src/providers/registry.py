"""Provider registry: routing, TPM buckets, health, retries.

Reads `src/config/providers.yaml` and instantiates adapters lazily so a
missing API key for one provider doesn't break the whole registry.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.providers.base import (
    LLMClient,
    LLMResult,
    Message,
    ModelRef,
    TPMBucket,
)
from src.providers.cerebras import CerebrasClient
from src.providers.gemini import GeminiClient
from src.providers.groq import GroqClient
from src.providers.openai import OpenAIClient

log = logging.getLogger(__name__)

_ADAPTERS: dict[str, type[LLMClient]] = {
    "groq": GroqClient,
    "cerebras": CerebrasClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


@dataclass(slots=True)
class ModelEntry:
    ref: ModelRef
    bucket: TPMBucket


class ProviderRegistry:
    """Routes LLM calls by tier with health filtering and TPM reservations."""

    def __init__(self, config_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self._config_path = config_path or root / "src" / "config" / "providers.yaml"
        self._clients: dict[str, LLMClient] = {}
        self._models: dict[str, ModelEntry] = {}
        self._by_tier: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self._config_path.exists():
            log.warning("providers.yaml not found at %s; registry empty", self._config_path)
            return
        data: dict[str, Any] = yaml.safe_load(self._config_path.read_text()) or {}
        for provider_name, cfg in (data.get("providers") or {}).items():
            adapter_cls = _ADAPTERS.get(provider_name)
            if not adapter_cls:
                log.warning("unknown provider %s in providers.yaml", provider_name)
                continue
            client = adapter_cls()
            if not client.healthy() and not cfg.get("required", False):
                log.info("provider %s skipped (no API key)", provider_name)
                continue
            self._clients[provider_name] = client
            for model_cfg in cfg.get("models", []) or []:
                ref = ModelRef(
                    provider=provider_name,
                    model=model_cfg["id"],
                    tpm=int(model_cfg.get("tpm", 0)),
                    tier=model_cfg.get("tier", "bulk"),
                )
                self._models[ref.key] = ModelEntry(ref=ref, bucket=TPMBucket(ref.tpm))
                self._by_tier.setdefault(ref.tier, []).append(ref.key)
        log.info(
            "registry loaded: %d providers, %d models (tiers=%s)",
            len(self._clients),
            len(self._models),
            list(self._by_tier.keys()),
        )

    @property
    def providers(self) -> list[str]:
        return list(self._clients.keys())

    @property
    def models(self) -> list[ModelRef]:
        return [entry.ref for entry in self._models.values()]

    def models_in_tier(self, tier: str) -> list[ModelRef]:
        return [self._models[k].ref for k in self._by_tier.get(tier, [])]

    def pick(self, tier: str, *, prefer: str | None = None) -> ModelRef | None:
        candidates = self._by_tier.get(tier, [])
        if not candidates:
            return None
        if prefer:
            for key in candidates:
                if self._models[key].ref.provider == prefer:
                    return self._models[key].ref
        return self._models[random.choice(candidates)].ref

    async def complete(
        self,
        tier: str,
        messages: list[Message],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        estimated_tokens: int = 1024,
        prefer: str | None = None,
        attempts: int = 3,
    ) -> LLMResult:
        """Pick a model in tier, reserve TPM, call, retry with alt providers on failure."""
        tried: set[str] = set()
        last_error: BaseException | None = None
        for attempt in range(attempts):
            model = self.pick(tier, prefer=prefer if attempt == 0 else None)
            if not model or model.key in tried:
                alternates = [
                    self._models[k].ref
                    for k in self._by_tier.get(tier, [])
                    if k not in tried
                ]
                if not alternates:
                    break
                model = alternates[0]
            tried.add(model.key)
            entry = self._models[model.key]
            await entry.bucket.reserve(estimated_tokens)
            client = self._clients[model.provider]
            try:
                return await client.complete(
                    model,
                    messages,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                log.warning(
                    "provider %s model %s failed (attempt %d/%d): %s",
                    model.provider,
                    model.model,
                    attempt + 1,
                    attempts,
                    exc,
                )
                await asyncio.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            f"no provider satisfied tier={tier} after {attempts} attempts: {last_error}"
        )


_singleton: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _singleton
    if _singleton is None:
        _singleton = ProviderRegistry()
    return _singleton
