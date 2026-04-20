"""Unit tests for the provider registry and TPM bucket."""
from __future__ import annotations

import asyncio
import time

import pytest

from src.providers.base import LLMResult, Message, ModelRef, TPMBucket


@pytest.mark.asyncio
async def test_tpm_bucket_admits_under_limit():
    bucket = TPMBucket(1000)
    await bucket.reserve(100)
    await bucket.reserve(400)
    # No exception means both reservations succeeded within the window.


@pytest.mark.asyncio
async def test_tpm_bucket_rejects_over_limit_until_window_resets(monkeypatch):
    bucket = TPMBucket(100)
    await bucket.reserve(80)

    started = time.monotonic()
    sleeps: list[float] = []

    original_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        # Fast-forward the bucket's window to trigger reset without real waiting.
        bucket._window_start = time.monotonic() - 61
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await bucket.reserve(80)
    assert time.monotonic() - started < 1.0
    assert sleeps, "expected at least one sleep before window reset"


def test_model_ref_key_is_unique():
    a = ModelRef(provider="groq", model="llama-3.1-8b-instant")
    b = ModelRef(provider="groq", model="llama-3.3-70b-versatile")
    assert a.key != b.key
    assert a.key == "groq:llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_registry_falls_back_on_first_provider_failure(monkeypatch, tmp_path):
    from src.providers import registry as registry_mod

    cfg = tmp_path / "providers.yaml"
    cfg.write_text(
        """
providers:
  groq:
    required: true
    models:
      - id: llama-3.1-8b-instant
        tpm: 1000
        tier: bulk
  openai:
    required: true
    models:
      - id: gpt-4o-mini
        tpm: 1000
        tier: bulk
"""
    )

    class _Healthy:
        def __init__(self, *_args, **_kwargs):
            pass

        def healthy(self):
            return True

    class _FlakyGroq(_Healthy):
        name = "groq"

        async def complete(self, model, messages, **_):
            raise RuntimeError("boom")

    class _GoodOpenAI(_Healthy):
        name = "openai"

        async def complete(self, model, messages, **_):
            return LLMResult(text="ok", model=model)

    monkeypatch.setitem(registry_mod._ADAPTERS, "groq", _FlakyGroq)
    monkeypatch.setitem(registry_mod._ADAPTERS, "openai", _GoodOpenAI)

    reg = registry_mod.ProviderRegistry(config_path=cfg)
    result = await reg.complete("bulk", [Message(role="user", content="hi")])
    assert result.text == "ok"
    assert result.model.provider == "openai"
