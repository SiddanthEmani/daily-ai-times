"""Shared pytest fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.pipeline.models import RawArticle

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Scrub provider keys so unit tests exercise the heuristic fallback path and
# never need vendor SDKs installed.
for _key in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(_key, None)

# Also force the registry to reload any cached instance under this scrubbed env.
try:
    from src.providers import registry as _registry

    _registry._singleton = None  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


@pytest.fixture
def raw_articles() -> list[RawArticle]:
    path = _FIXTURES / "articles.jsonl"
    return [RawArticle(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
