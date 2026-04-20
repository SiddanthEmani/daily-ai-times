"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.models import RawArticle

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def raw_articles() -> list[RawArticle]:
    path = _FIXTURES / "articles.jsonl"
    return [RawArticle(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
