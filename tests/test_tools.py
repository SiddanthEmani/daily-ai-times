"""Direct tool handler tests (bypass the MCP envelope, exercise the callable)."""
from __future__ import annotations

import json

import pytest

from src.pipeline.dedupe import dedupe
from src.pipeline.normalize import normalize
from src.tools import reset_state
from src.tools.publishing import validate_api
from src.tools.ranking import pick_headlines, rank_category
from src.tools.scoring import classify_articles, score_batch
from src.tools.state import get_state


def _call(tool, args):
    """Invoke the underlying handler (skips MCP envelope)."""
    inner = getattr(tool, "handler", None) or getattr(tool, "func", None) or tool
    return inner(args)


@pytest.mark.asyncio
async def test_score_then_classify_then_rank(raw_articles):
    reset_state()
    state = get_state()
    state.articles = dedupe(normalize(raw_articles))

    res = await _call(score_batch, {"limit": 10})
    assert res.get("is_error") is not True
    assert state.scored and len(state.scored) == len(state.articles)

    res = await _call(classify_articles, {})
    assert res.get("is_error") is not True

    any_cat = state.classified[0].article.category
    res = await _call(rank_category, {"category": any_cat, "limit": 5})
    assert res.get("is_error") is not True
    assert any_cat in state.category_rankings

    res = await _call(pick_headlines, {"latest_limit": 5, "widget_limit": 3})
    assert res.get("is_error") is not True
    assert len(state.latest) <= 5
    assert len(state.widget) <= 3


@pytest.mark.asyncio
async def test_validate_api_detects_missing(tmp_path, monkeypatch):
    import src.tools.publishing as pub_mod

    monkeypatch.setattr(pub_mod, "_FRONTEND_ROOT", tmp_path)
    result = await _call(validate_api, {})
    assert result.get("is_error") is True
    body = result["content"][0]["text"]
    assert "missing required files" in body


def test_server_registers_all_tools():
    from src.tools.server import ALLOWED_TOOL_NAMES, TOOLS

    assert len(ALLOWED_TOOL_NAMES) == len(TOOLS) > 0
    for name in ALLOWED_TOOL_NAMES:
        assert name.startswith("mcp__daily_ai__")
