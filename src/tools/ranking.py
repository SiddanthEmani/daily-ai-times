"""Ranking + headline selection MCP tools."""
from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import tool

from src.pipeline.models import ScoredArticle
from src.tools.state import get_state


def _rank(articles: list[ScoredArticle], *, recency_weight: float = 0.15) -> list[ScoredArticle]:
    # Newer + higher confidence gently boost score.
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)

    def _key(s: ScoredArticle) -> float:
        age_hours = max((now - s.article.published_at).total_seconds() / 3600, 0.0)
        recency = 1.0 / (1.0 + age_hours / 24.0)
        return s.score * (1 - recency_weight) + recency * recency_weight

    return sorted(articles, key=_key, reverse=True)


@tool(
    "rank_category",
    "Rank classified articles inside one category by combined score+recency. "
    "Writes state['category_rankings'][category]. Returns top 5 titles.",
    {"category": str, "limit": int},
)
async def rank_category(args: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    category = args["category"]
    limit = int(args.get("limit", 20))
    pool = state.classified or state.scored
    in_cat = [s for s in pool if s.article.category == category]
    if not in_cat:
        return {
            "content": [{"type": "text", "text": f"no articles in category '{category}'"}],
            "is_error": True,
        }
    ranked = _rank(in_cat)[:limit]
    state.category_rankings[category] = ranked
    payload = {
        "category": category,
        "count": len(ranked),
        "top_5": [{"title": s.article.title, "score": round(s.score, 3)} for s in ranked[:5]],
    }
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}


@tool(
    "pick_headlines",
    "Pick the top-N headlines across all categories for the 'latest' feed and "
    "widget. Writes state['latest'] and state['widget'].",
    {"latest_limit": int, "widget_limit": int},
)
async def pick_headlines(args: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    latest_limit = int(args.get("latest_limit", 30))
    widget_limit = int(args.get("widget_limit", 8))
    pool: list[ScoredArticle] = []
    if state.category_rankings:
        for ranked in state.category_rankings.values():
            pool.extend(ranked)
    else:
        pool = state.classified or state.scored
    ranked = _rank(pool)
    state.latest = ranked[:latest_limit]
    state.widget = ranked[:widget_limit]
    payload = {
        "latest": len(state.latest),
        "widget": len(state.widget),
        "top_titles": [s.article.title for s in state.widget],
    }
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}
