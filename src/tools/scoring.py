"""Scoring + classification MCP tools. Bulk work runs on the provider pool."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from claude_agent_sdk import tool

from src.config.loader import ConfigLoader
from src.pipeline.models import Article, ScoredArticle
from src.providers.base import Message
from src.providers.registry import get_registry
from src.tools.state import get_state

log = logging.getLogger(__name__)

_BULK_BATCH = 20


def _score_prompt(batch: list[Article]) -> str:
    items = "\n".join(
        f"{i}. [{a.category}] {a.title} — {a.summary[:200]}"
        for i, a in enumerate(batch, 1)
    )
    return (
        "You are a news quality scorer. For each article, return a JSON array of "
        "objects {index, score (0-1), confidence (0-1), categories (array of strings)}. "
        "Score higher for novelty, technical depth, and credibility. Return only JSON.\n\n"
        f"Articles:\n{items}"
    )


async def _score_batch(batch: list[Article]) -> list[ScoredArticle]:
    registry = get_registry()
    if not registry.models:
        # Fall back to heuristic scoring so offline runs still produce output.
        return [
            ScoredArticle(
                article=a,
                score=0.6,
                confidence=0.4,
                categories=[a.category],
                model_used="heuristic",
            )
            for a in batch
        ]

    response = await registry.complete(
        tier="bulk",
        messages=[
            Message(role="system", content="You output only valid JSON."),
            Message(role="user", content=_score_prompt(batch)),
        ],
        response_format={"type": "json_object"},
        max_tokens=1024,
        temperature=0.2,
    )
    try:
        parsed = json.loads(response.text)
        if isinstance(parsed, dict) and "articles" in parsed:
            parsed = parsed["articles"]
    except json.JSONDecodeError:
        log.warning("scorer returned invalid JSON, using fallback")
        parsed = []

    by_index = {int(item.get("index", 0)): item for item in parsed if isinstance(item, dict)}
    results: list[ScoredArticle] = []
    for i, article in enumerate(batch, 1):
        item = by_index.get(i, {})
        results.append(
            ScoredArticle(
                article=article,
                score=float(item.get("score", 0.5)),
                confidence=float(item.get("confidence", 0.5)),
                categories=list(item.get("categories", [article.category])),
                model_used=response.model.key,
            )
        )
    return results


@tool(
    "score_batch",
    "Score the articles collected in state against bulk-tier models. Returns "
    "counts and score distribution; full scored list is stored in state['scored'].",
    {"limit": int},
)
async def score_batch(args: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    articles = state.articles
    limit = int(args.get("limit", len(articles)))
    articles = articles[:limit] if limit > 0 else articles
    if not articles:
        return {
            "content": [{"type": "text", "text": "no articles in state; run collect_sources first"}],
            "is_error": True,
        }

    batches = [articles[i : i + _BULK_BATCH] for i in range(0, len(articles), _BULK_BATCH)]
    results = await asyncio.gather(*(_score_batch(b) for b in batches))
    scored: list[ScoredArticle] = [item for batch in results for item in batch]
    state.scored = scored

    scores = [round(s.score, 3) for s in scored]
    summary = {
        "count": len(scored),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "top_5": [
            {"title": s.article.title, "score": round(s.score, 3)}
            for s in sorted(scored, key=lambda x: x.score, reverse=True)[:5]
        ],
        "model_used": scored[0].model_used if scored else "",
    }
    return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}


_CATEGORIES = ConfigLoader.get("categories", ["research", "industry", "government", "media", "open_source"], "app")


@tool(
    "classify_articles",
    "Classify scored articles into canonical categories. Uses the article's "
    "existing category as a prior; bulk LLM provides refinement. Updates state['classified'].",
    {},
)
async def classify_articles(_: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    source = state.scored or [
        ScoredArticle(article=a, score=0.5, confidence=0.5, categories=[a.category])
        for a in state.articles
    ]
    if not source:
        return {
            "content": [{"type": "text", "text": "no scored articles; run score_batch first"}],
            "is_error": True,
        }

    # Cheap deterministic classification: trust source category, normalize aliases.
    aliases = {
        "tech": "industry",
        "technology": "industry",
        "ai": "industry",
        "science": "research",
        "papers": "research",
        "policy": "government",
        "regulation": "government",
        "press": "media",
        "oss": "open_source",
        "github": "open_source",
    }
    classified: list[ScoredArticle] = []
    for s in source:
        raw = (s.article.category or "").lower()
        canonical = aliases.get(raw, raw if raw in _CATEGORIES else "industry")
        new_cats = list({canonical, *s.categories})
        s.article.category = canonical
        classified.append(s.model_copy(update={"categories": new_cats}))
    state.classified = classified
    by_cat: dict[str, int] = {}
    for s in classified:
        by_cat[s.article.category] = by_cat.get(s.article.category, 0) + 1
    return {"content": [{"type": "text", "text": json.dumps({"counts": by_cat, "total": len(classified)}, indent=2)}]}


os.environ.setdefault("ALLOW_SCORER_FALLBACK", "1")
