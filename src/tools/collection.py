"""Collection-related MCP tools."""
from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import tool

from src.pipeline.collect import collect, load_sources
from src.pipeline.dedupe import dedupe
from src.pipeline.normalize import normalize
from src.tools.state import get_state


@tool(
    "collect_sources",
    "Fetch articles from the configured source list. Optional category filter. "
    "Returns a summary with counts and sample titles. Full results are stored "
    "in agent state under key 'articles' for downstream tools.",
    {"category": str},
)
async def collect_sources(args: dict[str, Any]) -> dict[str, Any]:
    try:
        category = args.get("category", "")
        all_sources = load_sources()
        sources = [s for s in all_sources if not category or s.category == category]
        raw = await collect(sources)
        articles = dedupe(normalize(raw))
        state = get_state()
        state.articles = articles
        payload = {
            "sources_polled": len(sources),
            "raw_count": len(raw),
            "deduped_count": len(articles),
            "sample_titles": [a.title for a in articles[:5]],
            "categories": sorted({a.category for a in articles}),
        }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2)}]
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "content": [{"type": "text", "text": f"collect_sources failed: {exc}"}],
            "is_error": True,
        }


@tool(
    "list_sources",
    "List configured news sources. Optional category filter.",
    {"category": str},
)
async def list_sources(args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category", "")
    sources = load_sources()
    if category:
        sources = [s for s in sources if s.category == category]
    summary = [
        {"id": s.id, "name": s.name, "category": s.category, "kind": s.kind, "url": s.url}
        for s in sources
    ]
    return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}


@tool(
    "update_source_config",
    "Enable or disable a source by id. Writes back to src/config/sources/*.yaml. "
    "Requires permission approval - this is a destructive filesystem write.",
    {"source_id": str, "enabled": bool},
)
async def update_source_config(args: dict[str, Any]) -> dict[str, Any]:
    # Intentionally a stub: real implementation needs permission middleware.
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"update_source_config({args['source_id']}, enabled={args['enabled']}) "
                    "deferred: awaiting permission middleware integration."
                ),
            }
        ],
        "is_error": True,
    }
