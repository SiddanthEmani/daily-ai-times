"""Expose all v2 tools through an in-process MCP server.

Usage:
    from src.tools.server import build_server
    server = build_server()

The server is attached to `ClaudeAgentOptions.mcp_servers={'daily_ai': server}`
so every tool becomes `mcp__daily_ai__<tool>`.
"""
from __future__ import annotations

from claude_agent_sdk import create_sdk_mcp_server

from src.tools.collection import (
    collect_sources,
    list_sources,
    update_source_config,
)
from src.tools.publishing import parity_check, publish_v2, validate_api
from src.tools.ranking import pick_headlines, rank_category
from src.tools.scoring import classify_articles, score_batch

TOOLS = [
    collect_sources,
    list_sources,
    update_source_config,
    score_batch,
    classify_articles,
    rank_category,
    pick_headlines,
    publish_v2,
    validate_api,
    parity_check,
]


def build_server(name: str = "daily_ai", version: str = "1.0.0"):
    return create_sdk_mcp_server(name=name, version=version, tools=TOOLS)


ALLOWED_TOOL_NAMES = [f"mcp__daily_ai__{t.name if hasattr(t, 'name') else ''}" for t in TOOLS]


if __name__ == "__main__":
    # Standalone stdio entrypoint for manual debugging with `make mcp`.
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    build_server()
    print("Registered tools:")
    for t in TOOLS:
        print(f"  - {getattr(t, 'name', t)}")
    print("\nIn-process SDK MCP servers are consumed by Claude Agent SDK's query().")
    print("Run `make publish-v2` to invoke them through the full agent loop.")
    asyncio.run(asyncio.sleep(0))
