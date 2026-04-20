"""Programmatic subagent definitions for isolated contexts.

The Claude Agent SDK exposes `AgentDefinition` so we can route heavy, noisy
sub-tasks (scoring hundreds of articles, classifying per-category) through
their own context windows without bloating the main orchestrator's context.
"""
from __future__ import annotations

from typing import Any

try:
    from claude_agent_sdk import AgentDefinition
except ImportError:  # pragma: no cover — SDK present in prod/CI
    class AgentDefinition:  # type: ignore[override]
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs


SCORER = AgentDefinition(
    description=(
        "Dedicated scoring subagent. Given an array of article metadata, returns "
        "a scored array via the daily_ai MCP server. Isolated context keeps the "
        "main orchestrator lean."
    ),
    prompt=(
        "You are a bulk-tier news scoring specialist. Your only job is to call "
        "`mcp__daily_ai__score_batch` once with the instructions given by the "
        "parent agent, then summarize the score distribution in at most three "
        "sentences. Never call any other tool."
    ),
    tools=["mcp__daily_ai__score_batch"],
    model="haiku",
)


CLASSIFIER = AgentDefinition(
    description=(
        "Dedicated classification subagent. Normalizes categories on the "
        "currently-scored article pool using the daily_ai classify tool."
    ),
    prompt=(
        "You are a classification specialist. Call `mcp__daily_ai__classify_articles` "
        "once, then report the per-category counts. Never call any other tool."
    ),
    tools=["mcp__daily_ai__classify_articles"],
    model="haiku",
)


AGENTS: dict[str, AgentDefinition] = {
    "scorer": SCORER,
    "classifier": CLASSIFIER,
}
