"""Mutable in-process state shared across MCP tool calls within one agent session.

Claude drives the sequence of tool calls; each call reads/writes this state so we
avoid shoveling entire article lists through the LLM context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.models import Article, ScoredArticle


@dataclass
class PipelineState:
    articles: list[Article] = field(default_factory=list)
    scored: list[ScoredArticle] = field(default_factory=list)
    classified: list[ScoredArticle] = field(default_factory=list)
    category_rankings: dict[str, list[ScoredArticle]] = field(default_factory=dict)
    latest: list[ScoredArticle] = field(default_factory=list)
    widget: list[ScoredArticle] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)


_state: PipelineState | None = None


def get_state() -> PipelineState:
    global _state
    if _state is None:
        _state = PipelineState()
    return _state


def reset_state() -> None:
    global _state
    _state = PipelineState()
