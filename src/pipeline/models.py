"""Pydantic models for pipeline stage I/O. Every stage boundary is validated."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    url: str
    kind: Literal["rss", "atom", "reddit", "hn", "github", "json"] = "rss"
    category: str = "general"
    weight: float = 1.0


class RawArticle(BaseModel):
    """Raw article straight from a source, before normalization."""

    source_id: str
    source_name: str
    category: str
    title: str
    url: str
    summary: str = ""
    published_at: datetime | None = None
    author: str = ""
    raw: dict | None = None


class Article(BaseModel):
    """Normalized, deduped article ready for scoring."""

    id: str
    source_id: str
    source_name: str
    category: str
    title: str
    url: str
    summary: str
    published_at: datetime
    author: str = ""
    fingerprint: str = ""

    @field_validator("published_at", mode="before")
    @classmethod
    def _tz_aware(cls, v: datetime | str | None) -> datetime:
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v


class ScoredArticle(BaseModel):
    article: Article
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    categories: list[str] = Field(default_factory=list)
    model_used: str = ""
    reasoning: str = ""


class RankedCategory(BaseModel):
    name: str
    articles: list[ScoredArticle]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApiBundle(BaseModel):
    """Everything the frontend needs. Drives the publish stage."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latest: list[ScoredArticle]
    widget: list[ScoredArticle]
    categories: dict[str, list[ScoredArticle]]
    stats: dict[str, int | float | str]
