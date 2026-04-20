"""Collection stage: fetch raw articles from configured sources.

Pure async. No LLM calls. Loads `src/config/sources/*.yaml` and hits each
source concurrently with a bounded semaphore.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import feedparser
import yaml
from pydantic import TypeAdapter

from src.pipeline.models import RawArticle, SourceSpec

log = logging.getLogger(__name__)

_SOURCES_DIR = Path(__file__).resolve().parents[1] / "config" / "sources"
_SOURCE_LIST = TypeAdapter(list[SourceSpec])


def load_sources(directory: Path = _SOURCES_DIR) -> list[SourceSpec]:
    """Flatten every sources/*.yaml into a list of SourceSpec."""
    specs: list[SourceSpec] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        data: dict = yaml.safe_load(yaml_file.read_text()) or {}
        category_from_file = yaml_file.stem
        sources_block = data.get("sources") or {}
        if isinstance(sources_block, dict):
            for source_id, cfg in sources_block.items():
                specs.append(
                    SourceSpec(
                        id=source_id,
                        name=cfg.get("name", source_id),
                        url=cfg.get("url") or cfg.get("rss_url", ""),
                        kind=cfg.get("type", cfg.get("kind", "rss")),
                        category=cfg.get("category", category_from_file),
                        weight=float(cfg.get("weight", 1.0)),
                    )
                )
        elif isinstance(sources_block, list):
            specs.extend(_SOURCE_LIST.validate_python(sources_block))
    return [s for s in specs if s.url]


async def _fetch_rss(
    session: aiohttp.ClientSession, source: SourceSpec, timeout: float
) -> list[RawArticle]:
    async with session.get(source.url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        resp.raise_for_status()
        body = await resp.read()
    feed = feedparser.parse(body)
    out: list[RawArticle] = []
    for entry in feed.entries:
        published = None
        if getattr(entry, "published_parsed", None):
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif getattr(entry, "updated_parsed", None):
            published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        out.append(
            RawArticle(
                source_id=source.id,
                source_name=source.name,
                category=source.category,
                title=(getattr(entry, "title", "") or "").strip(),
                url=getattr(entry, "link", "") or "",
                summary=(getattr(entry, "summary", "") or "").strip(),
                published_at=published,
                author=getattr(entry, "author", "") or "",
            )
        )
    return out


async def collect(
    sources: list[SourceSpec] | None = None,
    *,
    concurrency: int = 12,
    timeout: float = 30.0,
) -> list[RawArticle]:
    """Fan-out fetch across sources. Returns raw articles in arrival order."""
    sources = sources or load_sources()
    if not sources:
        return []
    sem = asyncio.Semaphore(concurrency)
    results: list[RawArticle] = []

    async with aiohttp.ClientSession(
        headers={"User-Agent": "DailyAITimes/3.0 (+https://github.com/SiddanthEmani/daily-ai-times)"}
    ) as session:

        async def _worker(src: SourceSpec) -> None:
            async with sem:
                try:
                    items = await _fetch_rss(session, src, timeout)
                    results.extend(items)
                except Exception as exc:  # noqa: BLE001 — log and continue
                    log.warning("collect failed: %s (%s): %s", src.id, src.url, exc)

        await asyncio.gather(*(_worker(s) for s in sources))

    log.info("collected %d raw articles from %d sources", len(results), len(sources))
    return results
