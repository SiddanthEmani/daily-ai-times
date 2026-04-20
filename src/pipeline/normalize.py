"""Normalize stage: clean HTML, canonicalize URLs, attach stable IDs."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

from src.pipeline.models import Article, RawArticle

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "fbclid",
    "gclid",
}


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        return url
    query_pairs = [
        pair
        for pair in parsed.query.split("&")
        if pair and pair.split("=", 1)[0] not in _TRACKING_PARAMS
    ]
    cleaned_query = "&".join(query_pairs)
    cleaned = parsed._replace(query=cleaned_query, fragment="")
    out = urlunparse(cleaned).rstrip("/")
    return out


def _stable_id(source_id: str, url: str, title: str) -> str:
    hasher = hashlib.sha1(f"{source_id}|{url}|{title}".encode()).hexdigest()
    return hasher[:16]


def _fingerprint(title: str) -> str:
    norm = _WHITESPACE_RE.sub(" ", title.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def normalize(raw_articles: list[RawArticle]) -> list[Article]:
    """Convert RawArticle → Article, dropping entries missing required fields."""
    out: list[Article] = []
    for raw in raw_articles:
        title = _clean_html(raw.title)
        summary = _clean_html(raw.summary)
        url = _canonical_url(raw.url)
        if not title or not url:
            continue
        out.append(
            Article(
                id=_stable_id(raw.source_id, url, title),
                source_id=raw.source_id,
                source_name=raw.source_name,
                category=raw.category,
                title=title,
                url=url,
                summary=summary,
                published_at=raw.published_at or datetime.now(timezone.utc),
                author=raw.author,
                fingerprint=_fingerprint(title),
            )
        )
    return out
