"""Dedupe stage: drop exact URL matches and near-duplicate titles."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.pipeline.models import Article

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "is",
    "are",
}


def _title_tokens(title: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(title.lower()) if t not in _STOPWORDS and len(t) > 2}


def _similar(a: Article, b: Article, threshold: float) -> bool:
    """Jaccard over content tokens with a SequenceMatcher tiebreaker."""
    ta, tb = _title_tokens(a.title), _title_tokens(b.title)
    if not ta or not tb:
        return False
    jaccard = len(ta & tb) / len(ta | tb)
    if jaccard >= threshold:
        return True
    if jaccard >= threshold * 0.6:
        return SequenceMatcher(None, a.title.lower(), b.title.lower()).ratio() >= threshold
    return False


def dedupe(articles: list[Article], *, title_threshold: float = 0.75) -> list[Article]:
    """Return articles deduplicated by canonical URL and title similarity."""
    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    survivors: list[Article] = []

    for article in sorted(articles, key=lambda a: a.published_at, reverse=True):
        if article.url in seen_urls or article.fingerprint in seen_fingerprints:
            continue
        duplicate = False
        for kept in survivors:
            if _similar(article, kept, title_threshold):
                duplicate = True
                break
        if duplicate:
            continue
        seen_urls.add(article.url)
        seen_fingerprints.add(article.fingerprint)
        survivors.append(article)

    return survivors
