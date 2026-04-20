"""Pipeline stage unit tests (collect is network, so only normalize+dedupe+publish here)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.dedupe import dedupe
from src.pipeline.models import ApiBundle, Article, ScoredArticle
from src.pipeline.normalize import normalize
from src.pipeline.publish import publish


def test_normalize_strips_tags_and_canonicalizes(raw_articles):
    articles = normalize(raw_articles)
    assert len(articles) == len(raw_articles)
    verge = next(a for a in articles if "theverge" in a.url)
    assert "utm_source" not in verge.url
    for article in articles:
        assert "<" not in article.summary
        assert article.id and len(article.id) == 16


def test_dedupe_drops_same_canonical_url(raw_articles):
    articles = normalize(raw_articles)
    assert len(articles) == 6
    deduped = dedupe(articles)
    assert len(deduped) == 5
    verge_matches = [a for a in deduped if "theverge" in a.url]
    assert len(verge_matches) == 1


def test_publish_writes_manifest_and_categories(tmp_path, raw_articles):
    articles = dedupe(normalize(raw_articles))
    scored = [
        ScoredArticle(article=a, score=0.9 - i * 0.1, confidence=0.8, categories=[a.category])
        for i, a in enumerate(articles)
    ]
    bundle = ApiBundle(
        latest=scored,
        widget=scored[:3],
        categories={"research": [s for s in scored if s.article.category == "research"]},
        stats={"total_articles": len(scored), "version": "2.0"},
    )
    digests = publish(bundle, output_dir=tmp_path)
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "widget.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "categories" / "research.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "2.0"
    assert set(manifest["artifacts"].keys()).issuperset({"latest.json", "widget.json"})
    assert "latest.json" in digests


def test_scored_article_roundtrip():
    article = Article(
        id="abc1234567890def",
        source_id="s",
        source_name="S",
        category="research",
        title="T",
        url="https://example.com/x",
        summary="sum",
        published_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        fingerprint="deadbeefdeadbeef",
    )
    scored = ScoredArticle(article=article, score=0.5, confidence=0.9, categories=["research"])
    payload = scored.model_dump(mode="json")
    restored = ScoredArticle.model_validate(payload)
    assert restored.article.id == article.id
