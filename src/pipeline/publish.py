"""Publish stage: atomically write v2 API files to src/frontend/api/v2/."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import orjson

from src.pipeline.models import ApiBundle, ScoredArticle

log = logging.getLogger(__name__)

_FRONTEND_V2 = Path(__file__).resolve().parents[2] / "src" / "frontend" / "api" / "v2"


def _serialize(scored: ScoredArticle) -> dict:
    article = scored.article
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "summary": article.summary,
        "category": article.category,
        "source": article.source_name,
        "source_id": article.source_id,
        "author": article.author,
        "published_at": article.published_at.isoformat(),
        "score": round(scored.score, 4),
        "confidence": round(scored.confidence, 4),
        "categories": scored.categories,
        "model_used": scored.model_used,
    }


def _atomic_write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    digest = hashlib.sha256(blob).hexdigest()[:16]
    path.with_suffix(path.suffix + ".hash").write_text(digest + "\n")
    return digest


def publish(
    bundle: ApiBundle,
    *,
    output_dir: Path = _FRONTEND_V2,
) -> dict[str, str]:
    """Write latest.json, widget.json, categories/*.json, manifest.json.

    Returns {filename: sha256_hex}.
    """
    digests: dict[str, str] = {}
    generated = bundle.generated_at.astimezone(timezone.utc).isoformat()

    latest_path = output_dir / "latest.json"
    digests["latest.json"] = _atomic_write(
        latest_path,
        {
            "generated_at": generated,
            "articles": [_serialize(s) for s in bundle.latest],
            "stats": bundle.stats,
        },
    )

    widget_path = output_dir / "widget.json"
    digests["widget.json"] = _atomic_write(
        widget_path,
        {
            "generated_at": generated,
            "articles": [_serialize(s) for s in bundle.widget],
        },
    )

    stats_path = output_dir / "stats.json"
    digests["stats.json"] = _atomic_write(
        stats_path,
        {
            "generated_at": generated,
            **bundle.stats,
        },
    )

    categories_dir = output_dir / "categories"
    for category, articles in bundle.categories.items():
        cat_path = categories_dir / f"{category}.json"
        digests[f"categories/{category}.json"] = _atomic_write(
            cat_path,
            {
                "generated_at": generated,
                "category": category,
                "articles": [_serialize(s) for s in articles],
            },
        )

    manifest_path = output_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        {
            "generated_at": generated,
            "schema_version": "2.0",
            "artifacts": digests,
            "counts": {
                "latest": len(bundle.latest),
                "widget": len(bundle.widget),
                "categories": {k: len(v) for k, v in bundle.categories.items()},
            },
        },
    )

    log.info("published %d files to %s", len(digests) + 1, output_dir)
    return digests
