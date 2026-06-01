"""Reader-engagement feedback for article ranking.

This is the *consumer* side of the anonymous preference loop. It loads the
aggregated, k-anonymized ``engagement_weights.json`` produced by
``engagement_aggregator.py`` and turns it into a bounded per-article multiplier
that the orchestrator applies at ranking time.

Design guardrails (see plan):
  - Engagement only re-orders already-quality-approved articles. The multiplier
    is clamped to a narrow band so popular topics get nudged, never allowed to
    dominate the editorial score.
  - Cold-start safe: if the file is missing/empty or a dimension has no signal,
    the multiplier is exactly 1.0 (today's behavior).
  - Feature-flagged: set ENGAGEMENT_LOOP_ENABLED=0 to disable instantly.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Bounded band: engagement can move an article's score by at most ±35% / -25%.
MULT_MIN = 0.75
MULT_MAX = 1.35

DEFAULT_EPSILON = 0.15

_DEFAULT_WEIGHTS: Dict[str, Any] = {
    "generated_at": None,
    "window_days": 7,
    "min_samples": 20,
    "exploration_epsilon": DEFAULT_EPSILON,
    "categories": {},
    "sources": {},
    "topics": {},
    "archetypes": {},
}

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "shared" / "config" / "engagement_weights.json"
)


def is_enabled() -> bool:
    """Feature flag. Defaults on; cold-start neutral weights make it a no-op."""
    return os.getenv("ENGAGEMENT_LOOP_ENABLED", "1").strip() not in ("0", "false", "False", "")


def load_weights(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load engagement weights, falling back to neutral defaults on any problem."""
    p = Path(path) if path else _CONFIG_PATH
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return dict(_DEFAULT_WEIGHTS)
        merged = dict(_DEFAULT_WEIGHTS)
        merged.update(data)
        for key in ("categories", "sources", "topics", "archetypes"):
            if not isinstance(merged.get(key), dict):
                merged[key] = {}
        return merged
    except (OSError, ValueError, json.JSONDecodeError):
        return dict(_DEFAULT_WEIGHTS)


def _archetype_of(article: Dict[str, Any]) -> str:
    atype = (article.get("article_type") or "").strip().lower()
    if atype in ("headline", "research", "article"):
        return atype
    category = (article.get("category") or "").strip().lower()
    if "research" in category:
        return "research"
    return "article"


def _clamp(value: float) -> float:
    return max(MULT_MIN, min(MULT_MAX, value))


def engagement_multiplier(article: Dict[str, Any], weights: Dict[str, Any]) -> float:
    """Bounded multiplier in [MULT_MIN, MULT_MAX] from reader engagement.

    Combines the available dimension weights (category, source, topics,
    archetype) via their geometric mean so multiple signals blend smoothly
    around 1.0 rather than compounding, then clamps to the band. Dimensions with
    no signal contribute nothing (the loop stays neutral until data accrues).
    """
    factors = []

    cat = article.get("category")
    if cat and cat in weights.get("categories", {}):
        factors.append(float(weights["categories"][cat]))

    src = article.get("source")
    if src and src in weights.get("sources", {}):
        factors.append(float(weights["sources"][src]))

    arch = _archetype_of(article)
    if arch in weights.get("archetypes", {}):
        factors.append(float(weights["archetypes"][arch]))

    topic_weights = weights.get("topics", {})
    if topic_weights:
        for tag in article.get("tags", []) or []:
            if tag in topic_weights:
                factors.append(float(topic_weights[tag]))

    factors = [f for f in factors if f and f > 0]
    if not factors:
        return 1.0

    geo_mean = math.exp(sum(math.log(f) for f in factors) / len(factors))
    return _clamp(geo_mean)


def exploration_epsilon(weights: Dict[str, Any]) -> float:
    """Fraction of slots reserved for high-novelty exploration (serendipity)."""
    try:
        eps = float(weights.get("exploration_epsilon", DEFAULT_EPSILON))
    except (TypeError, ValueError):
        eps = DEFAULT_EPSILON
    return max(0.0, min(0.5, eps))


if __name__ == "__main__":
    # Quick self-test: neutral file → 1.0; synthetic weights → clamped band.
    w = load_weights()
    assert engagement_multiplier({"category": "Industry", "source": "x"}, w) == 1.0
    synthetic = dict(_DEFAULT_WEIGHTS)
    synthetic["categories"] = {"Research": 5.0}  # absurd, must clamp
    synthetic["sources"] = {"arxiv": 1.2}
    m = engagement_multiplier({"category": "Research", "source": "arxiv"}, synthetic)
    assert MULT_MIN <= m <= MULT_MAX, m
    print(f"engagement self-test ok: neutral=1.0, clamped={m:.3f}, enabled={is_enabled()}")
