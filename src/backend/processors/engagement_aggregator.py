#!/usr/bin/env python3
"""Aggregate anonymous engagement events into ``engagement_weights.json``.

This is the *producer* side of the preference loop. It reads raw events from
Workers Analytics Engine (via the Cloudflare GraphQL Analytics API) over a
rolling window, derives **aggregates only** — never per-reader profiles — and
writes per-dimension multipliers consumed by the orchestrator at ranking time.

Privacy / safety properties:
  - k-anonymity: any bucket seen by fewer than ``--min-samples`` distinct
    readers (cids) is dropped to the neutral 1.0 multiplier. This protects
    privacy and suppresses noise/cold-start.
  - Clickbait-resistant: the engagement score weights *quality* signals (dwell,
    scroll depth, saves, net reactions, audio completion) above raw clicks.
  - Multipliers are centered on 1.0 and clamped to the same band the consumer
    enforces, so a runaway bucket can't dominate.

Usage:
  # Production: pull from Cloudflare (needs CF_ACCOUNT_ID + CF_API_TOKEN)
  python -m src.backend.processors.engagement_aggregator

  # Test/offline: aggregate a local JSONL file of client-shaped events
  python -m src.backend.processors.engagement_aggregator --input events.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # allow running both as a module and as a standalone file
    from src.backend.processors.engagement import MULT_MIN, MULT_MAX, DEFAULT_EPSILON
except ImportError:  # pragma: no cover - direct-file execution fallback
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "engagement", Path(__file__).with_name("engagement.py"))
    _eng = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eng)
    MULT_MIN, MULT_MAX, DEFAULT_EPSILON = _eng.MULT_MIN, _eng.MULT_MAX, _eng.DEFAULT_EPSILON

WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "shared" / "config" / "engagement_weights.json"
DATASET = os.getenv("CF_AE_DATASET", "dat_events")

# Quality-weighted blend. Clicks (open_rate) matter, but dwell/scroll/save/react
# matter more in aggregate so headline-bait can't win on clicks alone.
W_OPEN, W_DWELL, W_SCROLL, W_SAVE, W_REACT = 0.30, 0.20, 0.15, 0.15, 0.20
DWELL_CAP_MS = 30_000  # dwell normalized against 30s of consideration


def _clamp(v: float) -> float:
    return max(MULT_MIN, min(MULT_MAX, v))


def normalize_event(name: str, props: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a client/AE event into the uniform shape the aggregator uses."""
    return {
        "name": name,
        "section": props.get("section") or "",
        "source": props.get("source") or "",
        "archetype": props.get("archetype") or "",
        "cid": props.get("cid") or "",
        "value": props.get("value") or "",
        "tags": props.get("tags") or [],
        "ms": float(props.get("ms") or 0),
        "percent": float(props.get("percent") or 0),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
class _Bucket:
    __slots__ = ("cids", "imps", "opens", "dwell_ms", "scroll", "saves", "react_net", "audio")

    def __init__(self):
        self.cids = set()
        self.imps = 0
        self.opens = 0
        self.dwell_ms: List[float] = []
        self.scroll: List[float] = []
        self.saves = 0
        self.react_net = 0
        self.audio = 0

    def add(self, ev: Dict[str, Any]):
        if ev["cid"]:
            self.cids.add(ev["cid"])
        n = ev["name"]
        if n == "impression":
            self.imps += 1
        elif n == "article_open":
            self.opens += 1
        elif n == "dwell":
            self.dwell_ms.append(ev["ms"])
        elif n == "scroll_depth":
            self.scroll.append(ev["percent"])
        elif n == "save_toggle":
            self.saves += 1
        elif n == "reaction":
            self.react_net += 1 if ev["value"] == "up" else (-1 if ev["value"] == "down" else 0)
        elif n == "audio_complete":
            self.audio += 1

    def score(self) -> float:
        opens = max(1, self.opens)
        open_rate = self.opens / self.imps if self.imps else (1.0 if self.opens else 0.0)
        dwell_norm = (sum(self.dwell_ms) / len(self.dwell_ms) / DWELL_CAP_MS) if self.dwell_ms else 0.0
        scroll_norm = (sum(self.scroll) / len(self.scroll) / 100.0) if self.scroll else 0.0
        save_rate = self.saves / opens
        react_norm = (self.react_net / opens + 1) / 2  # map [-1,1] -> [0,1]
        return (
            W_OPEN * min(1.0, open_rate)
            + W_DWELL * min(1.0, dwell_norm)
            + W_SCROLL * min(1.0, scroll_norm)
            + W_SAVE * min(1.0, save_rate)
            + W_REACT * max(0.0, min(1.0, react_norm))
        )


def _dimension_weights(buckets: Dict[str, _Bucket], min_samples: int) -> Dict[str, float]:
    """Turn per-bucket scores into multipliers centered on 1.0, k-anonymized."""
    eligible = {k: b for k, b in buckets.items() if len(b.cids) >= min_samples and k}
    if not eligible:
        return {}
    scores = {k: b.score() for k, b in eligible.items()}
    mean = sum(scores.values()) / len(scores)
    if mean <= 0:
        return {}
    return {k: round(_clamp(s / mean), 4) for k, s in scores.items()}


def aggregate(events: Iterable[Dict[str, Any]], min_samples: int, window_days: int) -> Dict[str, Any]:
    dims = {d: defaultdict(_Bucket) for d in ("categories", "sources", "archetypes", "topics")}
    for ev in events:
        if ev["section"]:
            dims["categories"][ev["section"]].add(ev)
        if ev["source"]:
            dims["sources"][ev["source"]].add(ev)
        if ev["archetype"]:
            dims["archetypes"][ev["archetype"]].add(ev)
        for tag in ev["tags"]:
            dims["topics"][tag].add(ev)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "min_samples": min_samples,
        "exploration_epsilon": DEFAULT_EPSILON,
        "categories": _dimension_weights(dims["categories"], min_samples),
        "sources": _dimension_weights(dims["sources"], min_samples),
        "topics": _dimension_weights(dims["topics"], min_samples),
        "archetypes": _dimension_weights(dims["archetypes"], min_samples),
    }


# --------------------------------------------------------------------------- #
# Sources of events
# --------------------------------------------------------------------------- #
def read_local(path: Path) -> List[Dict[str, Any]]:
    """Read client-shaped events from a JSONL file: {"name":..,"properties":{..}}."""
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        events.append(normalize_event(obj.get("name", ""), obj.get("properties", {})))
    return events


def fetch_cloudflare(window_days: int) -> List[Dict[str, Any]]:
    """Pull recent events from Workers Analytics Engine via the GraphQL API.

    Returns [] (and logs) when credentials are absent or the query fails, so the
    caller leaves the existing weights file untouched rather than zeroing it.
    """
    account = os.getenv("CF_ACCOUNT_ID")
    token = os.getenv("CF_API_TOKEN")
    if not account or not token:
        print("[aggregator] CF_ACCOUNT_ID / CF_API_TOKEN not set; skipping fetch", file=sys.stderr)
        return []
    try:
        import urllib.request

        # AE exposes each dataset as a table in the GraphQL Analytics API. We map
        # the Worker's writeDataPoint layout back to named fields:
        #   blob1=name blob2=section blob3=source blob4=archetype
        #   blob6=reaction value blob8=cid ; double1=ms double3=percent
        since = (datetime.now(timezone.utc)).isoformat()
        query = {
            "query": f"""
            query {{
              viewer {{
                accounts(filter: {{accountTag: \"{account}\"}}) {{
                  events: {DATASET}(limit: 10000, filter: {{
                    datetime_geq: \"{_window_start(window_days)}\",
                    datetime_leq: \"{since}\"
                  }}) {{
                    dimensions {{ blob1 blob2 blob3 blob4 blob6 blob8 }}
                    sum {{ double1 double3 }}
                  }}
                }}
              }}
            }}"""
        }
        req = urllib.request.Request(
            "https://api.cloudflare.com/client/v4/graphql",
            data=json.dumps(query).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        rows = (data.get("data", {}).get("viewer", {}).get("accounts", [{}])[0] or {}).get("events", [])
        events = []
        for row in rows:
            d = row.get("dimensions", {})
            s = row.get("sum", {})
            events.append(normalize_event(d.get("blob1", ""), {
                "section": d.get("blob2"), "source": d.get("blob3"),
                "archetype": d.get("blob4"), "value": d.get("blob6"),
                "cid": d.get("blob8"), "ms": s.get("double1"), "percent": s.get("double3"),
            }))
        print(f"[aggregator] fetched {len(events)} event rows from Cloudflare")
        return events
    except Exception as e:  # network/schema issues must never break the pipeline
        print(f"[aggregator] Cloudflare fetch failed ({e}); skipping", file=sys.stderr)
        return []


def _window_start(window_days: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local JSONL of client events (offline mode)")
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--window-days", type=int, default=int(os.getenv("ENGAGEMENT_WINDOW_DAYS", "7")))
    parser.add_argument("--min-samples", type=int, default=int(os.getenv("ENGAGEMENT_MIN_SAMPLES", "20")))
    parser.add_argument("--dry-run", action="store_true", help="Print weights, don't write file")
    args = parser.parse_args(argv)

    if args.input:
        events = read_local(args.input)
    else:
        events = fetch_cloudflare(args.window_days)

    if not events:
        print("[aggregator] no events; leaving existing weights untouched")
        return 0

    weights = aggregate(events, args.min_samples, args.window_days)
    text = json.dumps(weights, indent=2) + "\n"
    if args.dry_run:
        print(text)
    else:
        args.output.write_text(text)
        total = sum(len(weights[k]) for k in ("categories", "sources", "topics", "archetypes"))
        print(f"[aggregator] wrote {args.output} ({total} weighted buckets, "
              f"window={args.window_days}d, min_samples={args.min_samples})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
