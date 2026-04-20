"""Publish + verification MCP tools."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
from claude_agent_sdk import tool

from src.pipeline.models import ApiBundle
from src.pipeline.publish import publish as publish_stage
from src.tools.state import get_state

_FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "src" / "frontend" / "api"


@tool(
    "publish_v2",
    "Write state['latest'] + state['widget'] + state['category_rankings'] to "
    "src/frontend/api/v2/**.json via atomic replace. Returns sha256 digests.",
    {},
)
async def publish_v2(_: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    if not state.latest:
        return {
            "content": [{"type": "text", "text": "no latest headlines; run pick_headlines first"}],
            "is_error": True,
        }
    stats = dict(state.stats)
    stats.setdefault("total_articles", len(state.classified or state.scored or []))
    stats.setdefault("schema_version", "2.0")
    bundle = ApiBundle(
        latest=state.latest,
        widget=state.widget,
        categories=state.category_rankings,
        stats=stats,
    )
    digests = publish_stage(bundle)
    return {"content": [{"type": "text", "text": json.dumps({"digests": digests}, indent=2)}]}


@tool(
    "validate_api",
    "Validate that src/frontend/api/v2/*.json conforms to the v2 contract. "
    "Checks manifest, required files, schema_version, and that counts match.",
    {},
)
async def validate_api(_: dict[str, Any]) -> dict[str, Any]:
    v2 = _FRONTEND_ROOT / "v2"
    required = ["latest.json", "widget.json", "stats.json", "manifest.json"]
    missing = [name for name in required if not (v2 / name).exists()]
    if missing:
        return {
            "content": [{"type": "text", "text": f"missing required files: {missing}"}],
            "is_error": True,
        }
    manifest = orjson.loads((v2 / "manifest.json").read_bytes())
    errors: list[str] = []
    if manifest.get("schema_version") != "2.0":
        errors.append(f"schema_version={manifest.get('schema_version')} != 2.0")
    for name in required[:-1]:
        if name not in manifest.get("artifacts", {}):
            errors.append(f"manifest missing artifact {name}")
    for name, count in manifest.get("counts", {}).get("categories", {}).items():
        cat_file = v2 / "categories" / f"{name}.json"
        if not cat_file.exists():
            errors.append(f"category file missing: {name}")
            continue
        actual = len(orjson.loads(cat_file.read_bytes()).get("articles", []))
        if actual != count:
            errors.append(f"category {name} count {actual} != manifest {count}")
    if errors:
        return {
            "content": [{"type": "text", "text": "validation failed:\n" + "\n".join(errors)}],
            "is_error": True,
        }
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {"status": "ok", "manifest": manifest, "verified_at": datetime.now(timezone.utc).isoformat()},
                    indent=2,
                ),
            }
        ]
    }


@tool(
    "parity_check",
    "Diff src/frontend/api/latest.json (v1) against src/frontend/api/v2/latest.json. "
    "Reports overlap of URLs and any schema drift.",
    {},
)
async def parity_check(_: dict[str, Any]) -> dict[str, Any]:
    v1 = _FRONTEND_ROOT / "latest.json"
    v2 = _FRONTEND_ROOT / "v2" / "latest.json"
    if not v1.exists() or not v2.exists():
        return {
            "content": [
                {"type": "text", "text": f"missing v1={v1.exists()} v2={v2.exists()}"}
            ],
            "is_error": True,
        }
    v1_data = orjson.loads(v1.read_bytes())
    v2_data = orjson.loads(v2.read_bytes())
    v1_urls = {a.get("url") or a.get("link") for a in v1_data.get("articles", [])}
    v2_urls = {a["url"] for a in v2_data.get("articles", [])}
    overlap = v1_urls & v2_urls
    payload = {
        "v1_count": len(v1_urls),
        "v2_count": len(v2_urls),
        "overlap": len(overlap),
        "v2_only": len(v2_urls - v1_urls),
        "v1_only": len(v1_urls - v2_urls),
    }
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}
