#!/usr/bin/env python3
"""
Generate sources.json for the frontend "Our Sources" modal.

Reads the modular YAML source config via SourcesLoader and emits a compact
JSON document grouping every enabled source by category as {name, url} pairs.
This keeps the published source list always in sync with what the collector
actually pulls from. Run standalone (`python -m src.backend.api.generate_sources`)
to refresh the committed copy, or call write_sources_api() from the pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Make `src...` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.shared.config.sources_loader import get_sources_loader

logger = logging.getLogger(__name__)

# Tokens that should be fully upper-cased when humanizing a source id.
_ACRONYMS = {
    "ai", "ml", "api", "rss", "llm", "nlp", "gpu", "cpu", "ar", "vr",
    "nist", "nih", "darpa", "dsit", "fda", "eu", "uk", "us", "un",
    "aws", "ibm", "mit", "gcp", "hp", "hpc", "ieee", "acm", "acl",
    "cmu", "jair", "aaai", "bair", "cacm",
}
# Tokens with a specific preferred casing (overrides title-case and acronyms).
_SPECIAL_CASE = {
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "deepmind": "DeepMind",
    "huggingface": "HuggingFace",
    "openai": "OpenAI",
    "arxiv": "arXiv",
    "github": "GitHub",
    "youtube": "YouTube",
    "techcrunch": "TechCrunch",
    "venturebeat": "VentureBeat",
    "hackernoon": "HackerNoon",
    "aihub": "AIhub",
    "ml": "ML",
}


def humanize_source_name(source_id: str) -> str:
    """Turn a snake_case source id into a readable display name.

    e.g. "nist_ai_news" -> "NIST AI News", "openai_blog" -> "OpenAI Blog".
    """
    words = []
    for token in source_id.replace("-", "_").split("_"):
        if not token:
            continue
        low = token.lower()
        if low in _SPECIAL_CASE:
            words.append(_SPECIAL_CASE[low])
        elif low in _ACRONYMS:
            words.append(low.upper())
        else:
            words.append(token[:1].upper() + token[1:])
    return " ".join(words) if words else source_id


def build_sources_data() -> Dict[str, Any]:
    """Build the sources.json payload grouped by category."""
    loader = get_sources_loader()
    sources = loader.get_sources(enabled_only=True)

    by_category: Dict[str, List[Dict[str, str]]] = {}
    for source_id, config in sources.items():
        url = config.get("url")
        if not url:
            continue
        category = config.get("category", "Other")
        by_category.setdefault(category, []).append({
            "name": humanize_source_name(source_id),
            "url": url,
        })

    categories = []
    total = 0
    for category in sorted(by_category):
        entries = sorted(by_category[category], key=lambda s: s["name"].lower())
        total += len(entries)
        categories.append({"name": category, "sources": entries})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "categories": categories,
    }


def write_sources_api(*output_dirs: Path) -> Dict[str, Any]:
    """Write sources.json into each given directory; returns the payload."""
    data = build_sources_data()
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sources.json").write_text(payload + "\n", encoding="utf-8")
        logger.info(f"✅ Sources API saved: {output_dir / 'sources.json'} ({data['total']} sources)")
    return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frontend_api = _PROJECT_ROOT / "src" / "frontend" / "api"
    backend_api = _PROJECT_ROOT / "src" / "backend" / "api"
    data = write_sources_api(frontend_api, backend_api)
    print(f"Wrote {data['total']} sources across {len(data['categories'])} categories.")


if __name__ == "__main__":
    main()
