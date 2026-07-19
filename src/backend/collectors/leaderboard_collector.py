#!/usr/bin/env python3
"""
Leaderboard Collector - Fetches the AI-model benchmark leaderboard from the
Artificial Analysis official API and writes src/frontend/api/leaderboard.json,
which powers the "Benchmark Leaderboard" chart in the frontend sidebar.

Standalone (stdlib-only) so it can run in a lightweight, isolated workflow
without installing the full news-pipeline dependency set - mirrors the design
of ticker_collector.py. Runs on a daily schedule (the leaderboard changes
slowly) and stays well within the Artificial Analysis free-tier limits.

Requires the ARTIFICIAL_ANALYSIS_API_KEY environment variable (a free key from
https://artificialanalysis.ai/api). When unset the collector no-ops and the
frontend chart falls back to its static built-in figures.
"""

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Official Artificial Analysis API - list of LLM models with evaluations,
# performance and pricing. Authenticated via the x-api-key header.
MODELS_URL = 'https://artificialanalysis.ai/api/v2/data/llms/models'

# Candidate keys for the score that powers the chart, tried in order. The
# Artificial Analysis Intelligence Index is the headline composite score; the
# extra spellings guard against minor schema drift so a rename doesn't silently
# blank the chart. Swap this list (and CHART_CHIP / CHART_METRIC below) to power
# the chart off a different metric.
METRIC_KEYS = (
    'artificial_analysis_intelligence_index',
    'artificialAnalysisIntelligenceIndex',
    'intelligence_index',
)
CHART_METRIC = 'Artificial Analysis Intelligence Index'
CHART_CHIP = 'INTELLIGENCE INDEX'

# Keys the score may live under directly on the model or nested in a block.
EVALUATION_CONTAINER_KEYS = ('evaluations', 'evaluation', 'benchmarks')
NAME_KEYS = ('name', 'model_name', 'label')
CREATOR_KEYS = ('model_creator', 'creator', 'organization', 'provider')

MAX_MODELS = 5  # Compact sidebar chart - top N models by score.
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / 'src' / 'frontend' / 'api' / 'leaderboard.json'
)


def _first(mapping: dict, keys) -> object:
    """Return the first present, non-None value among candidate keys."""
    for key in keys:
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return mapping[key]
    return None


def _extract_score(model: dict) -> float | None:
    """Pull the intelligence-index score from a model record, tolerating
    whether it sits directly on the model or nested in an evaluations block."""
    raw = _first(model, METRIC_KEYS)
    if raw is None:
        for container_key in EVALUATION_CONTAINER_KEYS:
            container = model.get(container_key)
            if isinstance(container, dict):
                raw = _first(container, METRIC_KEYS)
                if raw is not None:
                    break
    if raw is None:
        return None
    try:
        return round(float(raw), 1)
    except (TypeError, ValueError):
        return None


def _extract_creator(model: dict) -> str | None:
    creator = _first(model, CREATOR_KEYS)
    if isinstance(creator, dict):
        creator = _first(creator, NAME_KEYS)
    return creator if isinstance(creator, str) and creator.strip() else None


def fetch_models(api_key: str) -> list | None:
    """GET the models list from the Artificial Analysis API. Returns the raw
    list of model records, or None on any failure."""
    request = urllib.request.Request(
        MODELS_URL,
        headers={'x-api-key': api_key, 'User-Agent': USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        logger.warning('Failed to fetch leaderboard: %s', err)
        return None

    # The API wraps the list under "data"; accept a bare list defensively.
    if isinstance(payload, dict):
        models = payload.get('data') or payload.get('models') or payload.get('results')
    else:
        models = payload
    if not isinstance(models, list):
        logger.warning('Unexpected leaderboard payload shape: %s', type(payload).__name__)
        return None
    return models


def build_rows(models: list) -> list:
    """Transform raw model records into the chart's {label, value, creator}
    rows, sorted by score descending and capped at MAX_MODELS."""
    rows = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = _first(model, NAME_KEYS)
        score = _extract_score(model)
        if not isinstance(name, str) or not name.strip() or score is None:
            continue
        row = {'label': name.strip(), 'value': score}
        creator = _extract_creator(model)
        if creator:
            row['creator'] = creator
        rows.append(row)

    rows.sort(key=lambda r: r['value'], reverse=True)
    return rows[:MAX_MODELS]


def main() -> int:
    api_key = os.getenv('ARTIFICIAL_ANALYSIS_API_KEY')
    if not api_key:
        logger.warning('ARTIFICIAL_ANALYSIS_API_KEY not set - skipping leaderboard update')
        return 0

    models = fetch_models(api_key)
    if not models:
        logger.warning('No leaderboard data fetched - leaving existing leaderboard.json untouched')
        return 0

    rows = build_rows(models)
    if not rows:
        logger.warning('No models parsed from leaderboard payload - leaving existing leaderboard.json untouched')
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'metric': CHART_METRIC,
        'chip': CHART_CHIP,
        'models': rows,
    }, indent=2) + '\n')

    logger.info('Wrote %d models to %s', len(rows), OUTPUT_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
