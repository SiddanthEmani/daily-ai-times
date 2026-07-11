#!/usr/bin/env python3
"""
Ticker Collector - Fetches live market quotes from Alpha Vantage for the
masthead ticker bar and writes src/frontend/api/ticker.json.

Standalone (stdlib-only) so it can run in a lightweight, isolated workflow
without installing the full news-pipeline dependency set. Runs on a schedule
well within Alpha Vantage's free-tier limits (25 requests/day, 5/minute).
"""

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SYMBOLS = ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMD', 'TSM', 'AMZN', 'AAPL']
ALPHA_VANTAGE_URL = 'https://www.alphavantage.co/query'
REQUEST_SPACING_SECONDS = 13  # keeps 8 calls under the 5-requests/minute cap
OUTPUT_PATH = Path(__file__).resolve().parents[3] / 'src' / 'frontend' / 'api' / 'ticker.json'


def fetch_quote(symbol: str, api_key: str) -> dict | None:
    url = f'{ALPHA_VANTAGE_URL}?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        logger.warning('Failed to fetch %s: %s', symbol, err)
        return None

    if 'Note' in payload or 'Information' in payload:
        logger.warning('Alpha Vantage rate limit/notice for %s: %s',
                        symbol, payload.get('Note') or payload.get('Information'))
        return None

    quote = payload.get('Global Quote') or {}
    price = quote.get('05. price')
    change_percent = quote.get('10. change percent')
    if not price or not change_percent:
        logger.warning('Incomplete quote for %s: %s', symbol, quote)
        return None

    pct_value = change_percent.rstrip('%')
    try:
        up = float(pct_value) >= 0
    except ValueError:
        up = not change_percent.strip().startswith('-')

    return {
        'symbol': symbol,
        'price': f'{float(price):,.2f}',
        'change_percent': f'{"+" if up else ""}{pct_value}%',
        'up': up,
    }


def main() -> int:
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.warning('ALPHA_VANTAGE_API_KEY not set - skipping ticker update')
        return 0

    quotes = []
    for i, symbol in enumerate(SYMBOLS):
        quote = fetch_quote(symbol, api_key)
        if quote:
            quotes.append(quote)
        if i < len(SYMBOLS) - 1:
            time.sleep(REQUEST_SPACING_SECONDS)

    if not quotes:
        logger.warning('No quotes fetched successfully - leaving existing ticker.json untouched')
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'quotes': quotes,
    }, indent=2) + '\n')

    logger.info('Wrote %d/%d quotes to %s', len(quotes), len(SYMBOLS), OUTPUT_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
