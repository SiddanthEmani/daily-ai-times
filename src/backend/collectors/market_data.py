"""Real market quotes for the frontend ticker, sourced from Stooq's free,
keyless CSV endpoint (no API key required, unlike Alpha Vantage/IEX/etc.)."""

import logging
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger(__name__)

STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"

TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AMD", "TSM", "AMZN", "AAPL"]


def _parse_last_two_closes(csv_text: str) -> List[float]:
    """Parse a Stooq daily CSV (Date,Open,High,Low,Close,Volume) and return
    the last two closing prices in chronological order."""
    lines = [line for line in csv_text.strip().splitlines() if line.strip()]
    if len(lines) < 3:  # header + at least 2 data rows
        return []
    closes = []
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) < 5:
            continue
        try:
            closes.append(float(fields[4]))
        except ValueError:
            continue
    return closes[-2:]


async def _fetch_ticker(session: aiohttp.ClientSession, ticker: str) -> Dict[str, Any]:
    url = STOOQ_DAILY_URL.format(symbol=ticker.lower())
    async with session.get(url) as response:
        response.raise_for_status()
        text = await response.text()
    closes = _parse_last_two_closes(text)
    if len(closes) < 2:
        raise ValueError(f"Not enough Stooq data for {ticker}")
    prev_close, last_close = closes[0], closes[1]
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
    return {
        "ticker": ticker,
        "px": f"{last_close:,.2f}",
        "ch": f"{change_pct:+.1f}%",
        "up": change_pct >= 0,
    }


async def fetch_market_quotes(tickers: List[str] = TICKERS) -> List[Dict[str, Any]]:
    """Fetch real daily quotes for the given tickers. Tickers that fail to
    fetch/parse are silently skipped rather than failing the whole batch."""
    timeout = aiohttp.ClientTimeout(total=15)
    quotes: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for ticker in tickers:
            try:
                quotes.append(await _fetch_ticker(session, ticker))
            except Exception as e:
                logger.warning(f"Market data fetch failed for {ticker}: {e}")
    return quotes
