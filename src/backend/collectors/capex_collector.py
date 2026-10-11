#!/usr/bin/env python3
"""
Capex Collector - Builds the "AI Data Center Buildout" sidebar chart data and
writes src/frontend/api/capex.json.

The chart tracks annualized data center / AI-infrastructure spend ($B) for the
companies actually building or buying data-center capacity. Two data sources are
merged, because no single free API exposes this cleanly:

  * Public companies (Amazon, Microsoft, Alphabet, Meta, Oracle) - fetched live
    and keyless from the SEC EDGAR XBRL "companyconcept" API. The displayed value
    is trailing-12-month capital expenditure (latest fiscal year + current
    year-to-date - prior-year year-to-date, since cash-flow filings are
    cumulative). SEC filings do not break out data-center-only capex, so this is
    total company capex, which for these firms is overwhelmingly AI/data-center
    spend today.

  * Private companies (OpenAI, Anthropic) - have no SEC filings, so their figures
    come from a committed, human-curated file (capex_curated.json) with a source
    URL and as_of date, and are flagged estimated=true in the output.

The roster, CIKs, and private figures all live in capex_curated.json. A curator
may also add an explicit value/estimated/source/basis to a public company there
to override its live SEC figure (e.g. to show a data-center-only estimate).

Standalone (stdlib-only) so it can run in a lightweight, isolated workflow -
mirrors leaderboard_collector.py and ticker_collector.py. Needs no API key; SEC
only asks for a descriptive User-Agent. Runs weekly (SEC filings are quarterly
and curated edits are occasional). On failure it leaves the existing capex.json
untouched rather than emitting an empty chart.
"""

import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# SEC EDGAR XBRL "companyconcept" API - all filed values for one us-gaap concept.
# Keyless, but SEC requires a User-Agent identifying the caller.
SEC_CONCEPT_URL = (
    'https://data.sec.gov/api/xbrl/companyconcept/'
    'CIK{cik}/us-gaap/{tag}.json'
)
USER_AGENT = 'DailyAITimes/1.0 (+https://www.dailyai.wtf)'
REQUEST_TIMEOUT_SECONDS = 20

# Capex XBRL tags, tried in order. Most hyperscalers tag capex as
# PaymentsToAcquirePropertyPlantAndEquipment; Amazon uses the broader
# PaymentsToAcquireProductiveAssets. Trying both lets one company map cover
# minor schema differences.
CAPEX_TAGS = (
    'PaymentsToAcquirePropertyPlantAndEquipment',
    'PaymentsToAcquireProductiveAssets',
)

# Cash-flow statements are cumulative: a 10-Q reports year-to-date capex (3-, 6-
# or 9-month spans) and the 10-K reports the full year. Only Q1 ever appears as
# a standalone ~3-month figure, so summing "the last four 3-month datapoints"
# would add up Q1s from four different years. Instead TTM is derived as
#   latest full year + current YTD - prior-year YTD of the same length.
ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 350, 380
# Tolerance when matching period boundaries across fiscal calendars (52/53-week
# years, month-end drift).
MATCH_TOLERANCE_DAYS = 10
# A series whose newest datapoint is older than this is a tag the company has
# stopped using (e.g. Amazon's PaymentsToAcquirePropertyPlantAndEquipment, last
# filed years ago) and must not be shown as current spend.
MAX_STALENESS_DAYS = 200

BILLION = 1_000_000_000

METRIC = 'Annualized data center / AI-infrastructure spend'
CHIP = '$B / YR'

CURATED_PATH = Path(__file__).resolve().parent / 'capex_curated.json'
OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / 'src' / 'frontend' / 'api' / 'capex.json'
)


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except (TypeError, ValueError):
        return None


def _fetch_concept(cik: str, tag: str) -> dict | None:
    """GET one companyconcept document, or None on any failure."""
    url = SEC_CONCEPT_URL.format(cik=cik, tag=tag)
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as err:
        # 404 simply means this company doesn't use this tag - expected while
        # trying the fallback tag, so log quietly.
        logger.info('SEC %s/%s -> HTTP %s', cik, tag, err.code)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        logger.warning('SEC fetch failed for %s/%s: %s', cik, tag, err)
        return None


def _usd_entries(document: dict) -> list:
    """Pull the USD-denominated datapoints out of a companyconcept document."""
    units = (document or {}).get('units')
    if not isinstance(units, dict):
        return []
    entries = units.get('USD')
    return entries if isinstance(entries, list) else []


def _periods(entries: list) -> dict:
    """Map (start, end) -> value for well-formed USD datapoints, latest filing
    winning when the same period is reported more than once (restatements)."""
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start, end = _parse_date(entry.get('start')), _parse_date(entry.get('end'))
        val = entry.get('val')
        if start is None or end is None or end <= start or not isinstance(val, (int, float)):
            continue
        rows.append((str(entry.get('filed') or ''), start, end, abs(float(val))))
    periods: dict[tuple[date, date], float] = {}
    for _, start, end, val in sorted(rows):
        periods[(start, end)] = val
    return periods


def _near(a: date, b: date) -> bool:
    return abs((a - b).days) <= MATCH_TOLERANCE_DAYS


def _ttm_capex(entries: list, today: date | None = None) -> tuple[float, date] | None:
    """Trailing-12-month capex in $B plus the period end it is current to, or
    None when the series is empty or stale.

    If the newest period is a full fiscal year, that is the TTM. Otherwise the
    newest period is a year-to-date span and TTM = previous full year + this
    YTD - the same YTD span one year earlier. Falls back to the latest full
    year when the prior-year YTD comparative can't be found."""
    today = today or date.today()
    periods = _periods(entries)
    if not periods:
        return None
    latest_end = max(end for _, end in periods)
    if (today - latest_end).days > MAX_STALENESS_DAYS:
        return None

    annuals = sorted(
        ((start, end, val) for (start, end), val in periods.items()
         if ANNUAL_MIN_DAYS <= (end - start).days <= ANNUAL_MAX_DAYS),
        key=lambda r: r[1], reverse=True,
    )
    if annuals and _near(annuals[0][1], latest_end):
        return round(annuals[0][2] / BILLION, 1), annuals[0][1]

    # Newest YTD period: the longest sub-annual span ending at latest_end.
    ytd_candidates = [
        (start, val) for (start, end), val in periods.items()
        if end == latest_end and (end - start).days < ANNUAL_MIN_DAYS
    ]
    if ytd_candidates:
        ytd_start, ytd_val = min(ytd_candidates, key=lambda r: r[0])
        span = (latest_end - ytd_start).days
        prior_annual = next(
            (val for start, end, val in annuals if _near(end, ytd_start - timedelta(days=1))),
            None,
        )
        prior_ytd = next(
            (val for (start, end), val in periods.items()
             if _near(end, latest_end - timedelta(days=365))
             and abs((end - start).days - span) <= MATCH_TOLERANCE_DAYS),
            None,
        )
        if prior_annual is not None and prior_ytd is not None:
            return round((prior_annual + ytd_val - prior_ytd) / BILLION, 1), latest_end

    if annuals and (today - annuals[0][1]).days <= MAX_STALENESS_DAYS + 180:
        return round(annuals[0][2] / BILLION, 1), annuals[0][1]
    return None


def _public_row(company: dict) -> dict | None:
    """Resolve one public company's live SEC capex into a chart row, or None if
    no capex could be fetched."""
    cik = str(company.get('cik', '')).zfill(10)
    if not cik.strip('0'):
        logger.warning('Public company %s missing CIK - skipping', company.get('label'))
        return None
    # Try every tag and keep the freshest series: companies switch tags over
    # time and the abandoned one still returns (stale) historical data.
    best = None
    for tag in CAPEX_TAGS:
        document = _fetch_concept(cik, tag)
        result = _ttm_capex(_usd_entries(document)) if document else None
        if result is not None and (best is None or result[1] > best[1]):
            best = result
    if best is not None:
        value, as_of = best
        return {
            'label': company['label'],
            'value': value,
            'estimated': False,
            'source': 'SEC EDGAR 10-Q/10-K',
            'basis': 'trailing-12-mo capex',
            'as_of': as_of.isoformat(),
        }
    logger.warning('No capex resolved for %s (CIK %s)', company.get('label'), cik)
    return None


def _curated_row(company: dict) -> dict | None:
    """Build a chart row straight from curated fields (private companies, or a
    public company whose live figure is being overridden)."""
    value = company.get('value')
    if not isinstance(value, (int, float)):
        logger.warning('Curated company %s missing numeric value - skipping', company.get('label'))
        return None
    row = {
        'label': company['label'],
        'value': round(float(value), 1),
        'estimated': bool(company.get('estimated', True)),
    }
    for key in ('source', 'basis', 'as_of'):
        if company.get(key):
            row[key] = company[key]
    return row


def build_rows(curated: dict) -> list:
    """Merge live SEC data with curated figures into sorted chart rows.

    A curated explicit `value` always wins (the data-center-only / override
    lever). Public companies without an override are fetched live from SEC.
    Private companies are curated-only."""
    companies = curated.get('companies')
    if not isinstance(companies, list):
        logger.warning('Curated file has no "companies" list')
        return []

    rows = []
    for company in companies:
        if not isinstance(company, dict) or not company.get('label'):
            continue
        has_override = isinstance(company.get('value'), (int, float))
        if company.get('public') and not has_override:
            row = _public_row(company)
        else:
            row = _curated_row(company)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: r['value'], reverse=True)
    return rows


def main() -> int:
    try:
        curated = json.loads(CURATED_PATH.read_text())
    except (OSError, json.JSONDecodeError) as err:
        logger.error('Could not read curated file %s: %s', CURATED_PATH, err)
        return 0

    rows = build_rows(curated)
    if not rows:
        logger.warning('No capex rows resolved - leaving existing capex.json untouched')
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'metric': METRIC,
        'chip': CHIP,
        'companies': rows,
    }, indent=2) + '\n')

    logger.info('Wrote %d companies to %s', len(rows), OUTPUT_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
