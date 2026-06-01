#!/usr/bin/env python3
"""Generate static syndication artifacts from the curated articles.

Reads the already-generated ``latest.json`` and emits:
  - ``feed.xml``     RSS 2.0 feed (with a podcast enclosure for the daily audio)
  - ``sitemap.xml``  a minimal sitemap for the canonical page + category views
  - ``robots.txt``   allow-all with a sitemap pointer

These are pure derivations of ``latest.json`` (no network, no API keys), so the
step is cheap to run in CI right after the pipeline writes the API files, and is
safe to run locally to refresh the committed starter artifacts.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape

SITE_URL = "https://siddanthemani.github.io/daily-ai-times"
SITE_TITLE = "Daily AI Times"
SITE_DESC = (
    "AI-curated headlines, research, and analysis. The fastest way to catch up "
    "on artificial intelligence today."
)
AUDIO_PATH = "assets/audio/latest-podcast.wav"
MAX_ITEMS = 30


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def build_rss(articles: list[dict], audio_file: Path | None) -> str:
    now = format_datetime(datetime.now(timezone.utc))
    items: list[str] = []
    for a in articles[:MAX_ITEMS]:
        title = escape(a.get("title") or "Untitled")
        link = escape(a.get("url") or SITE_URL)
        desc = escape(a.get("description") or "")
        guid = escape(str(a.get("article_id") or a.get("url") or title))
        category = escape(a.get("category") or "AI")
        pub = format_datetime(_parse_dt(a.get("published_date")))
        items.append(
            f"""    <item>
      <title>{title}</title>
      <link>{link}</link>
      <guid isPermaLink="false">{guid}</guid>
      <category>{category}</category>
      <pubDate>{pub}</pubDate>
      <description>{desc}</description>
    </item>"""
        )

    # Podcast enclosure: advertise the daily audio briefing so the feed is
    # ingestible as a podcast. Only emitted when the audio file exists.
    enclosure = ""
    if audio_file and audio_file.exists():
        size = audio_file.stat().st_size
        enclosure = (
            f"""    <item>
      <title>Daily AI Times — Audio Briefing</title>
      <link>{SITE_URL}/{AUDIO_PATH}</link>
      <guid isPermaLink="false">audio-{datetime.now(timezone.utc):%Y%m%d%H}</guid>
      <pubDate>{now}</pubDate>
      <description>Today's AI briefing, narrated.</description>
      <enclosure url="{SITE_URL}/{AUDIO_PATH}" length="{size}" type="audio/wav"/>
    </item>"""
        )

    body = "\n".join([enclosure] + items) if enclosure else "\n".join(items)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{SITE_TITLE}</title>
    <link>{SITE_URL}/</link>
    <description>{escape(SITE_DESC)}</description>
    <language>en-us</language>
    <lastBuildDate>{now}</lastBuildDate>
    <generator>Daily AI Times pipeline</generator>
{body}
  </channel>
</rss>
"""


def build_sitemap(categories: list[str]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urls = [f"  <url><loc>{SITE_URL}/</loc><lastmod>{today}</lastmod><changefreq>hourly</changefreq><priority>1.0</priority></url>"]
    for cat in sorted(set(categories)):
        slug = escape(cat.lower().replace(" ", "-"))
        urls.append(
            f"  <url><loc>{SITE_URL}/api/categories/{slug}.json</loc><lastmod>{today}</lastmod><changefreq>hourly</changefreq><priority>0.6</priority></url>"
        )
    body = "\n".join(urls)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{body}
</urlset>
"""


def build_robots() -> str:
    return f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n"


def generate(frontend_dir: Path) -> None:
    latest = frontend_dir / "api" / "latest.json"
    if not latest.exists():
        print(f"[generate_feed] {latest} not found; skipping", file=sys.stderr)
        return
    data = json.loads(latest.read_text())
    articles = data.get("articles", []) if isinstance(data, dict) else []
    categories = [a.get("category", "AI") for a in articles if a.get("category")]

    audio_file = frontend_dir / AUDIO_PATH
    (frontend_dir / "feed.xml").write_text(build_rss(articles, audio_file))
    (frontend_dir / "sitemap.xml").write_text(build_sitemap(categories))
    (frontend_dir / "robots.txt").write_text(build_robots())
    print(
        f"[generate_feed] wrote feed.xml ({len(articles[:MAX_ITEMS])} items), "
        f"sitemap.xml, robots.txt to {frontend_dir}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "src" / "frontend"
    generate(target)
