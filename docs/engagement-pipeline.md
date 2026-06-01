# Anonymous preference pipeline

How Daily AI Times learns what readers care about **without any login, account,
or cookie**, and feeds that back into curation.

## Goal

A reader should be able to catch up on AI in ~10 minutes and leave — no signup.
Yet the paper should still get smarter about what to surface. We do this with
**anonymous, cookieless, aggregates-only** signals: we never build per-person
profiles; we learn which *categories / sources / topics / article types* readers
engage with and bias future ranking toward them, within guardrails.

## Data flow

```
Reader (no login)
  → anonymous events (random cid in localStorage, no cookies, DNT honored)
  → Analytics.flushEvents() sendBeacon → Cloudflare Worker /e
  → Workers Analytics Engine (raw events)
        │  scheduled GitHub Action (engagement_aggregator.py)
        ▼
  engagement_weights.json  (per category/source/topic/archetype, k-anonymized)
        │  consumed by the orchestrator at ranking time
        ▼
  bounded re-ranking + ~15% novelty exploration → latest.json → static site
```

## Pieces

| Concern | Where |
| --- | --- |
| Capture (client) | `src/frontend/utils/performance.js` — `getClientId`, `isTrackingAllowed` (DNT + `dat_optout`), real `flushEvents` via `sendBeacon`; event helpers |
| Event wiring + UX | `src/frontend/components/app.js`, `below.js`, `article-mapper.js`, `custom-audio-player.js` |
| Ingestion (edge) | `infra/cloudflare-worker/` — Worker writing to Analytics Engine |
| Aggregation | `src/backend/processors/engagement_aggregator.py` → `src/shared/config/engagement_weights.json` |
| Consumption (ranking) | `src/backend/processors/engagement.py` + `orchestrator.py` |

## Events captured

`impression`, `article_open`, `card_expand`, `dwell`, `scroll_depth`,
`category_nav`, `save_toggle`, `reaction` (👍/👎), `audio_play`,
`audio_complete`. Each carries only an anonymous `cid`/`sid` and
section/source/archetype — never the article URL or any identity.

## Guardrails

- **Privacy:** cookieless; random `cid` only; Do-Not-Track and a one-click
  footer opt-out (`dat_optout`) are honored before anything is captured. The
  Worker stores no IP. Aggregation is **k-anonymized** — any bucket seen by
  fewer than `min_samples` (default 20) distinct readers drops to neutral 1.0.
- **No filter bubble:** the ranking multiplier is clamped to `[0.75, 1.35]`, so
  engagement nudges order but never dominates the editorial score; quality /
  fact-check / bias gates stay engagement-blind.
- **Serendipity:** ~15% (`exploration_epsilon`) of the article grid is reserved
  for the highest raw-novelty stories, regardless of engagement.
- **Anti-clickbait:** the engagement score weights dwell/scroll/save/reaction
  above raw clicks.
- **Cold-start safe & reversible:** neutral when the weights file is empty;
  disable instantly with `ENGAGEMENT_LOOP_ENABLED=0`.

## Operator setup

1. **Deploy the Worker** (see `infra/cloudflare-worker/README.md`):
   `cd infra/cloudflare-worker && wrangler deploy`.
2. **Add GitHub Actions secrets:**
   - `COLLECTOR_URL` — the Worker URL, e.g. `https://dat-collector.<sub>.workers.dev/e`
     (injected into `index.html` at deploy time; unset = capture stays inert).
   - `CF_ACCOUNT_ID`, `CF_API_TOKEN` — for the aggregator to read aggregates
     back (token needs Account Analytics: Read).
3. That's it. The collector ships events; the 4-hourly pipeline refreshes
   `engagement_weights.json` before ranking. Until secrets are set everything
   runs exactly as before (neutral weights).

## Testing offline

```bash
# Aggregate a local JSONL of client-shaped events (no Cloudflare needed)
python3 src/backend/processors/engagement_aggregator.py --input events.jsonl --dry-run
# Multiplier self-test
python3 src/backend/processors/engagement.py
```
