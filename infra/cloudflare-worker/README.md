# dat-collector — anonymous engagement collector

A tiny Cloudflare Worker that receives cookieless, batched engagement events from
the static site and writes them to **Workers Analytics Engine**. It is the
ingestion half of the preference feedback loop; the aggregation half lives in
`src/backend/processors/engagement_aggregator.py`.

## What it does

- `POST /e` — accepts a JSON beacon `{ v, cid, sid, events: [...] }`, validates
  and caps it, drops obvious bots, and writes one Analytics Engine data point per
  known event. Never stores the client IP; never sets a cookie.
- `GET /health` — liveness check.

## Privacy

- Only the fields the client sends are stored (event name, anonymous
  `cid`/`sid`, `section`/`source`/`archetype`, and numeric measures).
- The client already honors Do-Not-Track and a local opt-out before sending.
- Data points are indexed by `cid` so the aggregator can enforce k-anonymity
  (drop any bucket seen by fewer than N distinct readers) — the `cid` is a random
  UUID, never linked to a person.

## Deploy

```bash
cd infra/cloudflare-worker
npm install -g wrangler          # or: npx wrangler
wrangler login
wrangler deploy
```

After deploy, set the site secret so the frontend ships events here:

- GitHub repo → Settings → Secrets and variables → Actions
  - `COLLECTOR_URL = https://dat-collector.<your-subdomain>.workers.dev/e`

And, for the aggregator GitHub Action to read aggregates back:

- `CF_ACCOUNT_ID = <your Cloudflare account id>`
- `CF_API_TOKEN = <token with Account Analytics:Read>`

## Local dev

```bash
wrangler dev
curl -X POST localhost:8787/e \
  -H 'Content-Type: application/json' \
  -d '{"v":1,"cid":"test","sid":"s1","events":[{"name":"impression","properties":{"section":"Research","source":"arxiv"}}]}'
```

Note: the `AE` binding is a no-op in local `wrangler dev` unless configured; the
endpoint still returns `{ "ok": true }`.
