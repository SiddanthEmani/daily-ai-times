/**
 * Daily AI Times — first-party anonymous event collector.
 *
 * A tiny edge endpoint that receives batched, cookieless engagement events from
 * the static site (via navigator.sendBeacon) and writes them to Workers
 * Analytics Engine. A scheduled GitHub Action later reads aggregates back out
 * (see src/backend/processors/engagement_aggregator.py) to produce
 * engagement_weights.json.
 *
 * Privacy posture:
 *   - We store only the fields the client sends (event name, anonymous cid/sid,
 *     section/source/archetype, and numeric measures). We never log the client
 *     IP, and we do not set or read any cookie.
 *   - The client already honors Do-Not-Track / opt-out before sending, so by the
 *     time a request reaches here the reader has not opted out.
 *
 * Bindings (wrangler.toml):
 *   - AE   : Analytics Engine dataset ("dat_events")
 *   - ALLOWED_ORIGIN : the site origin allowed via CORS (optional; defaults to *)
 */

const MAX_BODY_BYTES = 64 * 1024; // reject oversized payloads
const MAX_EVENTS = 50;            // cap events per beacon
const KNOWN_EVENTS = new Set([
  'impression', 'article_open', 'card_expand', 'dwell', 'scroll_depth',
  'category_nav', 'save_toggle', 'reaction', 'share', 'more_like_this',
  'audio_play', 'audio_complete', 'page_view',
]);

function corsHeaders(env, request) {
  const allowed = env.ALLOWED_ORIGIN || '*';
  const origin = request.headers.get('Origin') || '';
  const allowOrigin = allowed === '*' ? '*' : (origin === allowed ? origin : allowed);
  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

function looksLikeBot(ua) {
  if (!ua) return true; // beacons always carry a UA; absence is suspicious
  return /bot|crawl|spider|slurp|headless|curl|wget|python-requests/i.test(ua);
}

function str(v, max = 120) {
  if (v == null) return '';
  return String(v).slice(0, max);
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(env, request) });
    }
    if (url.pathname === '/health') {
      return new Response('ok', { status: 200, headers: corsHeaders(env, request) });
    }
    if (request.method !== 'POST' || url.pathname !== '/e') {
      return new Response('Not found', { status: 404, headers: corsHeaders(env, request) });
    }

    // Drop obvious bots/scrapers so they don't pollute preference aggregates.
    if (looksLikeBot(request.headers.get('User-Agent'))) {
      return new Response(null, { status: 204, headers: corsHeaders(env, request) });
    }

    const lenHeader = Number(request.headers.get('Content-Length') || '0');
    if (lenHeader > MAX_BODY_BYTES) {
      return new Response('Payload too large', { status: 413, headers: corsHeaders(env, request) });
    }

    let payload;
    try {
      const text = await request.text();
      if (text.length > MAX_BODY_BYTES) throw new Error('too large');
      payload = JSON.parse(text);
    } catch {
      return new Response('Bad request', { status: 400, headers: corsHeaders(env, request) });
    }

    const events = Array.isArray(payload?.events) ? payload.events.slice(0, MAX_EVENTS) : [];
    const cid = str(payload?.cid, 64);
    const sid = str(payload?.sid, 64);

    let written = 0;
    for (const ev of events) {
      const name = str(ev?.name, 40);
      if (!KNOWN_EVENTS.has(name)) continue;
      const p = ev?.properties || {};
      try {
        // Blobs: categorical dimensions. Doubles: numeric measures.
        // Index by cid so per-reader k-anonymity can be computed without ever
        // storing the cid->person mapping anywhere.
        env.AE?.writeDataPoint({
          blobs: [
            name,
            str(p.section, 40),
            str(p.source, 60),
            str(p.archetype, 24),
            str(p.id, 64),
            str(p.value, 16),     // reaction direction: up/down/clear
            str(p.channel, 24),   // share channel
            cid,
            sid,
          ],
          doubles: [
            num(p.ms),       // dwell
            num(p.rank),     // slot position
            num(p.percent),  // scroll depth
            num(p.duration), // audio length
          ],
          indexes: [cid || 'anon'],
        });
        written++;
      } catch {
        // Analytics Engine binding may be absent in dev; ignore.
      }
    }

    return new Response(JSON.stringify({ ok: true, written }), {
      status: 200,
      headers: { 'Content-Type': 'application/json', ...corsHeaders(env, request) },
    });
  },
};
