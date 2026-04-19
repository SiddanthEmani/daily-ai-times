// Map raw latest.json articles into the story shape the render layer expects.

const RTF = typeof Intl !== 'undefined' && Intl.RelativeTimeFormat
    ? new Intl.RelativeTimeFormat('en', { numeric: 'auto', style: 'short' })
    : null;

function relativeTime(iso) {
    if (!iso) return 'today';
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return 'today';
    const diffMs = then - Date.now();
    const absMin = Math.abs(diffMs) / 60000;
    if (!RTF) return absMin < 60 ? `${Math.round(absMin)} min ago` : `${Math.round(absMin / 60)} hr ago`;
    if (absMin < 60)   return RTF.format(Math.round(diffMs / 60000), 'minute');
    if (absMin < 1440) return RTF.format(Math.round(diffMs / 3600000), 'hour');
    return RTF.format(Math.round(diffMs / 86400000), 'day');
}

function bodyFromRaw(raw) {
    const text = (raw.content || raw.description || '').trim();
    if (!text) return [];
    const paras = text.split(/\n+/).map(s => s.trim()).filter(Boolean);
    return paras.length ? paras : [text];
}

function pickScore(raw) {
    return raw.final_consensus?.weighted_score
        ?? raw.consensus_multi_dimensional_score?.overall_score
        ?? 0;
}

export function normalize(raw, idx = 0) {
    const section = raw.category || 'News';
    return {
        id: raw.article_id ?? `article-${idx}`,
        section,
        kicker: section.toUpperCase(),
        headline: raw.title || 'Untitled',
        deck: raw.description || '',
        byline: raw.author ? `By ${raw.author.toUpperCase()}` : 'By STAFF',
        body: bodyFromRaw(raw),
        time: relativeTime(raw.published_date),
        type: 'text',
        url: raw.url || '',
        source: raw.source || '',
        publishedAt: raw.published_date || null,
        score: pickScore(raw),
    };
}

// Sort in place by score desc, then publishedAt desc as tiebreaker.
function rankByScore(stories) {
    return [...stories].sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        const ta = a.publishedAt ? Date.parse(a.publishedAt) : 0;
        const tb = b.publishedAt ? Date.parse(b.publishedAt) : 0;
        return tb - ta;
    });
}

// Partition ranked stories into frontpage slots.
//   lead: highest-scoring
//   swing: 2nd highest
//   briefing: next 5 (shown as a numbered list under the briefing)
//   extras: next 3 (shown as "Also In The News" under the briefing)
//   grid: everything remaining
export function partition(stories) {
    const ranked = rankByScore(stories);
    return {
        lead:     ranked[0] || null,
        swing:    ranked[1] || null,
        briefing: ranked.slice(2, 7),
        extras:   ranked.slice(7, 10),
        grid:     ranked.slice(2),
        all:      ranked,
    };
}

// Ticker items: top 5 headlines with the category as the tag.
export function tickerFromStories(stories, n = 5) {
    return rankByScore(stories).slice(0, n).map(s => ({
        tag: (s.section || 'NEWS').toUpperCase(),
        text: s.headline,
    }));
}

// Unique section values sorted alphabetically — used to render nav tabs.
export function sectionsFromStories(stories) {
    const set = new Set(stories.map(s => s.section).filter(Boolean));
    return ['All', ...[...set].sort()];
}
