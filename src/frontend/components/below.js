// Below-the-fold: story cards, tail lists, and sidebar fixture boxes.
import { escapeHTML } from '../utils/utils.js';
import { paperImageSVG } from './chrome.js';

const MARKETS = [
    { ticker: 'NVDA',  px: '1,184.22', ch: '+2.4%', up: true  },
    { ticker: 'MSFT',  px: '498.10',   ch: '+0.8%', up: true  },
    { ticker: 'GOOGL', px: '214.67',   ch: '-0.3%', up: false },
    { ticker: 'META',  px: '722.94',   ch: '+1.1%', up: true  },
    { ticker: 'AMD',   px: '208.55',   ch: '-1.6%', up: false },
    { ticker: 'TSM',   px: '241.08',   ch: '+0.9%', up: true  },
];

const WEATHER = [
    { city: 'SAN FRANCISCO', hi: 62, lo: 51, cond: 'Fog, then sun' },
    { city: 'NEW YORK',      hi: 71, lo: 58, cond: 'Partly cloudy' },
    { city: 'LONDON',        hi: 59, lo: 48, cond: 'Scattered rain' },
    { city: 'TOKYO',         hi: 68, lo: 55, cond: 'Clear' },
];

const OPINION = [
    { id: 'o1', title: 'The Case Against Another Benchmark', author: 'Harper Linde', excerpt: 'We are measuring the wrong things, with increasing precision.' },
    { id: 'o2', title: 'What Senior Engineers Owe the Junior Ones Now', author: 'Owen Matsuda', excerpt: 'A reading list is not a training plan. Neither is a chatbot.' },
    { id: 'o3', title: 'Why I Stopped Trusting Model Cards', author: 'Priya Ramachandran', excerpt: 'The disclosure document we adopted in 2019 is no longer serving anyone.' },
];

export function storyCardHTML(story, idx, { saved = false, focused = false } = {}) {
    const hasMedia = story.type === 'video' || story.type === 'photo';
    const media = hasMedia ? `
        <div class="media-thumb" data-action="open" data-story-id="${escapeHTML(story.id)}">
            ${paperImageSVG(idx + 10, story.type === 'video' ? 'VIDEO STILL' : 'PHOTOGRAPH')}
        </div>
    ` : '';
    const focusStyle = focused ? ' style="outline:2px solid var(--accent);outline-offset:4px"' : '';
    return `
        <article class="story" data-story-id="${escapeHTML(story.id)}"${focusStyle}>
            <div class="story-section">
                <span>${escapeHTML(story.section)}</span>
                <span class="sep"></span>
                <span style="color:var(--ink-soft)">${escapeHTML(story.source || '')}</span>
            </div>
            ${media}
            <h3 class="story-headline">${escapeHTML(story.headline)}</h3>
            <p class="story-summary">${escapeHTML(story.deck)}</p>
            <div class="story-meta">
                <span class="byline">${escapeHTML(story.byline)}</span>
                <span style="display:flex;gap:10px;align-items:center">
                    <span>${escapeHTML(story.time)}</span>
                    <button
                        class="save-btn${saved ? ' saved' : ''}"
                        data-action="save"
                        data-story-id="${escapeHTML(story.id)}"
                        title="${saved ? 'Remove from saved' : 'Save for later'}"
                    >${saved ? '★ Saved' : '☆ Save'}</button>
                </span>
            </div>
        </article>
    `;
}

export function tailSectionHTML(title, items, colIdx) {
    const list = items.map((h, j) => `
        <li data-action="open-tail" data-tail-key="${colIdx}-${j}" data-tail-headline="${escapeHTML(h)}">
            <span class="tail-bullet">—</span>
            <span>${escapeHTML(h)}</span>
        </li>
    `).join('');
    return `
        <section class="tail-section">
            <h4 class="tail-title">${escapeHTML(title)}</h4>
            <ul class="tail-list">${list}</ul>
        </section>
    `;
}

// Groups the brief lists (In Other News / On The Wire / From The Desks) under a
// single "Also In The News" heading, rendered below the lead story instead of
// scattered across the grid columns.
export function alsoNewsHTML(titles, pools) {
    const lists = titles.map((t, i) => tailSectionHTML(t, pools[i], i)).join('');
    return `
        <section class="also-news">
            <h3 class="briefing-title">Also In The News</h3>
            <div class="also-news-lists">${lists}</div>
        </section>
    `;
}

export function marketsBoxHTML() {
    const rows = MARKETS.map(r => `
        <div class="markets-row">
            <span class="t">${escapeHTML(r.ticker)}</span>
            <span>${escapeHTML(r.px)}</span>
            <span class="ch ${r.up ? 'up' : 'down'}">${escapeHTML(r.ch)}</span>
        </div>
    `).join('');
    return `
        <aside class="box">
            <div class="box-title"><span>Markets</span><span class="chip">● LIVE</span></div>
            <div class="markets-grid">${rows}</div>
        </aside>
    `;
}

export function weatherBoxHTML() {
    const rows = WEATHER.map(r => `
        <div class="weather-row">
            <span class="city">${escapeHTML(r.city)}</span>
            <span class="temps">${r.hi}° / ${r.lo}°</span>
            <span class="cond">${escapeHTML(r.cond)}</span>
        </div>
    `).join('');
    return `
        <aside class="box">
            <div class="box-title"><span>Cities</span><span class="chip">° FAHRENHEIT</span></div>
            ${rows}
        </aside>
    `;
}

export function opinionBoxHTML() {
    const rows = OPINION.map(o => `
        <div class="opinion-item">
            <h4 class="opinion-title" data-action="open-opinion" data-opinion-id="${escapeHTML(o.id)}">${escapeHTML(o.title)}</h4>
            <div class="opinion-author">By ${escapeHTML(o.author)}</div>
            <div class="opinion-excerpt">"${escapeHTML(o.excerpt)}"</div>
        </div>
    `).join('');
    return `
        <aside class="box">
            <div class="box-title"><span>Opinion</span><span class="chip">EDITORIAL</span></div>
            ${rows}
        </aside>
    `;
}

export function savedBoxHTML(savedIds, allStories) {
    if (!savedIds || savedIds.size === 0) return '';
    const saved = allStories.filter(s => savedIds.has(s.id));
    if (saved.length === 0) return '';
    const rows = saved.map(s => `
        <div class="opinion-item">
            <h4 class="opinion-title" data-action="open" data-story-id="${escapeHTML(s.id)}">${escapeHTML(s.headline)}</h4>
            <div class="opinion-author">
                ${escapeHTML(s.section)}
                <button class="save-btn saved" style="float:right" data-action="save" data-story-id="${escapeHTML(s.id)}">REMOVE</button>
            </div>
        </div>
    `).join('');
    return `
        <aside class="box" style="border-color:var(--accent)">
            <div class="box-title" style="border-bottom-color:var(--accent)">
                <span>Your Clippings</span>
                <span class="chip">${savedIds.size} SAVED</span>
            </div>
            ${rows}
        </aside>
    `;
}

export function getOpinionStory(id) {
    const o = OPINION.find(x => x.id === id);
    if (!o) return null;
    return {
        id: o.id,
        section: 'Opinion',
        kicker: 'OPINION',
        headline: o.title,
        deck: o.excerpt,
        byline: `By ${o.author.toUpperCase()}`,
        body: [o.excerpt, 'This column continues inside. The full text is available in the print edition.'],
        time: 'today',
        type: 'text',
        url: '',
        source: 'Daily AI Times',
        score: 0,
    };
}
