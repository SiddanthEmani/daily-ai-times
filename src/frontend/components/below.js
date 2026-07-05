// Below-the-fold: story cards, sidebar chart boxes, and the saved-clippings box.
import { escapeHTML } from '../utils/utils.js';
import { paperImageSVG } from './chrome.js';

// Illustrative static figures shipped with the redesign (not pipeline data).
const BENCHMARKS = [
    { label: 'GPT-5.2', value: 81.4 },
    { label: 'Claude Opus 5', value: 79.8 },
    { label: 'Gemini 3 Ultra', value: 78.2 },
    { label: 'Llama 5 405B', value: 71.6 },
    { label: 'DeepSeek V4', value: 69.9 },
];

const CAPEX = [
    { label: 'Amazon', value: 78 },
    { label: 'Microsoft', value: 64 },
    { label: 'Google', value: 52 },
    { label: 'Meta', value: 40 },
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

function barChartHTML(title, chip, rows) {
    const max = Math.max(...rows.map(r => r.value), 1);
    const barRows = rows.map((r, i) => `
        <div class="bar-row">
            <span class="bar-label">${escapeHTML(r.label)}</span>
            <div class="bar-track">
                <div class="bar-fill${i === 0 ? ' lead' : ''}" style="width:${(r.value / max * 100)}%"></div>
            </div>
            <span class="bar-value">${r.value}</span>
        </div>
    `).join('');
    return `
        <aside class="box chart-box">
            <div class="box-title"><span>${escapeHTML(title)}</span><span class="chip">${escapeHTML(chip)}</span></div>
            <div class="bar-chart">${barRows}</div>
        </aside>
    `;
}

export function benchmarksChartHTML() {
    return barChartHTML('Benchmark Leaderboard', 'REASONING INDEX', BENCHMARKS);
}

export function capexChartHTML() {
    return barChartHTML('Trailing 12-Mo. Data Center Capex', '$ BILLIONS', CAPEX);
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
