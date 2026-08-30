// Below-the-fold: story cards and sidebar chart boxes.
import { escapeHTML } from '../utils/utils.js';
import { paperImageSVG } from './chrome.js';

// Fallback figures used only when the live leaderboard feed (api/leaderboard.json,
// produced by leaderboard_collector.py) is unavailable.
const BENCHMARKS = [
    { label: 'GPT-5.2', value: 81.4 },
    { label: 'Claude Opus 5', value: 79.8 },
    { label: 'Gemini 3 Ultra', value: 78.2 },
    { label: 'Llama 5 405B', value: 71.6 },
    { label: 'DeepSeek V4', value: 69.9 },
];

// Fallback figures used only when the live capex feed (api/capex.json, produced
// by capex_collector.py) is unavailable. Annualized data center / AI-infra spend
// in $ billions; every entry is an estimate here, so all carry the est. marker.
const CAPEX = [
    { label: 'Amazon', value: 110, estimated: true },
    { label: 'Microsoft', value: 88, estimated: true },
    { label: 'Alphabet', value: 85, estimated: true },
    { label: 'Meta', value: 66, estimated: true },
    { label: 'OpenAI', value: 50, estimated: true },
    { label: 'Oracle', value: 30, estimated: true },
    { label: 'Anthropic', value: 15, estimated: true },
];

export function storyCardHTML(story, idx, { focused = false } = {}) {
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
                <span>${escapeHTML(story.time)}</span>
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

export function benchmarksChartHTML(rows, chip) {
    const data = Array.isArray(rows) && rows.length ? rows : BENCHMARKS;
    return barChartHTML('Benchmark Leaderboard', chip || 'REASONING INDEX', data);
}

// rows come from the live capex feed (app.js loadCapex → api/capex.json); any
// failure falls back to the static CAPEX figures above. Estimated entries
// (private-company guidance, or seed values) get a trailing "*" and the chip
// notes what it means.
export function capexChartHTML(rows, chip) {
    const data = Array.isArray(rows) && rows.length ? rows : CAPEX;
    const marked = data.map(r => (r.estimated ? { ...r, label: `${r.label} *` } : r));
    const hasEstimate = data.some(r => r.estimated);
    let chipText = chip || '$B / YR';
    if (hasEstimate && !/EST/i.test(chipText)) chipText += ' · * EST.';
    return barChartHTML('AI Data Center Buildout', chipText, marked);
}
