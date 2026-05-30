// Above-the-fold: briefing + extras (Also In The News), lead story, swing story.
import { escapeHTML } from '../utils/utils.js';
import { paperImageSVG } from './chrome.js';

// The category nav lives here (passed in as navMarkup) so it renders directly
// below the briefing list rather than as a full-width bar at the top of the page.
export function briefingHTML(briefing, navMarkup = '') {
    const items = (briefing || []).map((s, i) => `
        <li data-action="open" data-story-id="${escapeHTML(s.id)}">
            <span class="briefing-num">${String(i + 1).padStart(2, '0')}</span>
            <div class="briefing-text">
                <span class="kicker">${escapeHTML(s.kicker)}</span>
                ${escapeHTML(s.headline)}
            </div>
        </li>
    `).join('');

    return `
        <section>
            <h3 class="briefing-title">This Morning's Briefing</h3>
            <ol class="briefing-list">${items}</ol>
            ${navMarkup}
        </section>
    `;
}

export function leadHTML(story) {
    if (!story) return '';
    const paras = (story.body || []).slice(0, 3).map(p => `<p>${escapeHTML(p)}</p>`).join('');
    return `
        <section>
            <div class="lead-kicker">${escapeHTML(story.kicker)} · ${escapeHTML(story.section.toUpperCase())}</div>
            <h1 class="lead-headline" data-action="expand-deck" data-story-id="${escapeHTML(story.id)}" style="cursor:pointer">
                ${escapeHTML(story.headline)}
            </h1>
            <p class="lead-deck">${escapeHTML(story.deck)}</p>
            <div class="lead-media" data-action="open" data-story-id="${escapeHTML(story.id)}">
                ${paperImageSVG(1, 'PORTRAIT · STAFF PHOTOGRAPHER')}
            </div>
            <div class="lead-caption">
                ${escapeHTML(story.deck || '')}
                <div style="margin-top:4px;opacity:0.7">— ${escapeHTML(story.source || 'Staff')}</div>
            </div>
            <div class="lead-byline">
                ${escapeHTML(story.byline)}
                <span class="bullet"></span>
                Updated ${escapeHTML(story.time)}
            </div>
            <div class="lead-body">${paras}</div>
            <button class="lead-continued" data-action="open" data-story-id="${escapeHTML(story.id)}">
                Continue reading  →
            </button>
        </section>
    `;
}

export function swingHTML(story) {
    if (!story) return '';
    const paras = (story.body || []).slice(0, 4).map(p => `<p>${escapeHTML(p)}</p>`).join('');
    return `
        <section>
            <div class="swing-kicker">${escapeHTML(story.kicker)}</div>
            <h2 class="swing-headline" data-action="open" data-story-id="${escapeHTML(story.id)}" style="cursor:pointer">
                ${escapeHTML(story.headline)}
            </h2>
            <div class="swing-byline">${escapeHTML(story.byline)}</div>
            <div class="swing-body">
                ${paras}
                <button class="lead-continued" data-action="open" data-story-id="${escapeHTML(story.id)}">
                    Continue reading  →
                </button>
            </div>
        </section>
    `;
}
