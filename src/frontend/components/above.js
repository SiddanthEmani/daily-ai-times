// Above-the-fold: lead story + the audio briefing slot.
import { escapeHTML } from '../utils/utils.js';

export function leadHTML(story) {
    if (!story) return '';
    return `
        <section>
            <div class="lead-kicker">${escapeHTML(story.kicker)} · ${escapeHTML(story.section.toUpperCase())}</div>
            <h1 class="lead-headline" data-action="expand-deck" data-story-id="${escapeHTML(story.id)}">
                ${escapeHTML(story.headline)}
            </h1>
            <p class="lead-deck">${escapeHTML(story.deck)}</p>
            <div class="lead-byline">
                ${escapeHTML(story.byline)}
                <span class="bullet"></span>
                Updated ${escapeHTML(story.time)}
            </div>
        </section>
    `;
}

// Thin wrapper — app.js swaps in the persistent audio player DOM node here.
export function audioBriefingSlotHTML() {
    return `
        <section class="audio-slot">
            <div id="audio-placeholder"></div>
        </section>
    `;
}
