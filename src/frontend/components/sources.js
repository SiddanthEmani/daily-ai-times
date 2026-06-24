// "Our Sources" modal — fetches ./api/sources.json and renders the full list
// of feeds the paper is collected from, grouped by category. All dynamic text
// and attrs go through escapeHTML; trusted-only template assembly here.
import { escapeHTML } from '../utils/utils.js';

// Fetch the published source list. Mirrors loadStories() in app.js: short
// timeout via AbortController and cache-busting query string.
export async function loadSourcesData() {
    const url = `./api/sources.json?t=${Date.now()}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok && res.status !== 304) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

function categoryHTML(category) {
    const items = (category.sources || []).map(s => `
        <li class="sources-item">
            <a href="${escapeHTML(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(s.name)}</a>
        </li>
    `).join('');
    return `
        <section class="sources-group">
            <h3 class="sources-group-title">
                ${escapeHTML(category.name)}
                <span class="count">${(category.sources || []).length}</span>
            </h3>
            <ul class="sources-list">${items}</ul>
        </section>
    `;
}

// Build the full modal markup. The backdrop and close button carry data-action
// values so the existing document-level click delegation in app.js handles them.
// Pass { loading: true } to render the shell while sources.json is fetched.
export function sourcesModalHTML(data, { loading = false, error = false } = {}) {
    const categories = Array.isArray(data?.categories) ? data.categories : [];
    const total = data?.total ?? categories.reduce((n, c) => n + (c.sources?.length || 0), 0);
    const groups = categories.map(categoryHTML).join('');
    let body;
    if (error && !categories.length) {
        body = `<p class="sources-intro">Couldn't load the source list. Please try again later.</p>`;
    } else if (loading && !categories.length) {
        body = `<p class="sources-intro">Loading sources…</p>`;
    } else {
        body = `
            <p class="sources-intro">
                The Daily AI Times is compiled from <strong>${escapeHTML(String(total))}</strong>
                public RSS feeds across ${escapeHTML(String(categories.length))} categories.
                Headlines link to their original publishers.
            </p>
            <div class="sources-grid">${groups}</div>
        `;
    }
    return `
        <div class="modal-backdrop">
            <div class="modal sources-modal" role="dialog" aria-modal="true" aria-label="Our Sources">
                <div class="modal-head">
                    <div class="box-title" style="border:0;margin:0">
                        <span>Our Sources</span>
                        <span class="chip">TRANSPARENCY</span>
                    </div>
                    <button class="modal-close" data-action="close-sources" type="button" aria-label="Close">&times;</button>
                </div>
                ${body}
            </div>
        </div>
    `;
}
