// Daily AI Times — orchestrator for the newspaper frontend.
// Fetches ./api/latest.json, normalizes articles, and renders the page as
// chrome + above-fold + below-fold columns. All dynamic text/attrs passed
// through escapeHTML before template interpolation; trusted-only assembly here.
import { escapeHTML } from '../utils/utils.js';
import { PerformanceMonitor, Analytics, LazyLoader } from '../utils/performance.js';
import { CustomAudioPlayer } from './custom-audio-player.js';
import {
    normalize, partition, tickerFromStories, sectionsFromStories,
} from './article-mapper.js';
import {
    tickerHTML, mastheadHTML, startMastheadClock, navHTML, wireTickerPause,
} from './chrome.js';
import { briefingHTML, leadHTML, swingHTML } from './above.js';
import {
    storyCardHTML, alsoNewsHTML,
    marketsBoxHTML, weatherBoxHTML, opinionBoxHTML, savedBoxHTML,
    getOpinionStory,
} from './below.js';
import { loadSourcesData, sourcesModalHTML } from './sources.js';

const APP_VERSION = '2026.4.0';
const SAVED_KEY = 'dat_saved';
const THEME_KEY = 'dat_theme';

const state = {
    section: 'All',
    query: '',
    savedIds: loadSavedIds(),
    focusIdx: -1,
    partitioned: null,
    sections: ['All'],
    mastheadClock: null,
    sourcesData: null,
};

const TAIL_POOLS = [
    [
        'Post-training teams are the new status symbol inside frontier labs',
        'A quiet move to smaller, specialist models at three major banks',
        'Retrieval evaluations finally get a standardized benchmark suite',
        'Open letter: senior researchers call for reproducibility covenants',
        'Why inference cost curves are flattening sooner than expected',
    ],
    [
        'Chip startups pitch vertical integration to skeptical buyers',
        'A union push at one major lab fizzles, for now',
        'The rise of the internal AI platform role, explained',
        'Evaluation teams are hiring; researchers are not. A data note.',
        'How three universities are rewriting their CS curricula',
    ],
    [
        'Edge inference returns as latency SLAs tighten across the stack',
        'Agents meet accounting: what the early deployments are teaching',
        'A survey of formal verification in ML pipelines',
        'Notes from a week inside a model-evaluation contractor',
        'Why the data flywheel language is being retired, quietly',
    ],
];
const TAIL_TITLES = ['In Other News', 'On The Wire', 'From The Desks'];

function loadSavedIds() {
    try { return new Set(JSON.parse(localStorage.getItem(SAVED_KEY) || '[]')); }
    catch { return new Set(); }
}
function persistSavedIds() {
    localStorage.setItem(SAVED_KEY, JSON.stringify([...state.savedIds]));
}

function checkVersionAndRefresh() {
    try {
        const stored = localStorage.getItem('dat_app_version');
        const lastRefresh = localStorage.getItem('dat_last_refresh');
        const now = Date.now();
        const oneHour = 60 * 60 * 1000;
        const stale = !stored || stored !== APP_VERSION
            || !lastRefresh || (now - parseInt(lastRefresh, 10)) > oneHour;
        if (!stale) return false;
        localStorage.setItem('dat_app_version', APP_VERSION);
        localStorage.setItem('dat_last_refresh', String(now));
        const veryRecent = lastRefresh && (now - parseInt(lastRefresh, 10)) < 5000;
        if (!veryRecent) {
            window.location.reload(true);
            return true;
        }
    } catch (err) {
        console.warn('Version check failed:', err);
    }
    return false;
}

function installCacheBustingFetch() {
    const originalFetch = window.fetch;
    window.fetch = function (url, options = {}) {
        if (typeof url === 'string' && (!url.startsWith('http') || url.startsWith(window.location.origin))) {
            options.headers = {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                ...options.headers,
            };
            options.cache = 'no-store';
        }
        return originalFetch.call(this, url, options);
    };
}

async function loadStories() {
    const url = `./api/latest.json?t=${Date.now()}&v=${APP_VERSION}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok && res.status !== 304) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const raw = Array.isArray(data.articles) ? data.articles : [];
        return raw.map((r, i) => normalize(r, i));
    } finally {
        clearTimeout(timer);
    }
}

function filteredGrid() {
    const q = state.query.trim().toLowerCase();
    const matchesSection = (s) => state.section === 'All' || s.section === state.section;
    const matchesQuery = (s) => {
        if (!q) return true;
        return [s.headline, s.deck, s.section, s.byline, s.source]
            .filter(Boolean).join(' ').toLowerCase().includes(q);
    };
    return state.partitioned.grid.filter(s => matchesSection(s) && matchesQuery(s));
}

function sectionCounts(all) {
    const counts = { All: all.length };
    for (const s of all) counts[s.section] = (counts[s.section] || 0) + 1;
    return counts;
}

// Stories rendered as grid cards. Shared by the renderer and keyboard nav so
// focusIdx and card idx stay in lockstep.
function visibleGridStories() {
    return filteredGrid();
}

function buildPageMarkup() {
    const showAbove = state.section === 'All' && !state.query.trim();
    const grid = filteredGrid();
    const gridStories = visibleGridStories();

    const cols = [[], [], [], []];
    gridStories.forEach((s, i) => cols[i % 4].push({ kind: 'story', story: s, idx: i }));

    cols[3].unshift({ kind: 'raw', html: '<div id="audio-placeholder"></div>' });
    if (showAbove) {
        cols[3].splice(2, 0, { kind: 'raw', html: marketsBoxHTML() });
        cols[3].splice(4, 0, { kind: 'raw', html: weatherBoxHTML() });
        cols[3].push({ kind: 'raw', html: opinionBoxHTML() });
        const savedHTML = savedBoxHTML(state.savedIds, state.partitioned.all);
        if (savedHTML) cols[0].push({ kind: 'raw', html: savedHTML });
    }

    const counts = sectionCounts(state.partitioned.all);
    const tickerItems = tickerFromStories(state.partitioned.all, 5);
    const navMarkup = navHTML({ section: state.section }, state.sections, counts);

    const colsHTML = cols.map(col => {
        const inner = col.map(item => item.kind === 'story'
            ? storyCardHTML(item.story, item.idx, {
                saved: state.savedIds.has(item.story.id),
                focused: state.focusIdx === item.idx,
            })
            : item.html
        ).join('');
        return `<div class="col">${inner}</div>`;
    }).join('');

    const resultBar = showAbove ? '' : `
        <div class="result-bar">
            Showing <strong>${grid.length}</strong> stories
            ${state.section !== 'All' ? ` in <strong class="accent">${escapeHTML(state.section)}</strong>` : ''}
            ${state.query.trim() ? ` matching "<strong class="accent">${escapeHTML(state.query)}</strong>"` : ''}
        </div>
    `;

    const aboveBlock = showAbove ? `
        <div class="above">
            ${briefingHTML(state.partitioned.briefing, navMarkup)}
            <div class="vrule"></div>
            <section class="lead-col">
                ${leadHTML(state.partitioned.lead)}
                ${alsoNewsHTML(TAIL_TITLES, TAIL_POOLS)}
            </section>
            <div class="vrule"></div>
            ${swingHTML(state.partitioned.swing)}
        </div>
    ` : '';

    return `
        ${tickerHTML(tickerItems)}
        <div class="page">
            ${mastheadHTML()}
            ${aboveBlock}
            ${showAbove ? '' : navMarkup}
            ${resultBar}
            <div class="below">${colsHTML}</div>
            <footer class="footer">
                <div>© 2026 Daily AI Times · An AI-assisted publication</div>
                <div><a href="#" data-action="open-sources" role="button">Our Sources</a> · Source code: <a href="https://github.com/SiddanthEmani/daily-ai-times" target="_blank" rel="noopener noreferrer">github.com/SiddanthEmani/daily-ai-times</a></div>
                <div>Keys: <strong>J</strong>/<strong>K</strong> to move · <strong>Enter</strong> to open in new tab</div>
            </footer>
        </div>
    `;
}

function render() {
    const root = document.getElementById('root');
    if (!root) return;
    root.innerHTML = buildPageMarkup();
    wireTickerPause(root);
    mountAudioBox();
    if (state.mastheadClock == null) state.mastheadClock = startMastheadClock(root);
}

// The audio-box DOM is created once and kept alive across renders so the
// underlying <audio> element (and any in-progress playback) survives
// root.innerHTML resets. Each render inserts a placeholder; mountAudioBox
// swaps the persistent element in.
let audioBoxEl = null;

function ensureAudioBox() {
    if (audioBoxEl) return audioBoxEl;
    audioBoxEl = document.createElement('div');
    audioBoxEl.id = 'audio-box';
    audioBoxEl.className = 'audio-box';
    try {
        const title = document.createElement('div');
        title.className = 'box-title';
        const label = document.createElement('span');
        label.textContent = "Today's Briefing";
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.textContent = 'LISTEN';
        title.appendChild(label);
        title.appendChild(chip);
        audioBoxEl.appendChild(title);
        const host = document.createElement('div');
        audioBoxEl.appendChild(host);
        new CustomAudioPlayer(`assets/audio/latest-podcast.wav?t=${Date.now()}`, host);
    } catch (err) {
        console.warn('Audio player mount failed:', err);
    }
    return audioBoxEl;
}

function mountAudioBox() {
    const placeholder = document.getElementById('audio-placeholder');
    if (!placeholder) return;
    placeholder.replaceWith(ensureAudioBox());
}

function installEventDelegation() {
    const root = document.getElementById('root');
    if (!root) return;

    // Click delegation sits on document.body so it also catches clicks inside
    // #modal-root (a sibling of #root). Without this the modal's Save button
    // would emit data-action="save" that never reaches the handler.
    document.body.addEventListener('click', (e) => {
        // Backdrop click (outside the dialog) closes the sources modal. Checked
        // before data-action lookup so clicks inside the dialog don't close it.
        if (e.target?.classList?.contains('modal-backdrop')) {
            closeSourcesModal();
            return;
        }

        const actionEl = e.target.closest?.('[data-action]');

        // Theme toggle: flip data-theme on <html> and persist. No render() needed —
        // CSS variables + the icon-swap rule react instantly, and delegation here
        // means the re-rendered masthead button keeps working.
        if (actionEl?.dataset.action === 'theme-toggle') {
            const next = document.documentElement.getAttribute('data-theme') === 'dark'
                ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            try { localStorage.setItem(THEME_KEY, next); } catch { /* storage may be blocked */ }
            return;
        }

        // Sources modal: open lazily (fetch once, cache), close on backdrop/×.
        // Rendered into #modal-root so it survives root.innerHTML re-renders.
        if (actionEl?.dataset.action === 'open-sources') {
            e.preventDefault();
            openSourcesModal();
            return;
        }
        if (actionEl?.dataset.action === 'close-sources') {
            e.preventDefault();
            closeSourcesModal();
            return;
        }

        // Save always wins — stop propagation so card expand/open doesn't also fire.
        if (actionEl?.dataset.action === 'save') {
            const storyId = actionEl.dataset.storyId;
            if (!storyId) return;
            e.stopPropagation();
            toggleSave(storyId);
            render();
            return;
        }

        // Story cards use a two-step interaction: first click expands the hidden
        // description; second click opens the URL. We only enter this path when the
        // click has no nearer [data-action] ancestor, so media-thumbs and other
        // direct-open elements still work.
        if (!actionEl) {
            const storyEl = e.target.closest?.('.story[data-story-id]');
            if (storyEl) {
                if (!storyEl.classList.contains('expanded')) {
                    storyEl.classList.add('expanded');
                } else {
                    const story = findStory(storyEl.dataset.storyId);
                    if (story) openStory(story);
                }
                return;
            }

            return;
        }

        const action = actionEl.dataset.action;
        const storyId = actionEl.dataset.storyId;

        if (action === 'section') {
            const name = actionEl.dataset.section;
            if (name && name !== state.section) {
                state.section = name;
                state.focusIdx = -1;
                render();
            }
            return;
        }
        // Lead headline: expand deck on first click, open URL on second click.
        if (action === 'expand-deck' && storyId) {
            if (!actionEl.classList.contains('expanded')) {
                actionEl.classList.add('expanded');
            } else {
                const story = findStory(storyId);
                if (story) openStory(story);
            }
            return;
        }
        if (action === 'open' && storyId) {
            const story = findStory(storyId);
            if (story) openStory(story);
            return;
        }
        if (action === 'open-opinion') {
            const story = getOpinionStory(actionEl.dataset.opinionId);
            if (story) openStory(story);
            return;
        }
        // open-tail: tail briefs have no source URL, nothing to open.
    });
}

function findStory(id) {
    return state.partitioned.all.find(s => s.id === id) || null;
}

function modalRoot() {
    return document.getElementById('modal-root');
}

function isSourcesModalOpen() {
    return !!modalRoot()?.querySelector('.sources-modal');
}

async function openSourcesModal() {
    const root = modalRoot();
    if (!root) return;
    // Loading placeholder while the data is fetched the first time.
    root.innerHTML = sourcesModalHTML(state.sourcesData || {}, { loading: !state.sourcesData });
    if (!state.sourcesData) {
        try {
            state.sourcesData = await loadSourcesData();
            // Only repaint if the modal is still open (user may have closed it).
            if (isSourcesModalOpen()) root.innerHTML = sourcesModalHTML(state.sourcesData);
        } catch (err) {
            console.warn('Failed to load sources:', err);
            if (isSourcesModalOpen()) {
                root.innerHTML = sourcesModalHTML({}, { error: true });
            }
        }
    }
    try { Analytics?.trackEvent?.('sources_open'); } catch { /* best-effort */ }
}

function closeSourcesModal() {
    const root = modalRoot();
    if (root) root.innerHTML = '';
}

function toggleSave(id) {
    if (state.savedIds.has(id)) state.savedIds.delete(id);
    else state.savedIds.add(id);
    persistSavedIds();
}

function openStory(story) {
    if (!story.url) return;
    window.open(story.url, '_blank', 'noopener,noreferrer');
    try { Analytics?.trackEvent?.('article_open', { id: story.id, section: story.section }); }
    catch { /* analytics is best-effort */ }
}

function installKeyboardNav() {
    window.addEventListener('keydown', (e) => {
        if (e.target?.tagName === 'INPUT') return;
        // Esc closes the sources modal first if it's open.
        if (e.key === 'Escape' && isSourcesModalOpen()) {
            e.preventDefault();
            closeSourcesModal();
            return;
        }
        // Nav walks only the grid cards that actually render; this matches the
        // focused-outline target and the list card idx values.
        const grid = visibleGridStories();
        if (e.key === 'j') {
            e.preventDefault();
            state.focusIdx = Math.min(grid.length - 1, state.focusIdx + 1);
            render();
            scrollFocused();
        } else if (e.key === 'k') {
            e.preventDefault();
            state.focusIdx = Math.max(0, state.focusIdx - 1);
            render();
            scrollFocused();
        } else if (e.key === 'Enter' && state.focusIdx >= 0) {
            const s = grid[state.focusIdx];
            if (s) openStory(s);
        } else if (e.key === 'Escape') {
            if (state.focusIdx >= 0) { state.focusIdx = -1; render(); }
        }
    });
}

function scrollFocused() {
    const grid = visibleGridStories();
    const s = grid[state.focusIdx];
    if (!s) return;
    const el = document.querySelector(`[data-story-id="${s.id}"]`);
    el?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
}

async function main() {
    const perf = new PerformanceMonitor();
    perf.mark('app_init_start');
    try { Analytics?.init?.(); } catch (err) { console.warn('Analytics init failed:', err); }
    if (checkVersionAndRefresh()) return;
    installCacheBustingFetch();

    try {
        const stories = await loadStories();
        if (!stories.length) throw new Error('No articles in latest.json');
        state.partitioned = partition(stories);
        state.sections = sectionsFromStories(stories);
        render();
        installEventDelegation();
        installKeyboardNav();
        try { LazyLoader?.init?.(); } catch { /* lazy loader is optional */ }
        perf.mark('app_ready');
        perf.report();
        try { Analytics?.trackPageView?.('home'); } catch { /* best-effort */ }
    } catch (err) {
        console.error('Daily AI Times failed to initialize:', err);
        const root = document.getElementById('root');
        if (root) {
            const msg = document.createElement('div');
            msg.style.cssText = "padding:40px;text-align:center;font-family:'Source Serif 4', serif";
            const h1 = document.createElement('h1');
            h1.style.cssText = "font-family:'Playfair Display', serif";
            h1.textContent = 'Daily AI Times';
            const p1 = document.createElement('p');
            p1.style.color = '#a00';
            p1.textContent = `Unable to load the paper: ${err.message || 'unknown error'}`;
            const p2 = document.createElement('p');
            p2.textContent = 'Please check back later.';
            msg.appendChild(h1); msg.appendChild(p1); msg.appendChild(p2);
            root.replaceChildren(msg);
        }
    }
}

document.addEventListener('DOMContentLoaded', main);
