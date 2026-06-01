// Daily AI Times — orchestrator for the newspaper frontend.
// Fetches ./api/latest.json, normalizes articles, and renders the page as
// chrome + above-fold + below-fold columns. All dynamic text/attrs passed
// through escapeHTML before template interpolation; trusted-only assembly here.
import { escapeHTML } from '../utils/utils.js';
import { PerformanceMonitor, Analytics, LazyLoader } from '../utils/performance.js';
import { CustomAudioPlayer } from './custom-audio-player.js';
import {
    normalize, partition, tickerFromStories, sectionsFromStories, digestStories,
} from './article-mapper.js';
import {
    tickerHTML, mastheadHTML, startMastheadClock, navHTML, wireTickerPause,
} from './chrome.js';
import { briefingHTML, leadHTML, swingHTML } from './above.js';
import {
    storyCardHTML, alsoNewsHTML, digestHTML,
    marketsBoxHTML, weatherBoxHTML, opinionBoxHTML, savedBoxHTML,
    getOpinionStory,
} from './below.js';

const APP_VERSION = '2026.4.0';
const SAVED_KEY = 'dat_saved';
const REACT_KEY = 'dat_reactions';
// Pseudo-section used for the "Today in 10 minutes" digest view.
const DIGEST_SECTION = '10 Minutes';
const DIGEST_BUDGET_MIN = 10;

const state = {
    section: 'All',
    query: '',
    savedIds: loadSavedIds(),
    reactions: loadReactions(),
    focusIdx: -1,
    partitioned: null,
    sections: ['All'],
    mastheadClock: null,
};

// Records when a card was expanded so openStory can emit a "dwell" (consideration
// time) signal. In-memory only; resets each load.
const expandedAt = new Map();

// Minimal anonymous view descriptor attached to preference events. Carries no
// URL and no reader identity beyond the cid/sid added inside Analytics.
function viewOf(story, rank) {
    return {
        id: story.id,
        section: story.section,
        source: story.source || '',
        rank,
        archetype: story.section === 'Research' ? 'research' : 'article',
    };
}

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

// Reactions: { [storyId]: 'up' | 'down' }. Persisted like saved ids so the UI
// reflects prior taps; the aggregate signal is sent separately via Analytics.
function loadReactions() {
    try { return JSON.parse(localStorage.getItem(REACT_KEY) || '{}'); }
    catch { return {}; }
}
function persistReactions() {
    localStorage.setItem(REACT_KEY, JSON.stringify(state.reactions));
}
function toggleReaction(id, value) {
    const current = state.reactions[id];
    if (current === value) { delete state.reactions[id]; persistReactions(); return 'clear'; }
    state.reactions[id] = value;
    persistReactions();
    return value;
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

function isOptedOut() {
    try { return localStorage.getItem('dat_optout') === '1'; } catch { return false; }
}

// Shared footer. Includes a plain-language privacy line and a one-click opt-out,
// consistent with the cookieless / no-account posture (we also honor DNT).
function footerHTML(showKeys) {
    const optLabel = isOptedOut() ? 'Turn analytics back on' : 'Opt out of analytics';
    return `
        <footer class="footer">
            <div>© 2026 Daily AI Times · An AI-assisted publication</div>
            <div>Source code: <a href="https://github.com/SiddanthEmani/daily-ai-times" target="_blank" rel="noopener noreferrer">github.com/SiddanthEmani/daily-ai-times</a></div>
            <div class="privacy-note">No account, no cookies — only anonymous, aggregate analytics. <button class="linkish" data-action="optout">${optLabel}</button></div>
            ${showKeys ? '<div>Keys: <strong>J</strong>/<strong>K</strong> to move · <strong>Enter</strong> to open in new tab</div>' : ''}
        </footer>
    `;
}

function buildDigestMarkup() {
    const counts = sectionCounts(state.partitioned.all);
    const navMarkup = navHTML({ section: state.section }, state.sections, counts);
    const stories = digestStories(state.partitioned.all, DIGEST_BUDGET_MIN);
    const tickerItems = tickerFromStories(state.partitioned.all, 5);
    return `
        ${tickerHTML(tickerItems)}
        <div class="page">
            ${mastheadHTML()}
            ${navMarkup}
            ${digestHTML(stories, state.reactions, DIGEST_BUDGET_MIN)}
            ${footerHTML(false)}
        </div>
    `;
}

function buildPageMarkup() {
    if (state.section === DIGEST_SECTION) return buildDigestMarkup();
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
                reaction: state.reactions[item.story.id] || null,
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
            ${footerHTML(true)}
        </div>
    `;
}

function render() {
    const root = document.getElementById('root');
    if (!root) return;
    root.innerHTML = buildPageMarkup();
    wireTickerPause(root);
    mountAudioBox();
    installImpressionTracking();
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
        const actionEl = e.target.closest?.('[data-action]');

        // Save always wins — stop propagation so card expand/open doesn't also fire.
        if (actionEl?.dataset.action === 'save') {
            const storyId = actionEl.dataset.storyId;
            if (!storyId) return;
            e.stopPropagation();
            const willSave = !state.savedIds.has(storyId);
            toggleSave(storyId);
            const story = findStory(storyId);
            if (story) {
                try { Analytics?.trackEvent?.('save_toggle', { ...viewOf(story), saved: willSave }); }
                catch { /* best-effort */ }
            }
            render();
            return;
        }

        // Opt out / back in of anonymous analytics.
        if (actionEl?.dataset.action === 'optout') {
            e.stopPropagation();
            try {
                if (isOptedOut()) {
                    localStorage.removeItem('dat_optout');
                } else {
                    localStorage.setItem('dat_optout', '1');
                    Analytics.enabled = false;
                    Analytics.eventQueue = [];
                }
            } catch { /* ignore storage errors */ }
            // Reload so the (dis)enabled state takes effect cleanly everywhere.
            window.location.reload();
            return;
        }

        // Reactions are anonymous one-tap signals; they also win over expand/open.
        if (actionEl?.dataset.action === 'react') {
            const storyId = actionEl.dataset.storyId;
            const value = actionEl.dataset.react;
            if (!storyId || !value) return;
            e.stopPropagation();
            const result = toggleReaction(storyId, value);
            const story = findStory(storyId);
            if (story) {
                try { Analytics?.trackReaction?.(viewOf(story), result); }
                catch { /* best-effort */ }
            }
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
                const id = storyEl.dataset.storyId;
                if (!storyEl.classList.contains('expanded')) {
                    storyEl.classList.add('expanded');
                    expandedAt.set(id, Date.now());
                    const story = findStory(id);
                    if (story) {
                        try { Analytics?.trackCardExpand?.(viewOf(story)); }
                        catch { /* best-effort */ }
                    }
                } else {
                    const story = findStory(id);
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
                try { Analytics?.trackCategoryNav?.(name); } catch { /* best-effort */ }
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

function toggleSave(id) {
    if (state.savedIds.has(id)) state.savedIds.delete(id);
    else state.savedIds.add(id);
    persistSavedIds();
}

function openStory(story) {
    if (!story.url) return;
    window.open(story.url, '_blank', 'noopener,noreferrer');
    try {
        const view = viewOf(story);
        Analytics?.trackEvent?.('article_open', view);
        // If the reader expanded the card first, the time they spent considering
        // it before opening is a strong, clickbait-resistant interest signal.
        const since = expandedAt.get(story.id);
        if (since) {
            Analytics?.trackDwell?.(view, Date.now() - since);
            expandedAt.delete(story.id);
        }
    } catch { /* analytics is best-effort */ }
}

function installKeyboardNav() {
    window.addEventListener('keydown', (e) => {
        if (e.target?.tagName === 'INPUT') return;
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

// Thin top progress bar tied to scroll position; also emits scroll_depth
// milestones (25/50/75/100%) as engagement signals. Milestones fire once per
// load to avoid spamming the collector.
function installScrollTracking() {
    let bar = document.getElementById('read-progress');
    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'read-progress';
        document.body.appendChild(bar);
    }
    const fired = new Set();
    let ticking = false;
    const update = () => {
        ticking = false;
        const doc = document.documentElement;
        const max = doc.scrollHeight - doc.clientHeight;
        const pct = max > 0 ? Math.min(100, Math.round((doc.scrollTop / max) * 100)) : 0;
        bar.style.width = `${pct}%`;
        for (const m of [25, 50, 75, 100]) {
            if (pct >= m && !fired.has(m)) {
                fired.add(m);
                try { Analytics?.trackScrollDepth?.(m); } catch { /* best-effort */ }
            }
        }
    };
    window.addEventListener('scroll', () => {
        if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
}

// Per-article impressions: fire once when a card first becomes ~half visible.
// Re-run after each render so newly inserted cards are observed.
let impressionObserver = null;
const seenImpressions = new Set();
function installImpressionTracking() {
    if (!('IntersectionObserver' in window)) return;
    if (!impressionObserver) {
        impressionObserver = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                const id = entry.target.dataset.storyId;
                if (!id || seenImpressions.has(id)) { impressionObserver.unobserve(entry.target); continue; }
                seenImpressions.add(id);
                impressionObserver.unobserve(entry.target);
                const story = findStory(id);
                if (story) {
                    try { Analytics?.trackImpression?.(viewOf(story)); } catch { /* best-effort */ }
                }
            }
        }, { threshold: 0.5 });
    }
    document.querySelectorAll('.story[data-story-id]').forEach(el => {
        if (!seenImpressions.has(el.dataset.storyId)) impressionObserver.observe(el);
    });
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
        // Surface the "Today in 10 minutes" digest as the second tab, right
        // after All, so the fast-catch-up path is the first thing readers see.
        const baseSections = sectionsFromStories(stories);
        state.sections = ['All', DIGEST_SECTION, ...baseSections.filter(s => s !== 'All')];
        render();
        installEventDelegation();
        installKeyboardNav();
        installScrollTracking();
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
