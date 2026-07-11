// Ticker, masthead, nav, and the newsprint SVG placeholder.
import { escapeHTML } from '../utils/utils.js';

// Fallback shown before the first live fetch resolves, or if api/ticker.json
// is ever missing/malformed — keeps the ticker from ever rendering blank.
const FALLBACK_QUOTES = [
    { symbol: 'NVDA',  price: '1,184.22', change_percent: '+2.4%', up: true  },
    { symbol: 'MSFT',  price: '498.10',   change_percent: '+0.8%', up: true  },
    { symbol: 'GOOGL', price: '214.67',   change_percent: '-0.3%', up: false },
    { symbol: 'META',  price: '722.94',   change_percent: '+1.1%', up: true  },
    { symbol: 'AMD',   price: '208.55',   change_percent: '-1.6%', up: false },
    { symbol: 'TSM',   price: '241.08',   change_percent: '+0.9%', up: true  },
    { symbol: 'AMZN',  price: '231.40',   change_percent: '+0.5%', up: true  },
    { symbol: 'AAPL',  price: '241.83',   change_percent: '-0.2%', up: false },
];

export function tickerHTML(quotes) {
    const data = Array.isArray(quotes) && quotes.length ? quotes : FALLBACK_QUOTES;
    const strip = [...data, ...data]; // duplicate for seamless loop
    const spans = strip.map(q => `
        <span class="ticker-item ticker-market">
            <span class="ticker-tag">${escapeHTML(q.symbol)}</span>
            <span>${escapeHTML(q.price)}</span>
            <span class="ticker-ch ${q.up ? 'up' : 'down'}">${q.up ? '▲' : '▼'} ${escapeHTML(q.change_percent)}</span>
            <span class="ticker-dot"></span>
        </span>
    `).join('');
    return `
        <div class="ticker">
            <div class="ticker-track" data-action="ticker-track">
                <div class="ticker-strip">${spans}</div>
            </div>
        </div>
    `;
}

function formatDate(now) {
    return now.toLocaleDateString('en-US', {
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });
}
function formatTime(now) {
    return now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

// Dark-mode toggle button. Both icons live in the button; CSS shows the moon in
// light mode (click to go dark) and the sun in dark mode. Clicks are handled via
// the document-level delegation in app.js, so it survives masthead re-renders.
export function themeToggleHTML() {
    return `
        <button class="theme-toggle" data-action="theme-toggle" type="button"
                aria-label="Toggle dark mode" title="Toggle dark mode">
            <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path>
            </svg>
            <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="4"></circle>
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>
            </svg>
        </button>
    `;
}

export function mastheadHTML({ volume = 'Vol. XII · No. 1,508', query = '' } = {}) {
    return `
        <header class="masthead-wrap">
            <div class="masthead-topbar">
                <div class="left mast-meta-line"><span>${escapeHTML(volume)}</span></div>
                <div class="right">
                    <div class="nav-search">
                        <span>⌕</span>
                        <input id="search-input" placeholder="Search" value="${escapeHTML(query)}" autocomplete="off">
                    </div>
                    ${themeToggleHTML()}
                </div>
            </div>
            <h1 class="masthead-title">
                Daily <span class="amp">AI</span> Times
            </h1>
        </header>
    `;
}

// Left: "Sources" link that opens the sources overlay. Right: live date/time.
export function utilityRowHTML() {
    const now = new Date();
    return `
        <div class="utility-row">
            <button class="sources-link" data-action="sources" type="button">Sources</button>
            <div class="mast-meta-line">
                <span data-live="date">${escapeHTML(formatDate(now))}</span>
                <span data-live="time">${escapeHTML(formatTime(now))}</span>
            </div>
        </div>
    `;
}

const SOURCE_LIST = ['Reuters', 'Bloomberg', 'The Information', 'Axios', 'AP'];

// Deliberately not #modal-root — the e2e suite asserts that id is never
// attached to the DOM, so this overlay lives inside #root under its own id.
export function sourcesOverlayHTML() {
    const items = SOURCE_LIST.map(s => `<li>${escapeHTML(s)}</li>`).join('');
    return `
        <div class="sources-overlay" id="sources-overlay" data-action="close-sources">
            <div class="sources-card" data-action="none">
                <div class="sources-card-header">
                    <span>Sources</span>
                    <button class="sources-close" data-action="close-sources" type="button">✕</button>
                </div>
                <ul class="sources-list">${items}</ul>
            </div>
        </div>
    `;
}

// Keep masthead date/time fresh without re-rendering the whole page.
export function startMastheadClock(root = document) {
    const tick = () => {
        const now = new Date();
        const d = root.querySelector('[data-live="date"]');
        const t = root.querySelector('[data-live="time"]');
        if (d) d.textContent = formatDate(now);
        if (t) t.textContent = formatTime(now);
    };
    return setInterval(tick, 30000);
}

const NAV_LABELS = { All: 'today' };

export function navHTML({ section }, sections, counts) {
    const buttons = sections.map(s => {
        const count = counts[s];
        const active = s === section ? ' active' : '';
        const countHTML = count != null ? `<span class="count">${count}</span>` : '';
        return `
            <button class="nav-btn${active}" data-action="section" data-section="${escapeHTML(s)}" role="tab">
                ${escapeHTML(NAV_LABELS[s] || s)}${countHTML}
            </button>
        `;
    }).join('');
    return `
        <nav class="nav" role="tablist">
            ${buttons}
        </nav>
    `;
}

// Deterministic newsprint-style placeholder.
export function paperImageSVG(seed = 0, label = 'PHOTOGRAPH') {
    const rand = (i) => {
        const x = Math.sin(seed * 9301 + i * 49297) * 233280;
        return x - Math.floor(x);
    };
    const hue = Math.floor(rand(1) * 40) + 20;
    const tone1 = `oklch(0.68 0.05 ${hue})`;
    const tone2 = `oklch(0.42 0.04 ${hue + 20})`;
    const tone3 = `oklch(0.82 0.03 ${hue})`;
    const shapes = Math.floor(rand(2) * 3) + 2;
    const horizonY = (150 + rand(4) * 40).toFixed(1);
    const subjects = Array.from({ length: shapes }, (_, i) => {
        const cx = 80 + i * 90 + rand(10 + i) * 30;
        const cy = 160 + rand(20 + i) * 20;
        const r = 28 + rand(30 + i) * 18;
        return `
            <g>
                <ellipse cx="${cx}" cy="${(cy + r * 0.9).toFixed(1)}" rx="${(r * 0.7).toFixed(1)}" ry="${(r * 0.25).toFixed(1)}" fill="rgba(0,0,0,0.2)"/>
                <circle cx="${cx}" cy="${(cy - r * 0.4).toFixed(1)}" r="${(r * 0.38).toFixed(1)}" fill="${tone2}" opacity="0.85"/>
                <rect x="${(cx - r * 0.45).toFixed(1)}" y="${(cy - r * 0.1).toFixed(1)}" width="${(r * 0.9).toFixed(1)}" height="${(r * 0.85).toFixed(1)}" rx="4" fill="${tone2}" opacity="0.85"/>
            </g>
        `;
    }).join('');
    return `
        <svg viewBox="0 0 400 250" preserveAspectRatio="xMidYMid slice" style="width:100%;height:100%;display:block">
            <defs>
                <pattern id="hatch-${seed}" width="3" height="3" patternUnits="userSpaceOnUse" patternTransform="rotate(${(45 + rand(3) * 30).toFixed(1)})">
                    <line x1="0" y1="0" x2="0" y2="3" stroke="rgba(0,0,0,0.18)" stroke-width="1"/>
                </pattern>
                <pattern id="dots-${seed}" width="4" height="4" patternUnits="userSpaceOnUse">
                    <circle cx="2" cy="2" r="0.7" fill="rgba(0,0,0,0.25)"/>
                </pattern>
                <linearGradient id="sky-${seed}" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="${tone3}"/>
                    <stop offset="100%" stop-color="${tone1}"/>
                </linearGradient>
            </defs>
            <rect width="400" height="250" fill="url(#sky-${seed})"/>
            <rect width="400" height="250" fill="url(#dots-${seed})"/>
            <rect x="0" y="${horizonY}" width="400" height="250" fill="${tone2}" opacity="0.45"/>
            <rect x="0" y="${horizonY}" width="400" height="250" fill="url(#hatch-${seed})"/>
            ${subjects}
            <rect x="0" y="0" width="400" height="250" fill="none" stroke="rgba(0,0,0,0.1)" stroke-width="1"/>
            <text x="12" y="240" font-family="IBM Plex Mono, monospace" font-size="9" fill="rgba(0,0,0,0.5)" letter-spacing="1">${escapeHTML(label)}</text>
        </svg>
    `;
}

// Pause the CSS animation while hovering the ticker track.
export function wireTickerPause(root = document) {
    const track = root.querySelector('[data-action="ticker-track"]');
    if (!track) return;
    const strip = track.querySelector('.ticker-strip');
    if (!strip) return;
    track.addEventListener('mouseenter', () => strip.classList.add('paused'));
    track.addEventListener('mouseleave', () => strip.classList.remove('paused'));
}
