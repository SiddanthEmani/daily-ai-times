// Ticker, masthead, nav, and the newsprint SVG placeholder.
import { escapeHTML } from '../utils/utils.js';

export function tickerHTML(items) {
    if (!items || items.length === 0) return '';
    const strip = [...items, ...items]; // duplicate for seamless loop
    const spans = strip.map(it => `
        <span class="ticker-item">
            <span class="ticker-tag">${escapeHTML(it.tag)}</span>
            <span>${escapeHTML(it.text)}</span>
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

export function mastheadHTML({ volume = 'Vol. XII · No. 1,508', city = 'San Francisco' } = {}) {
    const now = new Date();
    return `
        <header class="masthead-wrap">
            <div class="masthead-topbar">
                <div class="left mast-meta-line"><span>${escapeHTML(volume)}</span></div>
                <div class="center"><span>${escapeHTML(city)}</span></div>
                <div class="right mast-meta-line" style="justify-content:flex-end">
                    <span data-live="date">${escapeHTML(formatDate(now))}</span>
                    <span data-live="time">${escapeHTML(formatTime(now))}</span>
                </div>
            </div>
            <h1 class="masthead-title">
                Daily <span class="amp">AI</span> Times
            </h1>
            <div class="masthead-motto" style="line-height:1.3;border-width:0">
                <span style="font-weight:800">A Newspaper for the Working Engineer</span>
                <span class="dot"></span>
                <span style="font-weight:600">Is AGI Already Among Us?</span>
                <span class="dot"></span>
                <span>Pollution &amp; Energy Index</span>
            </div>
        </header>
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

export function navHTML({ section, query }, sections, counts) {
    const buttons = sections.map(s => {
        const count = counts[s];
        const active = s === section ? ' active' : '';
        const countHTML = count != null ? `<span class="count">${count}</span>` : '';
        return `
            <button class="nav-btn${active}" data-action="section" data-section="${escapeHTML(s)}" role="tab">
                ${escapeHTML(s)}${countHTML}
            </button>
        `;
    }).join('');
    return `
        <nav class="nav" role="tablist">
            ${buttons}
            <div class="nav-search">
                <span>⌕</span>
                <input
                    data-action="search"
                    placeholder="Search the paper"
                    value="${escapeHTML(query || '')}"
                    aria-label="Search articles"
                />
            </div>
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
