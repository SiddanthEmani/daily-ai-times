// API base resolver. v1 by default; ?v2=1 or localStorage('dat:v2'='1') flips to v2.
// v2 writes to ./api/v2/ during dual-publishing. Flip the default to './api/v2' at cutover.

const V2_QUERY_KEY = 'v2';
const V2_STORAGE_KEY = 'dat:v2';

export function isV2Enabled() {
    try {
        const params = new URLSearchParams(window.location.search);
        if (params.has(V2_QUERY_KEY)) {
            const raw = params.get(V2_QUERY_KEY);
            const enable = raw === '' || raw === '1' || raw === 'true';
            try {
                window.localStorage.setItem(V2_STORAGE_KEY, enable ? '1' : '0');
            } catch (_) {
                // ignore storage errors (private mode, etc.)
            }
            return enable;
        }
        return window.localStorage.getItem(V2_STORAGE_KEY) === '1';
    } catch (_) {
        return false;
    }
}

export function apiBase() {
    return isV2Enabled() ? './api/v2' : './api';
}

export function apiUrl(path, { bust = true } = {}) {
    const base = apiBase();
    const clean = path.startsWith('/') ? path : `/${path}`;
    const url = `${base}${clean}`;
    return bust ? `${url}?t=${Date.now()}` : url;
}

export function banner() {
    if (!isV2Enabled()) return null;
    const el = document.createElement('div');
    el.textContent = 'v2 preview — Claude Agent SDK pipeline';
    el.style.cssText = [
        'position:fixed', 'top:0', 'right:0', 'z-index:9999',
        'padding:4px 12px', 'font:12px/1.4 system-ui,sans-serif',
        'background:#0f172a', 'color:#f8fafc', 'border-bottom-left-radius:6px'
    ].join(';');
    return el;
}
