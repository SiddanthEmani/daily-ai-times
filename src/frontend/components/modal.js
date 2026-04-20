// Article modal with "Open original" CTA that opens the source URL in a new tab.
// All dynamic values are escaped via escapeHTML before being placed in the template.
import { escapeHTML } from '../utils/utils.js';
import { paperImageSVG } from './chrome.js';

let modalRoot = null;
let escHandler = null;

function ensureRoot() {
    if (modalRoot) return modalRoot;
    modalRoot = document.createElement('div');
    modalRoot.id = 'modal-root';
    modalRoot.addEventListener('click', (e) => {
        const action = e.target?.closest?.('[data-action]')?.dataset?.action;
        if (action === 'modal-close') closeModal();
    });
    document.body.appendChild(modalRoot);
    return modalRoot;
}

function renderModal(story, saved) {
    const body = (story.body && story.body.length)
        ? story.body
        : [
            story.deck || '',
            'The full article continues at the source. Follow the link below to read the original reporting.',
        ];
    const paras = body.map(p => `<p>${escapeHTML(p)}</p>`).join('');
    const openOriginal = story.url ? `
        <a href="${escapeHTML(story.url)}" target="_blank" rel="noopener noreferrer"
           class="save-btn" style="text-decoration:none">
            Open original ↗
        </a>
    ` : '';

    return `
        <div class="modal-backdrop" data-action="modal-close">
            <article class="modal" data-action="modal-stop">
                <button class="modal-close" data-action="modal-close">× Close</button>
                <div class="kicker">${escapeHTML((story.kicker || story.section || '').toUpperCase())}</div>
                <h2>${escapeHTML(story.headline)}</h2>
                <div class="lead-byline" style="text-align:left;margin:6px 0 16px">
                    ${escapeHTML(story.byline)}
                    <span class="bullet"></span>Updated ${escapeHTML(story.time || 'today')}
                    <span class="bullet"></span>
                    <button class="save-btn${saved ? ' saved' : ''}" data-action="save" data-story-id="${escapeHTML(story.id)}">
                        ${saved ? '★ Saved to clippings' : '☆ Save for later'}
                    </button>
                    ${openOriginal ? `<span class="bullet"></span>${openOriginal}` : ''}
                </div>
                <div class="lead-media" style="margin-top:8px">
                    ${paperImageSVG((story.id || 'x').length + 9, 'ILLUSTRATION')}
                </div>
                <div class="body">
                    ${paras}
                    <p style="font-style:italic;color:var(--ink-soft)">— The Daily AI Times</p>
                </div>
            </article>
        </div>
    `;
}

export function openModal(story, { saved = false } = {}) {
    const root = ensureRoot();
    root.innerHTML = renderModal(story, saved);

    if (escHandler) window.removeEventListener('keydown', escHandler);
    escHandler = (e) => {
        if (e.key === 'Escape') {
            e.preventDefault();
            closeModal();
        }
    };
    window.addEventListener('keydown', escHandler);
}

export function closeModal() {
    if (modalRoot) modalRoot.innerHTML = '';
    if (escHandler) {
        window.removeEventListener('keydown', escHandler);
        escHandler = null;
    }
}

export function isModalOpen() {
    return !!(modalRoot && modalRoot.firstChild);
}
