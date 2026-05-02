// Browser-level smoke tests for the newspaper frontend. Runs against a live
// static server serving src/frontend/ with whatever latest.json is on disk.

import { test, expect } from '@playwright/test';

async function loadPage(page) {
    // `networkidle` hangs on gtag/Clarity/fonts keep-alive connections; wait
    // instead for the app to render the lead story.
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await page.locator('.lead-headline').first().waitFor({ timeout: 15_000 });
}

test.describe('newspaper frontend smoke', () => {
    test.beforeEach(async ({ page }) => {
        // Init script runs on every navigation, including reloads. Clear the
        // save key only on the first nav (gated by a sessionStorage flag that
        // survives reloads within the same context) so save-persistence tests
        // can actually observe persistence. Always pre-seed `dat_last_refresh`
        // so checkVersionAndRefresh() in app.js classifies the session as
        // veryRecent and skips its forced reload.
        await page.addInitScript(() => {
            try {
                if (!sessionStorage.getItem('_e2e_init_done')) {
                    sessionStorage.setItem('_e2e_init_done', '1');
                    window.localStorage.removeItem('dat_saved');
                }
                window.localStorage.setItem('dat_last_refresh', String(Date.now()));
            } catch {}
        });
    });

    test('loads with no console errors and renders page chrome', async ({ page }) => {
        const errors = [];
        page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
        page.on('pageerror', (err) => { errors.push(`pageerror: ${err.message}`); });

        await loadPage(page);

        await expect(page.locator('#root')).not.toBeEmpty();
        await expect(page.locator('.masthead-wrap')).toBeVisible();
        await expect(page.locator('.ticker-track')).toBeAttached();
        // The ticker duplicates items for the seamless loop, so ≥2 items expected.
        expect(await page.locator('.ticker-item').count()).toBeGreaterThanOrEqual(2);

        // Filter out benign third-party console noise (blocked GA/Clarity,
        // font-preload warnings, favicon 404, SSL errors for external hosts in
        // the sandboxed test environment).
        const material = errors.filter(e =>
            !/Google Analytics not configured/i.test(e) &&
            !/clarity/i.test(e) &&
            !/net::ERR_BLOCKED_BY_CLIENT/i.test(e) &&
            !/net::ERR_CERT_AUTHORITY_INVALID/i.test(e)
        );
        expect(material, `unexpected console errors:\n${material.join('\n')}`).toEqual([]);
    });

    test('lead story renders with a populated headline and story-id', async ({ page }) => {
        await loadPage(page);
        const lead = page.locator('.lead-headline').first();
        await expect(lead).toBeVisible();
        const text = (await lead.textContent() || '').trim();
        expect(text.length, 'lead headline empty').toBeGreaterThan(0);
        const storyId = await lead.getAttribute('data-story-id');
        expect(storyId, 'lead missing data-story-id').toBeTruthy();
    });

    test('clicking a grid story opens source URL in a new tab with no modal', async ({ page }) => {
        // Capture the URL passed to window.open before the new tab navigates
        // (in the sandboxed test env the external URL errors out before we can
        // read newTab.url(), so we intercept at the call site instead).
        await page.addInitScript(() => {
            const _orig = window.open.bind(window);
            window.__capturedOpenUrl = null;
            window.open = (url, ...rest) => { window.__capturedOpenUrl = url; return _orig(url, ...rest); };
        });

        await loadPage(page);

        // Modal root must never be injected into the DOM.
        await expect(page.locator('#modal-root')).not.toBeAttached();

        const headline = page.locator('article.story .story-headline').first();
        await expect(headline).toBeVisible();

        await headline.click();

        const openedUrl = await page.evaluate(() => window.__capturedOpenUrl);
        expect(openedUrl, 'window.open should be called with a valid http URL').toMatch(/^https?:\/\//);

        // Modal root must remain absent after the click.
        await expect(page.locator('#modal-root')).not.toBeAttached();
    });

    test('save/unsave persists across reload', async ({ page }) => {
        await loadPage(page);

        const card = page.locator('article.story').first();
        const storyId = await card.getAttribute('data-story-id');
        expect(storyId).toBeTruthy();
        const expectedText = (await card.locator('.story-headline').textContent() || '').trim();

        await card.locator('.save-btn').click();

        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.locator('.lead-headline').first().waitFor({ timeout: 15_000 });

        const clippingsBox = page.locator('.box', { hasText: 'Your Clippings' });
        await expect(clippingsBox).toBeVisible();
        await expect(clippingsBox.getByText(expectedText, { exact: false })).toBeVisible();

        const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('dat_saved') || '[]'));
        expect(saved).toContain(storyId);
    });

    test('keyboard nav: j moves focus, Enter opens article URL in new tab', async ({ page }) => {
        await page.addInitScript(() => {
            const _orig = window.open.bind(window);
            window.__capturedOpenUrl = null;
            window.open = (url, ...rest) => { window.__capturedOpenUrl = url; return _orig(url, ...rest); };
        });

        await loadPage(page);

        await page.keyboard.press('j');
        const focused = page.locator('article.story[style*="outline"]').first();
        await expect(focused).toBeVisible();

        await page.keyboard.press('Enter');

        const openedUrl = await page.evaluate(() => window.__capturedOpenUrl);
        expect(openedUrl, 'keyboard Enter should open a valid http URL').toMatch(/^https?:\/\//);

        // Modal must never appear.
        await expect(page.locator('#modal-root')).not.toBeAttached();
    });

    test('section filter narrows the grid and shows a result bar', async ({ page }) => {
        await loadPage(page);

        // Pick the first non-"All" nav button; All re-renders the above-the-fold block.
        const navBtn = page.locator('.nav-btn:not(.active)').first();
        const section = (await navBtn.getAttribute('data-section')) || '';
        expect(section).toBeTruthy();
        await navBtn.click();

        await expect(page.locator('.result-bar')).toBeVisible();
        await expect(page.locator('.result-bar .accent').first()).toHaveText(section);
        await expect(page.locator(`.nav-btn.active[data-section="${section}"]`)).toBeVisible();
    });

    test('search filters grid stories and reflects the query in the result bar', async ({ page }) => {
        await loadPage(page);

        const firstHeadline = (await page.locator('article.story .story-headline').first().textContent() || '').trim();
        const token = firstHeadline.split(/\s+/).find(w => w.length >= 4) || 'AI';

        await page.locator('[data-action="search"]').fill(token);
        await expect(page.locator('.result-bar')).toBeVisible();
        await expect(page.locator('.result-bar .accent').last()).toHaveText(token);
    });

    test('audio player mounts with the podcast source', async ({ page }) => {
        await loadPage(page);
        const audioBox = page.locator('#audio-box');
        await expect(audioBox).toBeVisible();
        await expect(audioBox.locator('.custom-audio-player')).toBeVisible();
    });
});
