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
        // font-preload warnings, favicon 404 on some servers).
        const material = errors.filter(e =>
            !/Google Analytics not configured/i.test(e) &&
            !/clarity/i.test(e) &&
            !/net::ERR_BLOCKED_BY_CLIENT/i.test(e)
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

    test('clicking a grid story opens modal with Open original CTA; Esc closes', async ({ page }) => {
        await loadPage(page);

        const headline = page.locator('article.story .story-headline').first();
        await expect(headline).toBeVisible();
        const expectedText = (await headline.textContent() || '').trim();

        await headline.click();

        const modal = page.locator('#modal-root .modal');
        await expect(modal).toBeVisible();
        await expect(modal.locator('h2')).toHaveText(expectedText);

        const cta = modal.locator('a', { hasText: 'Open original' });
        await expect(cta).toBeVisible();
        const href = await cta.getAttribute('href');
        expect(href, 'Open original CTA href').toMatch(/^https?:\/\//);
        await expect(cta).toHaveAttribute('target', '_blank');
        await expect(cta).toHaveAttribute('rel', /noopener/);

        await page.keyboard.press('Escape');
        await expect(modal).toBeHidden();
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

    test('keyboard nav: j moves focus, Enter opens modal, Esc closes', async ({ page }) => {
        await loadPage(page);

        await page.keyboard.press('j');
        const focused = page.locator('article.story[style*="outline"]').first();
        await expect(focused).toBeVisible();

        await page.keyboard.press('Enter');
        await expect(page.locator('#modal-root .modal')).toBeVisible();

        await page.keyboard.press('Escape');
        await expect(page.locator('#modal-root .modal')).toBeHidden();
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
