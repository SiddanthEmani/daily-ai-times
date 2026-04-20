// Playwright config: chromium only, Python static server as the webServer.
// Port 8765 avoids clashing with a locally-running `npm run dev` on 8000.

import { defineConfig, devices } from '@playwright/test';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '../..');
const PORT = Number(process.env.E2E_PORT || 8765);
const FRONTEND_DIR = process.env.E2E_FRONTEND_DIR || resolve(REPO_ROOT, 'src/frontend');

export default defineConfig({
    testDir: '.',
    testMatch: /.*\.spec\.mjs$/,
    timeout: 20_000,
    fullyParallel: false,
    retries: process.env.CI ? 1 : 0,
    reporter: process.env.CI ? [['list'], ['github']] : 'list',
    use: {
        baseURL: `http://127.0.0.1:${PORT}`,
        trace: 'retain-on-failure',
        viewport: { width: 1280, height: 900 },
    },
    projects: [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    ],
    webServer: {
        command: `python3 -m http.server ${PORT} --directory "${FRONTEND_DIR}"`,
        cwd: REPO_ROOT,
        port: PORT,
        reuseExistingServer: !process.env.CI,
        timeout: 30_000,
        stdout: 'ignore',
        stderr: 'pipe',
    },
});
