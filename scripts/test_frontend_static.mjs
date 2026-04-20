#!/usr/bin/env node
// Static checks: module parsing, import resolution, HTML references, GA markers.
// Runs in <5s with zero dependencies.
//
// Invocation: `node scripts/test_frontend_static.mjs [root]`
//   root defaults to <repo>/src/frontend. Pass a deploy-sim `site/` dir to
//   re-validate after the prepare-site step.

import { readFileSync, existsSync, statSync, readdirSync } from 'node:fs';
import { dirname, resolve, relative, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');
const frontendRoot = resolve(process.argv[2] || join(repoRoot, 'src/frontend'));

const failures = [];
const fail = (msg) => failures.push(msg);
const ok = (msg) => console.log(`  ok  ${msg}`);

function walk(dir) {
    const out = [];
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, entry.name);
        if (entry.isDirectory()) out.push(...walk(p));
        else if (entry.isFile()) out.push(p);
    }
    return out;
}

// Match: `import ... from 'specifier'` and `import('specifier')`.
const STATIC_IMPORT_RE = /\bimport\s+(?:[^'"`;]+?\s+from\s+)?['"]([^'"]+)['"]/g;
const DYNAMIC_IMPORT_RE = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

function extractImports(src) {
    const specs = new Set();
    let m;
    while ((m = STATIC_IMPORT_RE.exec(src)) !== null) specs.add(m[1]);
    while ((m = DYNAMIC_IMPORT_RE.exec(src)) !== null) specs.add(m[1]);
    return [...specs];
}

function resolveSpec(fromFile, spec) {
    if (!spec.startsWith('.') && !spec.startsWith('/')) return null; // bare specifier
    const base = spec.startsWith('/') ? join(frontendRoot, spec.slice(1)) : resolve(dirname(fromFile), spec);
    const candidates = [base, base + '.js', base + '.mjs', join(base, 'index.js')];
    for (const c of candidates) {
        try { if (statSync(c).isFile()) return c; } catch {}
    }
    return false;
}

// 1. Every JS file under components/ and utils/ parses. A SyntaxError fails the test.
console.log('== module parse ==');
const jsFiles = walk(join(frontendRoot, 'components'))
    .concat(walk(join(frontendRoot, 'utils')))
    .filter(f => f.endsWith('.js'));
if (jsFiles.length === 0) fail('no JS files found under components/ or utils/');
for (const f of jsFiles) {
    try {
        // Dynamic import into Node ESM. Browser-only globals (document, window)
        // make modules throw at evaluation; catch that separately and only
        // fail on SyntaxError, which is what we actually care about here.
        await import(pathToFileURL(f).href);
    } catch (err) {
        if (err instanceof SyntaxError) {
            fail(`SyntaxError in ${relative(frontendRoot, f)}: ${err.message}`);
        }
        // Evaluation errors (ReferenceError for window/document) are expected
        // for browser modules; ignore them here.
    }
}
ok(`parsed ${jsFiles.length} modules`);

// 2. Every relative import in the graph resolves. Seed with app.js.
console.log('== import graph ==');
const entry = join(frontendRoot, 'components/app.js');
if (!existsSync(entry)) fail(`entry module missing: ${relative(frontendRoot, entry)}`);
const seen = new Set();
const queue = [entry];
while (queue.length) {
    const f = queue.shift();
    if (seen.has(f)) continue;
    seen.add(f);
    const src = readFileSync(f, 'utf8');
    for (const spec of extractImports(src)) {
        const resolved = resolveSpec(f, spec);
        if (resolved === null) continue; // bare specifier — skip
        if (resolved === false) {
            fail(`unresolved import "${spec}" in ${relative(frontendRoot, f)}`);
            continue;
        }
        queue.push(resolved);
    }
}
ok(`walked ${seen.size} modules from app.js, all imports resolved`);

// 3. index.html references exist.
console.log('== index.html refs ==');
const htmlPath = join(frontendRoot, 'index.html');
if (!existsSync(htmlPath)) fail('index.html missing');
const html = readFileSync(htmlPath, 'utf8');
const refRe = /(?:href|src)=["']([^"']+)["']/g;
let rm;
const refs = [];
while ((rm = refRe.exec(html)) !== null) refs.push(rm[1]);
for (const r of refs) {
    if (r.startsWith('http') || r.startsWith('//') || r.startsWith('data:') || r.startsWith('#')) continue;
    const target = r.startsWith('/') ? join(frontendRoot, r.slice(1)) : join(frontendRoot, r);
    if (!existsSync(target)) fail(`index.html references missing local file: ${r}`);
}
for (const required of ['styles/newspaper.css', 'components/app.js', 'favicon.ico']) {
    if (!existsSync(join(frontendRoot, required))) fail(`required asset missing: ${required}`);
}
ok(`${refs.length} href/src references all resolve`);

// 4. Deleted legacy files must not have come back.
console.log('== deleted files stay deleted ==');
const mustNotExist = [
    'components/articles.js',
    'components/magazine-layout.js',
    'components/navigation.js',
    'components/section-band.js',
    'styles/main.css',
    'utils/dom-helpers.js',
    'utils/state-management.js',
];
for (const p of mustNotExist) {
    if (existsSync(join(frontendRoot, p))) fail(`deleted file resurfaced: ${p}`);
}
ok(`${mustNotExist.length} legacy paths confirmed absent`);

// 5. Runtime assets that components construct paths to (not visible to the
// HTML ref scan). The audio player hard-codes assets/audio/latest-podcast.wav.
console.log('== runtime assets ==');
const runtimeAssets = ['assets/audio/latest-podcast.wav'];
for (const a of runtimeAssets) {
    if (!existsSync(join(frontendRoot, a))) fail(`runtime asset missing: ${a}`);
}
ok(`${runtimeAssets.length} runtime assets present`);

// 6. GA markers for sed replacement. Skipped when re-running against a
// post-deploy site/ tree (the workflow substitutes these out).
if (process.env.SKIP_GA_MARKERS === '1') {
    console.log('== Google Analytics sed markers == (skipped: SKIP_GA_MARKERS=1)');
} else {
    console.log('== Google Analytics sed markers ==');
    const gaMarkers = (html.match(/<!-- __GOOGLE_ANALYTICS__ -->/g) || []).length;
    if (gaMarkers !== 2) fail(`expected exactly 2 <!-- __GOOGLE_ANALYTICS__ --> markers, found ${gaMarkers}`);
    const gaPlaceholders = (html.match(/__GOOGLE_ANALYTICS__/g) || []).length;
    // 2 markers + ≥1 placeholder in gtag script = ≥3 total occurrences.
    if (gaPlaceholders < 3) fail(`expected __GOOGLE_ANALYTICS__ placeholder in gtag/script block, total found: ${gaPlaceholders}`);
    ok(`GA markers intact (${gaMarkers} block markers, ${gaPlaceholders} total placeholders)`);
}

// Report.
console.log();
if (failures.length) {
    console.error(`FAIL — ${failures.length} issue${failures.length === 1 ? '' : 's'}:`);
    for (const f of failures) console.error(`  • ${f}`);
    process.exit(1);
}
console.log(`PASS — static checks clean (root: ${relative(repoRoot, frontendRoot) || '.'})`);
