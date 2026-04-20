#!/usr/bin/env node
// Data contract: validate latest.json fields article-mapper.js reads, then
// round-trip through the real mapper (normalize / partition / ticker) to catch
// silent regressions in article ordering or field consumption.
//
// Invocation: `node scripts/test_frontend_contract.mjs [path-to-latest.json]`

import { readFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');
const jsonPath = resolve(process.argv[2] || join(repoRoot, 'src/frontend/api/latest.json'));
const mapperPath = join(repoRoot, 'src/frontend/components/article-mapper.js');

const failures = [];
const fail = (msg) => failures.push(msg);
const ok = (msg) => console.log(`  ok  ${msg}`);

// 1. Top-level shape.
console.log('== top-level ==');
const raw = readFileSync(jsonPath, 'utf8');
let data;
try { data = JSON.parse(raw); } catch (e) { fail(`invalid JSON: ${e.message}`); process.exit(1); }

if (typeof data.generated_at !== 'string') fail('generated_at missing or not a string');
else if (Number.isNaN(Date.parse(data.generated_at))) fail(`generated_at not parseable: ${data.generated_at}`);
if (!Array.isArray(data.articles)) fail('articles missing or not an array');
else if (data.articles.length === 0) fail('articles array is empty');
ok(`${data.articles?.length ?? 0} articles, generated_at=${data.generated_at}`);

// 2. Per-article required fields.
console.log('== per-article fields ==');
let scoredCount = 0;
let badDateCount = 0;
for (let i = 0; i < (data.articles || []).length; i++) {
    const a = data.articles[i];
    const tag = `article[${i}] id=${a.article_id ?? '?'}`;
    if (typeof a.article_id !== 'string' || !a.article_id) fail(`${tag}: article_id missing`);
    if (typeof a.title !== 'string' || !a.title) fail(`${tag}: title missing`);
    if (typeof a.category !== 'string' || !a.category) fail(`${tag}: category missing`);
    // description OR content — mapper falls back between them.
    if ((typeof a.description !== 'string' || !a.description) && (typeof a.content !== 'string' || !a.content)) {
        fail(`${tag}: neither description nor content present`);
    }
    if (typeof a.url !== 'string' || (a.url && !/^https?:\/\//.test(a.url))) {
        fail(`${tag}: url invalid (${a.url})`);
    }
    if (typeof a.published_date !== 'string' || Number.isNaN(Date.parse(a.published_date))) {
        badDateCount++;
        fail(`${tag}: published_date invalid (${a.published_date})`);
    }
    const score = a.final_consensus?.weighted_score ?? a.consensus_multi_dimensional_score?.overall_score;
    if (typeof score === 'number' && !Number.isNaN(score)) scoredCount++;
}
const scoredRatio = data.articles.length ? scoredCount / data.articles.length : 0;
if (scoredRatio < 0.8) {
    fail(`only ${scoredCount}/${data.articles.length} articles (${(scoredRatio*100).toFixed(0)}%) have a score — ranker will collapse to insertion order`);
}
ok(`${data.articles.length} articles field-validated, ${scoredCount} scored (${(scoredRatio*100).toFixed(0)}%)`);

// 3. Mapper round-trip. Import the real module and exercise its public API.
console.log('== mapper round-trip ==');
const mapperUrl = pathToFileURL(mapperPath).href;
const mapper = await import(mapperUrl);
const normalized = data.articles.map((r, i) => mapper.normalize(r, i));
if (normalized.length !== data.articles.length) fail(`normalize dropped articles: ${data.articles.length} → ${normalized.length}`);

const parts = mapper.partition(normalized);
if (!parts.lead || typeof parts.lead.id !== 'string') fail('partition.lead missing or malformed');
if (!parts.swing || typeof parts.swing.id !== 'string') fail('partition.swing missing (need ≥2 articles)');
if (!Array.isArray(parts.briefing)) fail('partition.briefing not an array');
else if (parts.briefing.length > 5) fail(`briefing.length > 5 (got ${parts.briefing.length})`);
if (!Array.isArray(parts.extras)) fail('partition.extras not an array');
else if (parts.extras.length > 3) fail(`extras.length > 3 (got ${parts.extras.length})`);
if (!Array.isArray(parts.grid)) fail('partition.grid not an array');
if (!Array.isArray(parts.all) || parts.all.length !== normalized.length) fail('partition.all length mismatch');

const ticker = mapper.tickerFromStories(normalized);
if (!Array.isArray(ticker)) fail('tickerFromStories did not return array');
else if (ticker.length > 5) fail(`ticker.length > 5 (got ${ticker.length})`);
else for (const item of ticker) {
    if (typeof item.tag !== 'string' || typeof item.text !== 'string') fail(`ticker item malformed: ${JSON.stringify(item)}`);
}

const sections = mapper.sectionsFromStories(normalized);
if (!Array.isArray(sections) || sections[0] !== 'All') fail(`sectionsFromStories should start with "All", got ${JSON.stringify(sections)}`);

ok(`lead=${parts.lead?.id}, swing=${parts.swing?.id}, briefing=${parts.briefing.length}, extras=${parts.extras.length}, grid=${parts.grid.length}, ticker=${ticker.length}, sections=${sections.length}`);

// 4. Lead story sanity: it must be the highest-scored article.
console.log('== lead ranking ==');
const scores = normalized.map(s => s.score ?? 0);
const maxScore = Math.max(...scores);
if ((parts.lead.score ?? 0) !== maxScore) fail(`lead story score ${parts.lead.score} ≠ max score ${maxScore}`);
ok(`lead.score=${parts.lead.score} matches max score across ${normalized.length} articles`);

// Report.
console.log();
if (failures.length) {
    console.error(`FAIL — ${failures.length} issue${failures.length === 1 ? '' : 's'}:`);
    for (const f of failures) console.error(`  • ${f}`);
    process.exit(1);
}
console.log(`PASS — data contract clean (${jsonPath})`);
