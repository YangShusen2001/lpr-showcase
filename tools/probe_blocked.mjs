/**
 * One-off diagnostic: under the COOP/COEP headers added in ADR-004 §8, some
 * subresources on index.html get blocked with
 *   ERR_BLOCKED_BY_RESPONSE.NotSameOriginAfterDefaultedToSameOriginByCoep
 * This prints every failing request URL + the response headers, so we can tell
 * "cross-origin" from "same-origin but missing CORP".
 *
 * Usage: node tools/probe_blocked.mjs [/index.html]
 */
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const PORT = Number(process.env.PROBE_PORT || 8141);
const pagePath = process.argv[2] || '/index.html';

const require = createRequire('C:/Users/26671/Desktop/个人网站/');
const { chromium } = require('playwright-core');

const server = spawn(process.execPath, [path.join(HERE, 'serve.mjs'), ROOT, String(PORT)], {
  stdio: ['ignore', 'ignore', 'ignore'],
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const out = { pagePath, failures: [], ok: [] };

async function main() {
  await sleep(700);
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  page.on('requestfailed', (r) => {
    out.failures.push({
      url: r.url(),
      type: r.resourceType(),
      err: (r.failure() && r.failure().errorText) || '?',
    });
  });
  page.on('response', async (r) => {
    const u = r.url();
    if (!u.startsWith('http')) return;
    const h = r.headers();
    const row = {
      url: u,
      status: r.status(),
      corp: h['cross-origin-resource-policy'] || '(none)',
      acao: h['access-control-allow-origin'] || '(none)',
    };
    if (r.status() >= 400 || !h['cross-origin-resource-policy']) out.ok.push(row);
  });
  page.on('console', (m) => {
    if (m.type() === 'error') out.failures.push({ console: m.text() });
  });

  await page.goto(`http://127.0.0.1:${PORT}${pagePath}`, { waitUntil: 'load' });
  // index.html embeds the demo; force the runtime to actually fetch its assets.
  await page.evaluate(() => {
    const d = document.getElementById('demo');
    if (d) d.scrollIntoView();
  });
  await sleep(6000);

  const origin = await page.evaluate(() => ({
    href: location.href,
    origin: location.origin,
    isolated: crossOriginIsolated,
    sab: typeof SharedArrayBuffer !== 'undefined',
    demoReady: !!(window.__DEMO && window.__DEMO.ready),
    demoSrc: [...document.querySelectorAll('iframe')].map((f) => f.src),
  }));
  out.origin = origin;

  fs.writeFileSync(
    path.join(ROOT, '_evidence', 'probe_blocked.json'),
    JSON.stringify(out, null, 2),
    'utf8',
  );
  console.log(JSON.stringify(out, null, 2));
  await browser.close();
}

main()
  .catch((e) => console.log(JSON.stringify({ fatal: String((e && e.stack) || e) }, null, 2)))
  .finally(() => server.kill());
