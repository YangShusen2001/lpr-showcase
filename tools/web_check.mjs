/**
 * Headless-browser verification driver.
 *
 * Boots the local static server, opens a page in Chromium, waits for a result
 * object the page publishes on window, then prints it as JSON.
 *
 * Usage: node tools/web_check.mjs <pagePath> [resultKey] [timeoutMs]
 *
 * Playwright is borrowed from the sibling 个人网站 project (already installed with
 * browsers) so this project keeps zero runtime dependencies of its own.
 */
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const PORT = Number(process.env.CHECK_PORT || 8123);

const pagePath = process.argv[2] || '/tools/_ort_smoke.html';
const resultKey = process.argv[3] || '__RESULT';
const timeoutMs = Number(process.argv[4] || 180000);

// Viewport is overridable so the same driver can probe a phone-sized layout
// (CHECK_W=390 CHECK_H=844 node tools/web_check.mjs /index.html).
const VW = Number(process.env.CHECK_W || 1280);
const VH = Number(process.env.CHECK_H || 900);
const VP_TAG = process.env.CHECK_W ? `_${VW}x${VH}` : '';

const require = createRequire('C:/Users/26671/Desktop/个人网站/');
const { chromium } = require('playwright-core');

const server = spawn(process.execPath, [path.join(HERE, 'serve.mjs'), ROOT, String(PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
});
server.stdout.on('data', (d) => process.stderr.write('[serve] ' + d));
server.stderr.on('data', (d) => process.stderr.write('[serve] ' + d));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  await sleep(700);
  const browser = await chromium.launch({
    args: ['--no-sandbox', '--allow-file-access-from-files'],
  });
  const page = await browser.newPage({ viewport: { width: VW, height: VH } });
  const consoleErrors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') consoleErrors.push(m.text());
  });
  page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

  const url = `http://127.0.0.1:${PORT}${pagePath}`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  let result = null;
  try {
    await page.waitForFunction(
      (k) => !!(window[k] && window[k].done === true),
      resultKey,
      { timeout: timeoutMs, polling: 250 },
    );
    result = await page.evaluate((k) => window[k], resultKey);
  } catch (e) {
    result = { error: 'timeout waiting for ' + resultKey, detail: String(e) };
  }

  const shot = path.join(ROOT, '_evidence', `web_${path.basename(pagePath).replace(/\W+/g, '_')}${VP_TAG}.png`);
  try {
    // Pages that use scroll-reveal only become fully opaque once every section has
    // entered the viewport, so walk the page before taking a full-page screenshot.
    await page.evaluate(async () => {
      const step = Math.round(window.innerHeight * 0.75);
      for (let y = 0; y < document.body.scrollHeight; y += step) {
        window.scrollTo(0, y);
        await new Promise((r) => setTimeout(r, 120));
      }
      window.scrollTo(0, 0);
      await new Promise((r) => setTimeout(r, 500));
    });
    // A position:fixed bar (sticky nav, phone action bar) is painted once at its
    // viewport position in a full-page capture, so it covers whatever page content
    // happens to sit under it and the screenshot silently lies about the layout.
    // CHECK_HIDE_FIXED=1 hides every fixed element for the shot only.
    if (process.env.CHECK_HIDE_FIXED === '1') {
      await page.evaluate(() => {
        for (const n of document.querySelectorAll('*')) {
          if (getComputedStyle(n).position === 'fixed') n.style.visibility = 'hidden';
        }
      });
    }
    await page.screenshot({ path: shot, fullPage: true });
  } catch { /* screenshot is best-effort */ }

  // Page-weight probe: how many viewport-heights does the page force the reader
  // to scroll through? A showcase page that needs 7 screens has already failed.
  let layout = null;
  try {
    layout = await page.evaluate(() => ({
      docHeight: document.documentElement.scrollHeight,
      viewport: window.innerHeight,
      screens: +(document.documentElement.scrollHeight / window.innerHeight).toFixed(2),
      overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      sections: [...document.querySelectorAll('section, header.nav, footer')].map((s) => ({
        tag: s.id || s.tagName.toLowerCase(),
        top: Math.round(s.getBoundingClientRect().top + window.scrollY),
        h: Math.round(s.getBoundingClientRect().height),
      })),
    }));
  } catch { /* probe is best-effort */ }

  const payload = { url, result, layout, consoleErrors, screenshot: shot };  // Written directly by Node so the evidence file keeps its UTF-8 (a PowerShell
  // pipe would re-decode stdout as the legacy console codepage and mangle CJK).
  const jsonPath = path.join(
    ROOT, '_evidence', `web_${path.basename(pagePath).replace(/\W+/g, '_')}${VP_TAG}.json`,
  );
  fs.writeFileSync(jsonPath, JSON.stringify(payload, null, 2), 'utf8');

  console.log(JSON.stringify({ ...payload, json: jsonPath }, null, 2));
  await browser.close();
}

main()
  .catch((e) => {
    console.log(JSON.stringify({ fatal: String(e && e.stack || e) }, null, 2));
    process.exitCode = 1;
  })
  .finally(() => {
    server.kill();
  });
