/**
 * Element-level screenshot helper for visual QA.
 *
 * Usage: node tools/shot_el.mjs <pagePath> <selector> <outPng> [timeoutMs]
 */
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const PORT = Number(process.env.SHOT_PORT || 8124);

const pagePath = process.argv[2];
const selector = process.argv[3];
const out = path.resolve(ROOT, process.argv[4]);
const timeoutMs = Number(process.argv[5] || 180000);

const require = createRequire('C:/Users/26671/Desktop/个人网站/');
const { chromium } = require('playwright-core');

const server = spawn(process.execPath, [path.join(HERE, 'serve.mjs'), ROOT, String(PORT)], {
  stdio: ['ignore', 'pipe', 'pipe'],
});
server.stdout.on('data', (d) => process.stderr.write('[serve] ' + d));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  await sleep(700);
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.goto(`http://127.0.0.1:${PORT}${pagePath}`, { waitUntil: 'domcontentloaded' });

  if (pagePath.includes('auto=')) {
    await page.waitForFunction(() => window.__PAGE && window.__PAGE.done === true, null,
      { timeout: timeoutMs, polling: 250 }).catch(() => {});
  }
  await page.waitForSelector(selector, { timeout: 15000 });
  await page.locator(selector).scrollIntoViewIfNeeded();
  await sleep(900);
  await page.locator(selector).screenshot({ path: out });
  console.log('saved ' + out);
  await browser.close();
}

main().catch((e) => { console.error(e); process.exitCode = 1; })
  .finally(() => server.kill());
