/**
 * PC-side driver for the phone's WebView — the transport half of
 * "computer orchestrates, phone computes".
 *
 * Why this works at all: ArkWeb is Chromium, and `WebviewController.setWebDebuggingAccess(true)`
 * makes it open a DevTools domain socket on the device. `hdc fport` exposes that socket
 * as a TCP port on the PC, at which point the PC speaks plain CDP to the phone's page —
 * the same protocol Chrome DevTools uses. No custom protocol, no server on the phone.
 *
 * Setup (see tools/harmony/README.md):
 *   hdc shell "cat /proc/net/unix | grep devtools"     -> @webview_devtools_remote_<pid>
 *   hdc fport tcp:9222 localabstract:webview_devtools_remote_<pid>
 *   node tools/harmony/pc_drive_phone.mjs
 *
 * Usage:
 *   node tools/harmony/pc_drive_phone.mjs                       # list targets, then
 *   node tools/harmony/pc_drive_phone.mjs <imagePath> [label]   # push a local image
 *   node tools/harmony/pc_drive_phone.mjs --sample scene-2.jpg  # ask the phone for a built-in
 *   node tools/harmony/pc_drive_phone.mjs --eval "expr"         # raw escape hatch
 *
 * Requires Node >= 21 for the global WebSocket (no dependencies).
 */
import fs from 'node:fs';
import path from 'node:path';

const PORT = Number(process.env.CDP_PORT || 9222);
const HOST = `127.0.0.1:${PORT}`;

async function targets() {
  const r = await fetch(`http://${HOST}/json/list`);
  return r.json();
}

/** Minimal CDP client: one socket, id-matched request/response. */
class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error.message));
      else p.resolve(msg.result);
    });
  }

  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true });
      ws.addEventListener('error', () => rej(new Error('websocket failed: ' + url)), { once: true });
    });
    return new Cdp(ws);
  }

  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`timeout: ${method}`));
      }, 120000);
    });
  }

  /** Evaluate in the page, await promises, return the value (not a handle). */
  async eval(expression) {
    const r = await this.send('Runtime.evaluate', {
      expression, awaitPromise: true, returnByValue: true,
    });
    if (r.exceptionDetails) {
      const d = r.exceptionDetails;
      throw new Error('page threw: ' + (d.exception?.description || d.text));
    }
    return r.result?.value;
  }

  close() { this.ws.close(); }
}

const fileToDataUrl = (p) => {
  const ext = path.extname(p).toLowerCase();
  const mime = ext === '.png' ? 'image/png' : ext === '.webp' ? 'image/webp' : 'image/jpeg';
  return `data:${mime};base64,` + fs.readFileSync(p).toString('base64');
};

/** Drive one run on the phone and return the evidence object the page publishes. */
const RUN_AND_REPORT = (srcExpr, label) => `(async () => {
  if (!window.__LPR) return { error: 'window.__LPR missing — page not ready' };
  if (window.__DEMO.running) return { error: 'phone busy' };
  const t0 = performance.now();
  let src = ${srcExpr};
  if (typeof src === 'string') src = await (await fetch(src)).blob();
  await window.__LPR.run(src, ${JSON.stringify(label)});
  return {
    label: ${JSON.stringify(label)},
    wallMs: +(performance.now() - t0).toFixed(1),
    threads: window.__DEMO.threads,
    isolated: window.__DEMO.isolated,
    last: window.__DEMO.results[window.__DEMO.results.length - 1] || null,
    plate: document.getElementById('plateText').textContent,
    meta: document.getElementById('plateMeta').textContent,
    total: document.getElementById('msTotal').textContent,
    stages: [...document.querySelectorAll('.m-st')].map(s => s.textContent.trim()),
    errors: window.__DEMO.errors,
  };
})()`;

async function main() {
  const args = process.argv.slice(2);
  const list = await targets();
  if (!list.length) {
    console.error('no CDP target — is the app running and fport set up?');
    process.exit(1);
  }
  const page = list.find((t) => t.type === 'page') || list[0];
  console.log(`[pc] target: ${page.title}`);
  console.log(`[pc] url:    ${page.url}`);

  const cdp = await Cdp.connect(page.webSocketDebuggerUrl);

  const ready = await cdp.eval(
    `({ lpr: !!window.__LPR, demo: !!window.__DEMO, ready: !!(window.__DEMO && window.__DEMO.ready),
        threads: window.__DEMO && window.__DEMO.threads, isolated: window.__DEMO && window.__DEMO.isolated,
        hc: navigator.hardwareConcurrency, sab: typeof SharedArrayBuffer !== 'undefined' })`);
  console.log('[pc] page state:', JSON.stringify(ready));

  let out;
  if (args[0] === '--eval') {
    out = await cdp.eval(args.slice(1).join(' '));
  } else if (args[0] === '--sample') {
    const f = args[1] || 'scene-2.jpg';
    out = await cdp.eval(RUN_AND_REPORT(
      `window.__LPR.samplesBase + encodeURIComponent(${JSON.stringify(f)})`, `pc-sample:${f}`));
  } else if (args[0]) {
    const abs = path.resolve(args[0]);
    if (!fs.existsSync(abs)) throw new Error('no such image: ' + abs);
    const bytes = fs.statSync(abs).size;
    console.log(`[pc] pushing ${path.basename(abs)} (${bytes} B) as a data URL`);
    out = await cdp.eval(RUN_AND_REPORT(
      JSON.stringify(fileToDataUrl(abs)), args[1] || `pc-push:${path.basename(abs)}`));
  } else {
    out = await cdp.eval(`({ samples: window.__LPR.samples.map(s => s.file),
                              samplesBase: window.__LPR.samplesBase })`);
  }

  console.log('[pc] result:', JSON.stringify(out, null, 2));
  cdp.close();
  // The CDP socket keeps the event loop alive; without this the process lingers
  // for minutes after the work is done.
  process.exit(0);
}

main().catch((e) => {
  console.error('[pc] FAILED:', e.message);
  process.exitCode = 1;
});
