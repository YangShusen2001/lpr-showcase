/**
 * Minimal static file server for local verification and preview.
 *
 * Usage: node tools/serve.mjs <rootDir> [port]
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(process.argv[2] || path.join(fileURLToPath(new URL('.', import.meta.url)), '..'));
const port = Number(process.argv[3] || 8099);
// Bind host is overridable so the same server can be exposed to a phone on the
// LAN, e.g. node tools/serve.mjs . 8123 192.168.43.10 — the default stays on
// loopback so nothing is exposed unless it is asked for explicitly.
const host = process.argv[4] || '127.0.0.1';

// Request log. Written to stdout (visible when run in background) and, if
// SERVE_LOG is set, appended to that file so a phone session leaves durable
// evidence of exactly which assets were fetched.
const LOG = process.env.SERVE_LOG || '';
function logLine(s) {
  const line = `${new Date().toISOString()} ${s}\n`;
  process.stdout.write(line);
  if (LOG) {
    try { fs.appendFileSync(LOG, line); } catch { /* ignore */ }
  }
}

// Cross-origin isolation. Without it SharedArrayBuffer is unavailable and ORT stays
// pinned to one thread (demo.js falls back to 1 automatically). With it, this browser
// build gets the same multi-threaded WASM the HarmonyOS shell has — ADR-004 §5.3
// measured crossOriginIsolated false → true purely from these three headers.
// Every asset the pages load is same-origin, so require-corp costs us nothing.
const ISOLATION = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
};

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.md': 'text/markdown; charset=utf-8',
  '.pdf': 'application/pdf',
};

const server = http.createServer((req, res) => {
  const ua = req.headers['user-agent'] || '-';
  let rel;
  try {
    rel = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    logLine(`400 ${req.method} ${req.url} ua="${ua}"`);
    res.writeHead(400).end('bad url');
    return;
  }
  if (rel.endsWith('/')) rel += 'index.html';
  const full = path.join(root, rel);
  if (!full.startsWith(root)) {
    logLine(`403 ${req.method} ${rel} ua="${ua}"`);
    res.writeHead(403).end('forbidden');
    return;
  }
  fs.readFile(full, (err, buf) => {
    if (err) {
      logLine(`404 ${req.method} ${rel} ua="${ua}"`);
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('not found: ' + rel);
      return;
    }
    logLine(`200 ${req.method} ${rel} ${buf.length}B ua="${ua}"`);
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(full).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
      ...ISOLATION,
    });
    res.end(buf);
  });
});

server.listen(port, host, () => {
  console.log(`serving ${root} at http://${host}:${port}/`);
});
