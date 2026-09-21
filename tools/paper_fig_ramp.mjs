/**
 * Scale sweep for the paper's "P(difference = +1) vs fractional coordinate" claim.
 *
 * Imports the **shipped** assets/js/pipeline.js and calls its exported
 * `resizeLinear`, so the data describes the code that actually ships rather than
 * a re-implementation.  Only the port's output is dumped here; the OpenCV side
 * is produced by tools/paper_fig_ramp.py, which consumes the concatenated blob
 * written below.
 *
 * Usage: node tools/paper_fig_ramp.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const EVID = path.join(ROOT, '_evidence', 'ramp');

const SRC_W = 256;
const SRC_H = 144;

// Deterministic pseudo-random RGBA source, reproducible in Python (see the
// identical LCG in tools/paper_fig_ramp.py).
function makeSource() {
  const data = new Uint8ClampedArray(SRC_W * SRC_H * 4);
  let s = 0x12345678;
  for (let i = 0; i < SRC_W * SRC_H; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    data[i * 4 + 0] = s & 0xff;
    data[i * 4 + 1] = (s >>> 8) & 0xff;
    data[i * 4 + 2] = (s >>> 16) & 0xff;
    data[i * 4 + 3] = 255;
  }
  return data;
}

const src = { data: makeSource(), width: SRC_W, height: SRC_H };
fs.mkdirSync(EVID, { recursive: true });

const P = await import(pathToFileURL(path.join(ROOT, 'assets', 'js', 'pipeline.js')).href);

const chunks = [];
const index = [];
let offset = 0;
for (const dstH of [72, 144, 216]) {
  for (let dstW = 40; dstW <= 480; dstW += 10) {
    if (dstW === SRC_W && dstH === SRC_H) continue;
    const out = P.resizeLinear(src, dstW, dstH);
    const buf = Buffer.from(out.data.buffer, out.data.byteOffset, out.data.byteLength);
    chunks.push(buf);
    index.push({ dstW, dstH, offset, len: buf.length });
    offset += buf.length;
  }
}

fs.writeFileSync(path.join(EVID, 'port_resized.bin'), Buffer.concat(chunks));
fs.writeFileSync(path.join(EVID, 'port_resized.json'),
  JSON.stringify({ srcW: SRC_W, srcH: SRC_H, configs: index }, null, 1), 'utf8');
console.log(`wrote ${index.length} configs, ${offset} bytes`);
