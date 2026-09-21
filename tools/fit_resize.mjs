/**
 * Empirical fitter: find the resize variant that reproduces cv2.resize(INTER_LINEAR)
 * bit for bit.
 *
 * Reads the ground-truth tensor dumped by tools/dump_tensor.py and brute-forces a
 * small grid of fixed-point two-pass bilinear variants, reporting the exact-match
 * count for each. The winning variant is then transcribed into assets/js/pipeline.js.
 *
 * Usage: node tools/fit_resize.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const EVID = path.join(ROOT, '_evidence');

const meta = JSON.parse(fs.readFileSync(path.join(EVID, 'tensor_meta.json'), 'utf8'));
const SIZE = meta.size;
const COEF_BITS = 11;
const COEF_SCALE = 1 << COEF_BITS;

/** OpenCV's cvRound: round to nearest, ties to even. */
function cvRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const rounders = {
  halfEven: cvRound,
  halfUp: (x) => Math.floor(x + 0.5),
  trunc: (x) => Math.trunc(x),
};

/**
 * Two-pass fixed-point bilinear, parameterised so the exact OpenCV layout can be
 * recovered by search: `shiftH`/`roundH` control the horizontal pass, `shiftV`/
 * `roundV` the vertical one.
 */
function resizeFixed(src, dstW, dstH, sw, sh, cfg) {
  const { shiftH, roundH, shiftV, roundV, rnd } = cfg;
  const r = rounders[rnd];
  const data = src;

  const xo = new Int32Array(dstW);
  const cx = new Int32Array(dstW * 2);
  for (let x = 0; x < dstW; x++) {
    let fx = ((x + 0.5) * sw) / dstW - 0.5;
    let sx = Math.floor(fx);
    fx -= sx;
    if (sx < 0) { fx = 0; sx = 0; }
    if (sx >= sw - 1) { fx = 0; sx = sw - 1; }
    xo[x] = sx;
    cx[x * 2] = r((1 - fx) * COEF_SCALE);
    cx[x * 2 + 1] = r(fx * COEF_SCALE);
  }
  const yo = new Int32Array(dstH);
  const cy = new Int32Array(dstH * 2);
  for (let y = 0; y < dstH; y++) {
    let fy = ((y + 0.5) * sh) / dstH - 0.5;
    let sy = Math.floor(fy);
    fy -= sy;
    if (sy < 0) { fy = 0; sy = 0; }
    if (sy >= sh - 1) { fy = 0; sy = sh - 1; }
    yo[y] = sy;
    cy[y * 2] = r((1 - fy) * COEF_SCALE);
    cy[y * 2 + 1] = r(fy * COEF_SCALE);
  }

  // horizontal pass -> integer buffer
  const tmp = new Int32Array(dstW * sh * 4);
  for (let y = 0; y < sh; y++) {
    const row = y * sw * 4;
    const orow = y * dstW * 4;
    for (let x = 0; x < dstW; x++) {
      const s0 = row + xo[x] * 4;
      const s1 = s0 + 4;
      const c0 = cx[x * 2];
      const c1 = cx[x * 2 + 1];
      const o = orow + x * 4;
      for (let c = 0; c < 4; c++) {
        const v = data[s0 + c] * c0 + data[s1 + c] * c1;
        tmp[o + c] = shiftH === 0 ? v : (v + roundH) >> shiftH;
      }
    }
  }
  // vertical pass
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const stride = dstW * 4;
  for (let y = 0; y < dstH; y++) {
    const r0 = yo[y] * stride;
    const r1 = r0 + stride;
    const c0 = cy[y * 2];
    const c1 = cy[y * 2 + 1];
    const o = y * stride;
    for (let i = 0; i < stride; i++) {
      const v = tmp[r0 + i] * c0 + tmp[r1 + i] * c1;
      out[o + i] = shiftV === 0 ? v : (v + roundV) >> shiftV;
    }
  }
  return out;
}

function letterBoxVariant(src, w, h, size, cfg) {
  const r = Math.min(size / h, size / w);
  const newH = Math.trunc(h * r);
  const newW = Math.trunc(w * r);
  const top = Math.trunc((size - newH) / 2);
  const left = Math.trunc((size - newW) / 2);
  const resized = resizeFixed(src, newW, newH, w, h, cfg);
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < newH; y++) {
    out.set(resized.subarray(y * newW * 4, (y + 1) * newW * 4), ((y + top) * size + left) * 4);
  }
  return { data: out, r, left, top, newW, newH };
}

function toNCHW(data, w, h) {
  const plane = w * h;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = data[p] / 255;
    out[plane + i] = data[p + 1] / 255;
    out[2 * plane + i] = data[p + 2] / 255;
  }
  return out;
}

function compare(got, want) {
  let maxAbs = 0;
  let nDiff = 0;
  let firstAt = -1;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > 0) { nDiff++; if (firstAt < 0) firstAt = i; }
    if (d > maxAbs) maxAbs = d;
  }
  return { maxAbs, nDiff, firstAt, total: want.length };
}

const variants = [];
for (const [rndName] of Object.entries(rounders)) {
  variants.push({ name: `H_unshifted / V>>22  [${rndName}]`, shiftH: 0, roundH: 0, shiftV: 22, roundV: 1 << 21, rnd: rndName });
  variants.push({ name: `H>>11(r) / V>>11(r)  [${rndName}]`, shiftH: 11, roundH: 1 << 10, shiftV: 11, roundV: 1 << 10, rnd: rndName });
  variants.push({ name: `H>>11(t) / V>>11(r)  [${rndName}]`, shiftH: 11, roundH: 0, shiftV: 11, roundV: 1 << 10, rnd: rndName });
  variants.push({ name: `H>>11(r) / V>>11(t)  [${rndName}]`, shiftH: 11, roundH: 1 << 10, shiftV: 11, roundV: 0, rnd: rndName });
}

for (const im of meta.images) {
  const src = fs.readFileSync(path.join(EVID, `tensor_src_${im.stem}.rgba`));
  const buf = fs.readFileSync(path.join(EVID, `tensor_lb_${im.stem}.f32`));
  const want = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  console.log(`\n=== ${im.name}  ${im.srcW}x${im.srcH} -> r=${im.r.toFixed(6)} left=${im.left} top=${im.top} ===`);
  const rows = [];
  for (const v of variants) {
    const lb = letterBoxVariant(src, im.srcW, im.srcH, SIZE, v);
    const got = toNCHW(lb.data, SIZE, SIZE);
    const c = compare(got, want);
    rows.push({ variant: v.name, ...c });
  }
  rows.sort((a, b) => a.nDiff - b.nDiff || a.maxAbs - b.maxAbs);
  for (const row of rows) {
    const tag = row.nDiff === 0 ? '  <== EXACT' : '';
    console.log(`  ${row.variant.padEnd(38)} diff=${String(row.nDiff).padStart(8)}/${row.total}  maxAbs=${row.maxAbs.toExponential(3)}${tag}`);
  }
}

// ------------------------------------------------------------------ shipped code
// Re-check the variant that is actually compiled into assets/js/pipeline.js, so the
// guarantee attaches to the delivered module rather than to a copy in this script.
const P = await import(pathToFileURL(path.join(ROOT, 'assets', 'js', 'pipeline.js')).href);
console.log('\n=== shipped assets/js/pipeline.js ===');
for (const im of meta.images) {
  const src = fs.readFileSync(path.join(EVID, `tensor_src_${im.stem}.rgba`));
  const buf = fs.readFileSync(path.join(EVID, `tensor_lb_${im.stem}.f32`));
  const want = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

  const image = { data: src, width: im.srcW, height: im.srcH };
  const lb = P.letterBox(image, SIZE);
  const got = P.toNCHW(lb, true);
  const c = compare(got, want);
  const params = lb.r === im.r && lb.left === im.left && lb.top === im.top ? 'params OK' : 'PARAM DIFF';
  const verdict = c.nDiff === 0 ? 'BIT-EXACT' : 'DIFFERS';
  console.log(`  ${im.name.padEnd(16)} ${params}  ${verdict}  diff=${c.nDiff}/${c.total}  maxAbs=${c.maxAbs.toExponential(3)}`);
}
