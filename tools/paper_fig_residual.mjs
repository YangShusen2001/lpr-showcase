/**
 * Residual-error data for the paper's fidelity figure.
 *
 * Compares the **shipped** assets/js/pipeline.js letterbox against the OpenCV
 * ground-truth tensor dumped by tools/dump_tensor.py, and emits a per-channel
 * difference histogram as JSON (consumed by tools/paper_figures.py).
 *
 * Nothing here re-implements the port: it imports the delivered module, so the
 * figure describes the code that actually ships.
 *
 * Usage: node tools/paper_fig_residual.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const EVID = path.join(ROOT, '_evidence');

const meta = JSON.parse(fs.readFileSync(path.join(EVID, 'tensor_meta.json'), 'utf8'));
const P = await import(pathToFileURL(path.join(ROOT, 'assets', 'js', 'pipeline.js')).href);

const out = [];
for (const im of meta.images) {
  const src = fs.readFileSync(path.join(EVID, `tensor_src_${im.stem}.rgba`));
  const buf = fs.readFileSync(path.join(EVID, `tensor_lb_${im.stem}.f32`));
  const want = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

  const image = { data: src, width: im.srcW, height: im.srcH };
  const lb = P.letterBox(image, meta.size);
  const got = P.toNCHW(lb, true);

  const hist = new Map();
  let maxAbs = 0;
  let nDiff = 0;
  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < want.length; i++) {
    const d = got[i] - want[i];
    if (d !== 0) {
      nDiff++;
      if (d > 0) nPos++; else nNeg++;
    }
    const a = Math.abs(d);
    if (a > maxAbs) maxAbs = a;
    // quantise to the LSB grid: the tensor is uint8/255, so 1 LSB = 1/255
    const key = Math.round(d * 255);
    hist.set(key, (hist.get(key) || 0) + 1);
  }

  const paramsOk = lb.r === im.r && lb.left === im.left && lb.top === im.top;
  const rec = {
    name: im.name,
    stem: im.stem,
    srcW: im.srcW,
    srcH: im.srcH,
    scale: im.r,
    scaleExpr: `${im.srcW}->${Math.round(im.srcW * im.r)}`,
    total: want.length,
    nDiff,
    nPos,
    nNeg,
    maxAbs,
    paramsOk,
    hist: Object.fromEntries([...hist.entries()].sort((a, b) => Number(a[0]) - Number(b[0]))),
  };
  out.push(rec);
  console.log(`${im.name.padEnd(16)} r=${im.r.toFixed(6)}  diff=${nDiff}/${want.length}  `
    + `(+${nPos}/-${nNeg})  maxAbs=${maxAbs.toExponential(3)}  ${paramsOk ? 'params OK' : 'PARAM DIFF'}`);
}

const dst = path.join(EVID, 'paper_residual.json');
fs.writeFileSync(dst, JSON.stringify(out, null, 2), 'utf8');
console.log('\nwrote', dst);
