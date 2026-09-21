/**
 * HyperLPR3 pipeline — browser port.
 *
 * A faithful JavaScript transcription of the upstream Python/C++ algorithm, kept
 * deliberately free of DOM dependencies (pure functions over plain
 * {data,width,height} buffers) so it can be unit-tested and diffed against the
 * Python reference in tools/hlpr_reference.py.
 *
 * Upstream sources transcribed here (HyperLPR, Apache-2.0):
 *   inference/multitask_detect.py  -> letterBox / detectPreprocess / postProcessing / nms / restoreBox
 *   common/tools_process.py        -> get_rotate_crop_image
 *   inference/recognition.py       -> encodeImages / CTC greedy decode
 *   inference/classification.py    -> encodeImages (no channel swap)
 *
 * Numeric fidelity notes (intentional, verified against the Python reference):
 *   - cv2.resize default INTER_LINEAR -> bilinear with OpenCV's half-pixel mapping
 *   - cv2.warpPerspective INTER_CUBIC -> bicubic kernel with a = -0.75
 *   - Channel order: detector expects RGB; recogniser and classifier expect BGR.
 */

// ---------------------------------------------------------------- charset
export const TOKEN = [
  'blank', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '民', '沪',
  '津', '浙', '渝', '港', '湘', '琼', '甘', '皖', '粤', '航', '苏', '蒙', '藏',
  '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑', '领', '使', '澳',
];

/** Layer class index that means "double-layer plate" (upstream typedef.DOUBLE). */
export const DOUBLE_LAYER = 1;

/** Plate colour class order produced by litemodel_cls_96x_r1.onnx. */
export const PLATE_COLOURS = ['yellow', 'blue', 'green'];

// ---------------------------------------------------------------- buffers
export function createImageData(width, height) {
  return { data: new Uint8ClampedArray(width * height * 4), width, height };
}

function makeCanvas(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  return c;
}

export function canvasToImageData(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

export function imageDataToCanvas(img) {
  const canvas = makeCanvas(img.width, img.height);
  canvas.getContext('2d').putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  return canvas;
}

// ---------------------------------------------------------------- resize (cv2.INTER_LINEAR)
/** OpenCV's fixed-point scale for resize coefficients (INTER_RESIZE_COEF_BITS = 11). */
const COEF_BITS = 11;
const COEF_SCALE = 1 << COEF_BITS;

/**
 * Bilinear resize replicating cv2.resize(..., INTER_LINEAR) bit for bit.
 *
 * OpenCV does *not* interpolate in floating point. It builds two-pass separable
 * fixed-point weights and, crucially, keeps the horizontal pass **unshifted** in an
 * integer buffer before a single `(v + 2^21) >> 22` rounding in the vertical pass.
 * Rounding each pass independently loses a fraction of a level and shifts the result
 * by one grey level for ~6% of pixels — enough to move a detector keypoint across an
 * integer boundary and flip a character. Verified against cv2 with
 * tools/fit_resize.mjs: 0 differing values out of 307200.
 *
 * Mapping: src = (dst + 0.5) * scale - 0.5, with edge clamping.
 */
export function resizeLinear(src, dstW, dstH) {
  const { data, width: sw, height: sh } = src;
  if (dstW === sw && dstH === sh) {
    const copy = new Uint8ClampedArray(data);
    return { data: copy, width: dstW, height: dstH };
  }

  const xo = new Int32Array(dstW);
  const cx = new Int32Array(dstW * 2);
  for (let x = 0; x < dstW; x++) {
    let fx = ((x + 0.5) * sw) / dstW - 0.5;
    let sx = Math.floor(fx);
    fx -= sx;
    if (sx < 0) { fx = 0; sx = 0; }
    if (sx >= sw - 1) { fx = 0; sx = sw - 1; }
    xo[x] = sx;
    cx[x * 2] = Math.round((1 - fx) * COEF_SCALE);
    cx[x * 2 + 1] = Math.round(fx * COEF_SCALE);
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
    cy[y * 2] = Math.round((1 - fy) * COEF_SCALE);
    cy[y * 2 + 1] = Math.round(fy * COEF_SCALE);
  }

  // horizontal pass: integer intermediate, deliberately left unscaled
  const tmp = new Int32Array(dstW * sh * 4);
  for (let y = 0; y < sh; y++) {
    const row = y * sw * 4;
    const orow = y * dstW * 4;
    for (let x = 0; x < dstW; x++) {
      // The +1 tap is clamped so a 1-pixel axis (or a unit scale) cannot read past
      // the buffer; when clamping kicks in the matching coefficient is 0 anyway.
      const s0 = row + xo[x] * 4;
      const s1 = row + Math.min(xo[x] + 1, sw - 1) * 4;
      const c0 = cx[x * 2];
      const c1 = cx[x * 2 + 1];
      const o = orow + x * 4;
      tmp[o] = data[s0] * c0 + data[s1] * c1;
      tmp[o + 1] = data[s0 + 1] * c0 + data[s1 + 1] * c1;
      tmp[o + 2] = data[s0 + 2] * c0 + data[s1 + 2] * c1;
      tmp[o + 3] = data[s0 + 3] * c0 + data[s1 + 3] * c1;
    }
  }

  // vertical pass: single rounding shift of 2 * COEF_BITS
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const stride = dstW * 4;
  const ROUND = 1 << (COEF_BITS * 2 - 1);
  for (let y = 0; y < dstH; y++) {
    const r0 = yo[y] * stride;
    const r1 = Math.min(yo[y] + 1, sh - 1) * stride;
    const c0 = cy[y * 2];
    const c1 = cy[y * 2 + 1];
    const o = y * stride;
    for (let i = 0; i < stride; i++) {
      out[o + i] = (tmp[r0 + i] * c0 + tmp[r1 + i] * c1 + ROUND) >> (COEF_BITS * 2);
    }
  }
  return { data: out, width: dstW, height: dstH };
}

// ---------------------------------------------------------------- detection preprocess
/** Upstream multitask_detect.letter_box: integer padding offsets, black border. */
export function letterBox(src, size) {
  const h = src.height;
  const w = src.width;
  const r = Math.min(size / h, size / w);
  const newH = Math.trunc(h * r);
  const newW = Math.trunc(w * r);
  const top = Math.trunc((size - newH) / 2);
  const left = Math.trunc((size - newW) / 2);

  const resized = resizeLinear(src, newW, newH);
  const out = new Uint8ClampedArray(size * size * 4); // zero-filled == (0,0,0) border
  for (let y = 0; y < newH; y++) {
    const srcOff = y * newW * 4;
    const dstOff = ((y + top) * size + left) * 4;
    out.set(resized.data.subarray(srcOff, srcOff + newW * 4), dstOff);
  }
  return { data: out, width: size, height: size, r, left, top };
}

/**
 * NCHW float tensor. `swapRB` mirrors upstream's channel handling: the detector
 * flips BGR->RGB, the recogniser and classifier keep BGR as-is.
 */
export function toNCHW(img, swapRB) {
  const { data, width: w, height: h } = img;
  const plane = w * h;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    const r = data[p];
    const g = data[p + 1];
    const b = data[p + 2];
    out[i] = (swapRB ? r : b) / 255;
    out[plane + i] = g / 255;
    out[2 * plane + i] = (swapRB ? b : r) / 255;
  }
  return out;
}

// ---------------------------------------------------------------- detection postprocess
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Upstream multitask_detect.nms: greedy, descending score, strict > iouThresh keeps. */
function nmsKeep(rows, iouThresh) {
  const order = rows.map((_, i) => i).sort((a, b) => rows[b][4] - rows[a][4]);
  const keep = [];
  const alive = order.slice();
  while (alive.length > 0) {
    const i = alive.shift();
    keep.push(i);
    const rest = [];
    for (const j of alive) {
      if (iou(rows[i], rows[j]) <= iouThresh) rest.push(j);
    }
    alive.length = 0;
    alive.push(...rest);
  }
  return keep;
}

/**
 * Upstream multitask_detect.post_precessing. `raw` is the model output reshaped to
 * [N, 15] laid out as [cx, cy, w, h, obj, kp0x..kp3y, cls0, cls1]; the returned rows
 * are [x1, y1, x2, y2, score, kp0x..kp3y, layer] with 14 columns.
 */
export function decodeDetections(raw, rows, confThresh = 0.25, iouThresh = 0.5, r = 1, left = 0, top = 0) {
  const cand = [];
  for (let i = 0; i < rows; i++) {
    const o = i * 15;
    const obj = raw[o + 4];
    if (!(obj > confThresh)) continue;
    const s0 = raw[o + 13] * obj;
    const s1 = raw[o + 14] * obj;
    const score = Math.max(s0, s1);
    const layer = s1 > s0 ? 1 : 0;
    const cx = raw[o];
    const cy = raw[o + 1];
    const bw = raw[o + 2];
    const bh = raw[o + 3];
    const row = new Array(14);
    row[0] = cx - bw / 2;
    row[1] = cy - bh / 2;
    row[2] = cx + bw / 2;
    row[3] = cy + bh / 2;
    row[4] = score;
    for (let k = 0; k < 8; k++) row[5 + k] = raw[o + 5 + k];
    row[13] = layer;
    cand.push(row);
  }
  if (cand.length === 0) return [];

  const kept = nmsKeep(cand, iouThresh).map((i) => cand[i]);

  // restore_box: undo padding then scale, on x-indices [0,2,5,7,9,11] and y-indices [1,3,6,8,10,12]
  for (const row of kept) {
    for (const ix of [0, 2, 5, 7, 9, 11]) row[ix] = (row[ix] - left) / r;
    for (const iy of [1, 3, 6, 8, 10, 12]) row[iy] = (row[iy] - top) / r;
  }
  return kept;
}

// ---------------------------------------------------------------- perspective rectification
/** Solve the 8 unknowns of the homography mapping src -> dst (cv2.getPerspectiveTransform). */
export function getPerspectiveTransform(src, dst) {
  const a = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = src[i];
    const [dx, dy] = dst[i];
    a.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
    a.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    b.push(dx, dy);
  }
  const h = solve8(a, b);
  return h;
}

/** Gaussian elimination with partial pivoting on an 8x8 system. */
function solve8(a, b) {
  const n = 8;
  const m = a.map((row, i) => row.concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    }
    if (piv !== col) {
      const t = m[piv];
      m[piv] = m[col];
      m[col] = t;
    }
    const d = m[col][col];
    if (Math.abs(d) < 1e-12) continue;
    for (let r = col + 1; r < n; r++) {
      const f = m[r][col] / d;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) m[r][c] -= f * m[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r][n];
    for (let c = r + 1; c < n; c++) s -= m[r][c] * x[c];
    x[r] = Math.abs(m[r][r]) < 1e-12 ? 0 : s / m[r][r];
  }
  return [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1];
}

export function invert3x3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  return [
    A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det,
  ];
}

function cubicWeight(t) {
  const a = -0.75;
  const x = Math.abs(t);
  if (x <= 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  return 0;
}

/**
 * Warp with the inverse homography and bicubic sampling — the equivalent of
 * cv2.warpPerspective(..., flags=INTER_CUBIC, borderMode=BORDER_REPLICATE).
 */
export function warpPerspectiveCubic(src, hInv, outW, outH) {
  const { data, width: sw, height: sh } = src;
  const out = new Uint8ClampedArray(outW * outH * 4);
  const [h11, h12, h13, h21, h22, h23, h31, h32, h33] = hInv;

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const dz = h31 * x + h32 * y + h33;
      const sx = (h11 * x + h12 * y + h13) / dz;
      const sy = (h21 * x + h22 * y + h23) / dz;

      const ix = Math.floor(sx);
      const iy = Math.floor(sy);
      const o = (y * outW + x) * 4;

      for (let c = 0; c < 4; c++) {
        let acc = 0;
        let wsum = 0;
        for (let m = -1; m <= 2; m++) {
          const wy = cubicWeight(sy - (iy + m));
          if (wy === 0) continue;
          let py = iy + m;
          if (py < 0) py = 0;
          else if (py > sh - 1) py = sh - 1;
          for (let n = -1; n <= 2; n++) {
            const wx = cubicWeight(sx - (ix + n));
            if (wx === 0) continue;
            let px = ix + n;
            if (px < 0) px = 0;
            else if (px > sw - 1) px = sw - 1;
            const wgt = wx * wy;
            acc += data[(py * sw + px) * 4 + c] * wgt;
            wsum += wgt;
          }
        }
        out[o + c] = wsum === 0 ? 0 : acc / wsum;
      }
      out[o + 3] = 255;
    }
  }
  return { data: out, width: outW, height: outH };
}

/**
 * Upstream tools_process.get_rotate_crop_image: rectifies the quad to an
 * axis-aligned rectangle whose size is derived from the quad's edge lengths, and
 * rotates 90 degrees when the crop comes out portrait (h/w >= 1.5).
 */
export function rotateCrop(src, points) {
  const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const cropW = Math.trunc(Math.max(d(points[0], points[1]), d(points[2], points[3])));
  const cropH = Math.trunc(Math.max(d(points[0], points[3]), d(points[1], points[2])));
  if (cropW <= 0 || cropH <= 0) return null;

  const dst = [[0, 0], [cropW, 0], [cropW, cropH], [0, cropH]];
  const h = getPerspectiveTransform(points, dst);
  const warped = warpPerspectiveCubic(src, invert3x3(h), cropW, cropH);

  if (cropH / cropW >= 1.5) return rotate90(warped);
  return warped;
}

function rotate90(img) {
  const { data, width: w, height: h } = img;
  const out = new Uint8ClampedArray(w * h * 4);
  const nw = h;
  const nh = w;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // np.rot90: out[y', x'] = in[x, w-1-y]  (counter-clockwise)
      const ny = w - 1 - x;
      const nx = y;
      const s = (y * w + x) * 4;
      const o = (ny * nw + nx) * 4;
      out[o] = data[s];
      out[o + 1] = data[s + 1];
      out[o + 2] = data[s + 2];
      out[o + 3] = data[s + 3];
    }
  }
  return { data: out, width: nw, height: nh };
}

// ---------------------------------------------------------------- recognition
/**
 * Upstream recognition.encodeImages: aspect-preserving resize to height imgH,
 * width clamped to [limitedMinWidth, limitedMaxWidth], normalised to [-1, 1],
 * then right-padded with zeros. BGR channel order is preserved.
 */
export function encodePlate(crop, imgH = 48, imgW = 160, limitedMaxWidth = 160, limitedMinWidth = 48) {
  const h = crop.height;
  const w = crop.width;
  let maxWhRatio = Math.max(w / h, imgW / imgH);
  let targetW = Math.trunc(imgH * maxWhRatio);
  targetW = Math.max(Math.min(targetW, limitedMaxWidth), limitedMinWidth);

  const ratio = w / h;
  let ratioImgH = Math.ceil(imgH * ratio);
  ratioImgH = Math.max(ratioImgH, limitedMinWidth);
  const resizedW = ratioImgH > targetW ? targetW : Math.trunc(ratioImgH);

  const resized = resizeLinear(crop, resizedW, imgH);

  const plane = imgH * targetW;
  const out = new Float32Array(3 * plane);
  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < resizedW; x++) {
      const s = (y * resizedW + x) * 4;
      const i = y * targetW + x;
      out[i] = (resized.data[s + 2] - 127.5) / 127.5;           // B
      out[plane + i] = (resized.data[s + 1] - 127.5) / 127.5;    // G
      out[2 * plane + i] = (resized.data[s] - 127.5) / 127.5;    // R
    }
  }
  return { data: out, width: targetW, height: imgH };
}

/** Upstream recognition.decode: CTC greedy — drop blank(0) and repeated indices. */
export function ctcGreedy(indexRow, probRow) {
  const chars = [];
  const probs = [];
  for (let i = 0; i < indexRow.length; i++) {
    const idx = indexRow[i];
    if (idx === 0) continue;
    if (i > 0 && indexRow[i - 1] === idx) continue;
    chars.push(idx < TOKEN.length ? TOKEN[idx] : '?');
    probs.push(probRow[i]);
  }
  const conf = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0;
  return { code: chars.join(''), conf, chars, probs };
}

/** Argmax over the class axis of a [T, C] logit block. */
export function argmaxRows(logits, T, C) {
  const idx = new Array(T);
  const prob = new Array(T);
  for (let t = 0; t < T; t++) {
    let best = 0;
    let bv = -Infinity;
    const base = t * C;
    for (let c = 0; c < C; c++) {
      const v = logits[base + c];
      if (v > bv) {
        bv = v;
        best = c;
      }
    }
    idx[t] = best;
    prob[t] = bv;
  }
  return { idx, prob };
}

// ---------------------------------------------------------------- classification
export function encodeClassify(crop, size = 96) {
  const resized = resizeLinear(crop, size, size);
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const s = i * 4;
    out[i] = resized.data[s + 2] / 255;          // B (upstream keeps BGR)
    out[plane + i] = resized.data[s + 1] / 255;  // G
    out[2 * plane + i] = resized.data[s] / 255;  // R
  }
  return out;
}

// ---------------------------------------------------------------- high-level pipeline
function now() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

/**
 * Upstream casts keypoints with `.astype(int)` *before* measuring the quad's edge
 * lengths, so the rectified size is derived from integer corners. Skipping the
 * truncation shifts the crop by up to one pixel, which is enough to flip the
 * recogniser on small plates — hence this helper exists.
 */
export function truncateMarks(row) {
  const marks = [];
  for (let k = 0; k < 4; k++) {
    marks.push([Math.trunc(row[5 + k * 2]), Math.trunc(row[6 + k * 2])]);
  }
  return marks;
}

/** Contiguous copy of rows [y0, y1) — mirrors numpy's crop[y0:y1, :]. */
export function sliceRows(img, y0, y1) {
  const { data, width, height } = img;
  const a = Math.max(0, y0);
  const b = Math.min(height, y1);
  const out = new Uint8ClampedArray(Math.max(0, b - a) * width * 4);
  out.set(data.subarray(a * width * 4, b * width * 4));
  return { data: out, width, height: b - a };
}

/** Single-plate recognition: encode -> run -> argmax -> CTC greedy. */
export async function recognise(rec, ort, crop) {
  const enc = encodePlate(crop);
  const out = await rec.run({
    [rec.inputNames[0]]: new ort.Tensor('float32', enc.data, [1, 3, enc.height, enc.width]),
  });
  const lg = out[rec.outputNames[0]];
  const { idx, prob } = argmaxRows(lg.data, lg.dims[1], lg.dims[2]);
  return ctcGreedy(idx, prob);
}

/**
 * End-to-end pipeline, a line-for-line mirror of hlpr_reference.run_pipeline.
 *
 * sessions = { ort, det, rec, cls } — `ort` is injected so this module stays free
 * of imports and can be exercised in Node as well as the browser.
 * `img` is a {data,width,height} RGBA buffer.
 *
 * Pass a `debug` object to receive the raw detector tensor for numeric diffing.
 */
export async function runPipeline(img, sessions, opts = {}) {
  const { detSize = 320, confThresh = 0.25, iouThresh = 0.5, full = true, debug = null,
          captureCrops = false } = opts;
  const { ort, det, rec, cls } = sessions;
  const tensor = (data, dims) => new ort.Tensor('float32', data, dims);

  const t0 = now();
  const lb = letterBox(img, detSize);
  const detInput = toNCHW(lb, true);
  const dOut = await det.run({
    [det.inputNames[0]]: tensor(detInput, [1, 3, detSize, detSize]),
  });
  const raw = dOut[det.outputNames[0]];
  const dets = decodeDetections(raw.data, raw.dims[1], confThresh, iouThresh, lb.r, lb.left, lb.top);
  const t1 = now();
  if (debug) {
    debug.detect = {
      size: detSize, r: lb.r, left: lb.left, top: lb.top,
      rows: raw.dims[1], raw: raw.data, input: detInput,
    };
  }

  const results = [];
  for (const row of dets) {
    const marks = truncateMarks(row);
    const crop = rotateCrop(img, marks);
    if (!crop) continue;
    const t2 = now();

    let code = '';
    let rconf = 0;
    let rchars = [];
    let rprobs = [];
    if (row[13] === DOUBLE_LAYER) {
      const line = Math.trunc(crop.height * 0.4);
      const a = await recognise(rec, ort, sliceRows(crop, 0, line));
      const b = await recognise(rec, ort, sliceRows(crop, line, crop.height));
      code = a.code + b.code;
      rconf = (a.conf + b.conf) / 2;
      rchars = a.chars.concat(b.chars);
      rprobs = a.probs.concat(b.probs);
    } else {
      const r = await recognise(rec, ort, crop);
      code = r.code;
      rconf = r.conf;
      rchars = r.chars;
      rprobs = r.probs;
    }
    const t3 = now();
    if (code === '') continue;

    const item = {
      rect: row.slice(0, 4).map(Math.trunc),
      detScore: Number(row[4].toFixed(4)),
      marks,
      layer: row[13],
      code,
      recConf: Number(rconf.toFixed(4)),
      chars: rchars,
      charProbs: rprobs.map((v) => Number(v.toFixed(4))),
      cropShape: [crop.height, crop.width],
    };
    if (captureCrops) item.crop = crop;
    if (debug) {
      // RGB-only checksum (alpha excluded) so it is comparable to numpy's BGR sum
      let s = 0;
      for (let i = 0; i < crop.data.length; i++) if (i % 4 !== 3) s += crop.data[i];
      item.cropSum = s;
    }
    if (full) {
      const cOut = await cls.run({ [cls.inputNames[0]]: tensor(encodeClassify(crop), [1, 3, 96, 96]) });
      item.cls = Array.from(cOut[cls.outputNames[0]].data).map((v) => Number(v.toFixed(4)));
      item.tDetectMs = Number((t1 - t0).toFixed(2));
      item.tRectifyMs = Number((t2 - t1).toFixed(2));
      item.tRecogMs = Number((t3 - t2).toFixed(2));
    }
    results.push(item);
  }
  return results;
}

// ---------------------------------------------------------------- text helpers
/** Province / letter / tail split used for the pipeline's type filter. */
export function splitPlate(code) {
  if (!code) return { province: '', letter: '', tail: '' };
  return { province: code[0] ?? '', letter: code[1] ?? '', tail: code.slice(2) };
}
