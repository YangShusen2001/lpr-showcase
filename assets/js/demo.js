/**
 * Live demo controller for the showcase page.
 *
 * Loads onnxruntime-web plus the three HyperLPR3 models lazily (first interaction or
 * first scroll into view), runs the ported pipeline from assets/js/pipeline.js and
 * renders every intermediate the pipeline actually produces — no mock-ups.
 *
 * Deliberately offline: ORT and the models are served from this repository, so the
 * page works with no network access. It does need an HTTP origin (fetch of .onnx and
 * .wasm is blocked on file://), which the loader reports explicitly instead of
 * leaving a blank panel.
 *
 * Debug hooks:
 *   ?auto=1        run the first sample automatically once models are ready
 *   window.__DEMO  { ready, running, results, errors }
 */
import * as P from './pipeline.js';

// Sample URLs must be resolved against this module, not against the document:
// `src="../samples/x.jpg"` in markup resolves relative to the page (which lives
// at the site root), so it 404s. import.meta.url is the only stable anchor.
const SAMPLES_BASE = new URL('../samples/', import.meta.url).href;

const SAMPLES = [
  { file: 'scene-2.jpg', label: '整车 · 黄牌', note: '1140×456' },
  { file: 'hlpr-test.jpg', label: '整车 · 真值 8 位', note: '1920×1080 · 牌色两口径未裁决' },
  { file: 'crop-0-津B6H920.jpg', label: '特写 · 蓝牌', note: '7 位' },
  { file: 'crop-1-皖KD01833.jpg', label: '特写 · 绿牌', note: '8 位新能源' },
  { file: 'crop-6-蒙B023H6.jpg', label: '特写 · 蓝牌', note: '7 位' },
  { file: 'crop-8-冀D5L690.jpg', label: '特写 · 黄牌', note: '7 位' },
];

const COLOUR_HEX = { blue: '#1a63c4', green: '#12a150', yellow: '#dda000' };
const COLOUR_CN = { blue: '蓝牌', green: '绿牌', yellow: '黄牌' };

const $ = (id) => document.getElementById(id);
const el = {
  drop: $('drop'), file: $('file'), samples: $('samples'), log: $('log'),
  src: $('cvSrc'), crop: $('cvCrop'),
  plateText: $('plateText'), plateMeta: $('plateMeta'),
  clsBars: $('clsBars'), charBars: $('charBars'),
  msDet: $('msDet'), msRect: $('msRect'), msRec: $('msRec'), msCls: $('msCls'),
  msTotal: $('msTotal'), rtState: $('rtState'), empty: $('demoEmpty'),
  rtTitle: $('rtTitle'),
};

window.__DEMO = { ready: false, running: false, results: [], errors: [], samples: SAMPLES.length };

/**
 * Programmatic entry points.
 *
 * Two callers need to drive this page without touching its DOM:
 *   - the portrait shell (mobile.html) camera loop, which feeds captured frames
 *   - a PC-side driver reaching in over ArkWeb DevTools / CDP (Runtime.evaluate),
 *     which is how "computer orchestrates, phone computes" is wired
 * Both take the same arguments as the UI path, so a frame pushed from either
 * caller runs the identical pipeline and lands in the same evidence object.
 */
window.__LPR = { run, bench, ensureModels, samples: SAMPLES, samplesBase: SAMPLES_BASE };

let sessions = null;
let loading = null;
let activeSample = null;
/** Effective ORT wasm thread count, decided once in ensureModels(). */
let effThreads = 1;

// ------------------------------------------------------------------ logging
function logLine(text, cls = '') {
  if (!el.log) return;
  const t = new Date().toTimeString().slice(0, 8);
  const span = cls ? `<span class="${cls}">${text}</span>` : text;
  el.log.innerHTML += `<span class="dim">[${t}]</span> ${span}\n`;
  el.log.scrollTop = el.log.scrollHeight;
}

function setState(text) {
  if (el.rtState) el.rtState.textContent = text;
}

// ------------------------------------------------------------------ model load
function isFileProtocol() {
  return location.protocol === 'file:';
}

/**
 * onnxruntime-web forwards the C++ graph checker's warnings through console.error,
 * and these HyperLPR3 graphs declare hundreds of initialisers as graph *inputs*,
 * so DevTools fills with ~200 red lines that are entirely benign. Mute console
 * warn/error only while the sessions are being built, then restore.
 */
async function quiet(fn) {
  const warn = console.warn;
  const error = console.error;
  console.warn = () => {};
  console.error = () => {};
  try {
    return await fn();
  } finally {
    console.warn = warn;
    console.error = error;
  }
}

async function ensureModels() {
  if (sessions) return sessions;
  if (loading) return loading;

  loading = (async () => {
    if (isFileProtocol()) {
      throw new Error(
        'file:// 协议下浏览器禁止读取 .onnx / .wasm。请用项目自带的一键启动脚本通过 http://localhost 打开本页。',
      );
    }
    const t0 = performance.now();
    logLine('loading onnxruntime-web …', 'hl');
    // The runtime assets ship as .js / .onnx.json rather than .mjs / .onnx because
    // some preview hosts serve a static extension whitelist that 404s on .mjs and
    // .onnx. Browsers only care about the bytes, never the extension.
    const ort = await import('../ort/ort.min.js');
    // Object form (not a directory string) so ORT loads the renamed module loader.
    ort.env.wasm.wasmPaths = {
      mjs: new URL('../ort/ort-wasm-simd-threaded.js', import.meta.url).href,
      wasm: new URL('../ort/ort-wasm-simd-threaded.wasm', import.meta.url).href,
    };
    // Multithreaded WASM needs SharedArrayBuffer, which needs COOP/COEP response
    // headers. A plain static host will not send them, so the public web build stays
    // single-threaded; the HarmonyOS shell injects COOP/COEP itself (ADR-004 §5.3),
    // which flips crossOriginIsolated to true and unlocks every core.
    // ?threads=N pins the count — used by the on-device thread sweep.
    const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
    const cores = Math.max(1, navigator.hardwareConcurrency || 1);
    const forced = Number.parseInt(new URLSearchParams(location.search).get('threads') || '', 10);
    ort.env.wasm.numThreads = isolated
      ? (Number.isFinite(forced) && forced >= 1 ? Math.min(forced, cores) : cores)
      : 1;
    effThreads = ort.env.wasm.numThreads;
    ort.env.logLevel = 'error';

    // format is pinned so the .onnx.json extension cannot mislead the format guesser.
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', format: 'onnx' };
    const base = new URL('../models/', import.meta.url).href;
    logLine('loading detector y5fu_320x_sim …');
    const det = await quiet(() => ort.InferenceSession.create(base + 'y5fu_320x_sim.onnx.json', opt));
    logLine('loading recogniser rpv3_mdict_160_r3 …');
    const rec = await quiet(() => ort.InferenceSession.create(base + 'rpv3_mdict_160_r3.onnx.json', opt));
    logLine('loading classifier litemodel_cls_96x_r1 …');
    const cls = await quiet(() => ort.InferenceSession.create(base + 'litemodel_cls_96x_r1.onnx.json', opt));

    const ms = Math.round(performance.now() - t0);
    sessions = { ort, det, rec, cls };
    window.__DEMO.ready = true;
    window.__DEMO.modelLoadMs = ms;
    window.__DEMO.threads = effThreads;
    window.__DEMO.isolated = isolated;
    const tn = `${effThreads} thread${effThreads > 1 ? 's' : ''}`;
    logLine(`3 models ready in ${ms} ms  (wasm, ${tn}, FP32)`, 'ok');
    // The console header in demo.html used to hard-code "1 thread"; keep it honest.
    if (el.rtTitle) el.rtTitle.textContent = `hyperlpr3 · browser-runtime · wasm/simd · ${tn} · fp32`;
    return sessions;
  })();

  try {
    return await loading;
  } catch (e) {
    loading = null;
    throw e;
  }
}

// ------------------------------------------------------------------ image io
async function loadImage(fileOrUrl) {
  const bmp = typeof fileOrUrl === 'string'
    ? await createImageBitmap(await (await fetch(fileOrUrl)).blob())
    : await createImageBitmap(fileOrUrl);
  const cv = document.createElement('canvas');
  cv.width = bmp.width;
  cv.height = bmp.height;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  return { canvas: cv, img: ctx.getImageData(0, 0, cv.width, cv.height) };
}

// ------------------------------------------------------------------ rendering
function drawSource(canvas, plates) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const maxW = 520;
  const scale = Math.min(1, maxW / canvas.width);
  const w = Math.round(canvas.width * scale);
  const h = Math.round(canvas.height * scale);
  const out = el.src;
  out.width = Math.round(w * dpr);
  out.height = Math.round(h * dpr);
  out.style.width = w + 'px';
  const ctx = out.getContext('2d');
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
  ctx.drawImage(canvas, 0, 0);

  const lw = Math.max(1.5, 2.4 / scale);
  for (const p of plates) {
    ctx.lineWidth = lw;
    ctx.strokeStyle = '#ff3b5c';
    ctx.strokeRect(p.rect[0], p.rect[1], p.rect[2] - p.rect[0], p.rect[3] - p.rect[1]);

    ctx.fillStyle = '#22e06a';
    const r = Math.max(2, 4 / scale);
    for (const [mx, my] of p.marks) {
      ctx.beginPath();
      ctx.arc(mx, my, r, 0, Math.PI * 2);
      ctx.fill();
    }
    // quad outline through the keypoints, in detection order
    ctx.strokeStyle = 'rgba(34, 224, 106, .55)';
    ctx.lineWidth = Math.max(1, 1.2 / scale);
    ctx.beginPath();
    p.marks.forEach(([mx, my], i) => (i ? ctx.lineTo(mx, my) : ctx.moveTo(mx, my)));
    ctx.closePath();
    ctx.stroke();
  }
}

function drawCrop(crop) {
  const out = el.crop;
  if (!crop) { out.width = 1; out.height = 1; return; }
  const scale = Math.max(1, Math.min(6, Math.floor(420 / crop.width)));
  out.width = crop.width * scale;
  out.height = crop.height * scale;
  out.style.width = Math.min(420, out.width) + 'px';
  const ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  const tmp = document.createElement('canvas');
  tmp.width = crop.width;
  tmp.height = crop.height;
  tmp.getContext('2d').putImageData(new ImageData(crop.data, crop.width, crop.height), 0, 0);
  ctx.drawImage(tmp, 0, 0, out.width, out.height);
}

function renderPlate(p, cls) {
  el.plateText.innerHTML = p.chars
    .map((c, i) => `<span class="ph-char" title="${(p.charProbs[i] * 100).toFixed(1)}%">${c}</span>`)
    .join('');

  const idx = cls.indexOf(Math.max(...cls));
  const colour = P.PLATE_COLOURS[idx];
  const layer = p.layer === P.DOUBLE_LAYER ? '双层' : '单层';
  el.plateMeta.innerHTML =
    `<span class="badge"><i class="sw" style="background:${COLOUR_HEX[colour]}"></i>${COLOUR_CN[colour]}`
    + ` ${(cls[idx] * 100).toFixed(1)}%</span> `
    + `<span class="badge">${layer}</span> `
    + `<span class="badge">det ${(p.detScore * 100).toFixed(1)}%</span> `
    + `<span class="badge">${p.cropShape[1]}×${p.cropShape[0]} px</span>`;

  el.clsBars.innerHTML = P.PLATE_COLOURS.map((name, i) => {
    const v = cls[i];
    const k = v > 0.66 ? 'good' : v > 0.33 ? 'mid' : '';
    return `<div class="bar ${k}"><span class="bl">${COLOUR_CN[name]}</span>`
      + `<span class="bt"><span class="bf" style="width:${(v * 100).toFixed(1)}%"></span></span>`
      + `<span class="bv">${(v * 100).toFixed(1)}%</span></div>`;
  }).join('');

  el.charBars.innerHTML = p.chars.map((c, i) => {
    const v = p.charProbs[i];
    const k = v > 0.9 ? 'good' : v > 0.6 ? 'mid' : '';
    return `<div class="bar ${k}"><span class="bl">${i + 1} · ${c}</span>`
      + `<span class="bt"><span class="bf" style="width:${(v * 100).toFixed(1)}%"></span></span>`
      + `<span class="bv">${(v * 100).toFixed(1)}%</span></div>`;
  }).join('');
}

function renderTimings(p) {
  const total = (p.tDetectMs || 0) + (p.tRectifyMs || 0) + (p.tRecogMs || 0);
  el.msDet.textContent = p.tDetectMs != null ? p.tDetectMs.toFixed(1) + ' ms' : '—';
  el.msRect.textContent = p.tRectifyMs != null ? p.tRectifyMs.toFixed(1) + ' ms' : '—';
  el.msRec.textContent = p.tRecogMs != null ? p.tRecogMs.toFixed(1) + ' ms' : '—';
  el.msCls.textContent = p.tClsMs != null ? p.tClsMs.toFixed(1) + ' ms' : '—';
  el.msTotal.textContent = total.toFixed(1) + ' ms';
}

function clearPanes() {
  el.plateText.textContent = '—';
  el.plateMeta.innerHTML = '';
  el.clsBars.innerHTML = '';
  el.charBars.innerHTML = '';
  el.msDet.textContent = el.msRect.textContent = el.msRec.textContent = el.msCls.textContent = '—';
  el.msTotal.textContent = '—';
  drawCrop(null);
}

// ------------------------------------------------------------------ run
async function run(source, label) {
  if (window.__DEMO.running) return;
  window.__DEMO.running = true;
  clearPanes();
  setState('running');

  try {
    const s = await ensureModels();
    logLine(`── ${label ?? 'input'} ──`, 'hl');

    const t0 = performance.now();
    const { canvas, img } = await loadImage(source);
    const tLoad = performance.now() - t0;
    logLine(`decode ${canvas.width}×${canvas.height} in ${tLoad.toFixed(1)} ms`);

    const t1 = performance.now();
    const plates = await P.runPipeline(img, s, {
      detSize: 320, full: true, captureCrops: true,
    });
    const tAll = performance.now() - t1;

    if (!plates.length) {
      logLine('no plate above the 0.25 objectness threshold', 'warn');
      drawSource(canvas, []);
      el.plateText.textContent = '未检出';
      setState('no plate');
      window.__DEMO.results.push({ label, plates: 0 });
      return;
    }

    drawSource(canvas, plates);
    logLine(`${plates.length} plate(s) detected`, 'ok');

    // classify timing is not part of runPipeline's per-stage clocks; measure it once
    const cls0 = performance.now();
    await s.cls.run({
      [s.cls.inputNames[0]]: new s.ort.Tensor('float32', P.encodeClassify(plates[0].crop), [1, 3, 96, 96]),
    });
    plates[0].tClsMs = performance.now() - cls0;

    plates.forEach((p, i) => {
      logLine(
        `plate ${i + 1}  code=${p.code}  rec=${(p.recConf * 100).toFixed(1)}%  `
        + `det=${(p.detScore * 100).toFixed(1)}%  layer=${p.layer}  crop=${p.cropShape[1]}×${p.cropShape[0]}`,
        'ok',
      );
    });

    const best = plates.slice().sort((a, b) => b.detScore - a.detScore)[0];
    renderPlate(best, best.cls);
    renderTimings(best);
    drawCrop(best.crop);

    logLine(
      `pipeline total ${tAll.toFixed(1)} ms  `
      + `(det ${best.tDetectMs} + rectify ${best.tRectifyMs} + rec ${best.tRecogMs} ms)`,
      'hl',
    );

    setState(`${best.code} · ${tAll.toFixed(0)} ms`);
    window.__DEMO.results.push({ label, plates: plates.length, code: best.code, ms: Number(tAll.toFixed(1)) });
  } catch (e) {
    const msg = String((e && e.message) || e);
    window.__DEMO.errors.push(msg);
    logLine('ERROR ' + msg, 'err');
    setState('error');
  } finally {
    window.__DEMO.running = false;
  }
}

// ------------------------------------------------------------------ bench
/**
 * ?bench=N → time the full pipeline N times on the first sample and publish
 * p50 / mean / min / max to window.__BENCH (plus the effective thread count).
 *
 * Exists for the on-device thread sweep (ADR-004 §8): the HarmonyOS shell loads
 * this page once per thread count and reads __BENCH back through runJavaScript.
 * Inert on the public page, where no ?bench is present.
 */
async function bench(n) {
  const s = await ensureModels();
  const url = `${SAMPLES_BASE}${encodeURIComponent(SAMPLES[0].file)}`;
  const { img } = await loadImage(url);
  // one warm-up run so first-call kernel compilation does not land in the samples
  await P.runPipeline(img, s, { detSize: 320, full: true });

  const times = [];
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    await P.runPipeline(img, s, { detSize: 320, full: true });
    times.push(performance.now() - t);
  }

  const sorted = times.slice().sort((a, b) => a - b);
  const at = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  window.__BENCH = {
    done: true,
    threads: effThreads,
    isolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true,
    sab: typeof SharedArrayBuffer !== 'undefined',
    hardwareConcurrency: navigator.hardwareConcurrency || 1,
    n,
    p50: +at(0.5).toFixed(1),
    mean: +(times.reduce((a, b) => a + b, 0) / n).toFixed(1),
    min: +sorted[0].toFixed(1),
    max: +sorted[sorted.length - 1].toFixed(1),
  };
  logLine('BENCH ' + JSON.stringify(window.__BENCH), 'hl');
}

// ------------------------------------------------------------------ wiring
function buildSamples() {
  el.samples.innerHTML = SAMPLES.map((s, i) => `
    <div class="sample" data-i="${i}" title="${s.label} · ${s.note}">
      <img src="${SAMPLES_BASE}${encodeURIComponent(s.file)}" alt="${s.label}" loading="lazy" />
      <span>${s.label}</span>
    </div>`).join('');

  el.samples.addEventListener('click', (ev) => {
    const box = ev.target.closest('.sample');
    if (!box) return;
    [...el.samples.children].forEach((c) => c.classList.toggle('active', c === box));
    const s = SAMPLES[Number(box.dataset.i)];
    activeSample = s;
    run(`${SAMPLES_BASE}${encodeURIComponent(s.file)}`, s.label);
  });
}

function wireDrop() {
  el.drop.addEventListener('click', () => el.file.click());
  el.file.addEventListener('change', () => {
    if (el.file.files && el.file.files[0]) {
      [...el.samples.children].forEach((c) => c.classList.remove('active'));
      activeSample = null;
      run(el.file.files[0], el.file.files[0].name);
    }
  });
  ['dragenter', 'dragover'].forEach((t) =>
    el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) =>
    el.drop.addEventListener(t, () => el.drop.classList.remove('over')));
  el.drop.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) { activeSample = null; run(f, f.name); }
  });
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const it of items) {
      if (it.type.startsWith('image/')) {
        const f = it.getAsFile();
        if (f) { activeSample = null; run(f, 'clipboard'); }
        break;
      }
    }
  });
}

/** Kick model loading off the first time the demo scrolls into view. */
function wireLazyLoad() {
  const target = document.querySelector('.demo-shell');
  if (!target || !('IntersectionObserver' in window)) return;
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      logLine('demo in view — prefetching models in the background …', 'hl');
      ensureModels().catch((e) => {
        logLine('model prefetch failed: ' + String(e.message || e), 'err');
        setState('model load failed');
      });
    }
  }, { rootMargin: '200px' });
  io.observe(target);
}

function init() {
  buildSamples();
  wireDrop();
  wireLazyLoad();
  logLine('端到端车牌识别 · 浏览器实时推理控制台', 'hl');
  logLine('模型：HyperLPR3 ONNX 三件套（检测 / 识别 / 颜色分类）', 'dim');
  logLine('选一张样本或拖入自己的照片即可开始 —— 全部推理在本机浏览器完成，无网络请求。', 'dim');
  setState('idle');

  const params = new URLSearchParams(location.search);

  // ?bench=N takes precedence over ?auto — it is the on-device sweep entry point.
  const benchN = Number.parseInt(params.get('bench') || '', 10);
  if (Number.isFinite(benchN) && benchN > 0) {
    bench(benchN).catch((e) => {
      window.__BENCH = { done: true, error: String((e && e.message) || e) };
      logLine('BENCH ERROR ' + window.__BENCH.error, 'err');
    });
    return;
  }

  if (params.get('auto')) {
    ensureModels()
      .then(() => {
        const s = SAMPLES[0];
        const box = el.samples.querySelector('.sample');
        if (box) box.classList.add('active');
        activeSample = s;
        return run(`${SAMPLES_BASE}${encodeURIComponent(s.file)}`, s.label);
      })
      .then(() => { window.__DEMO.autoDone = true; })
      .catch((e) => { window.__DEMO.errors.push('auto: ' + String(e.message || e)); window.__DEMO.autoDone = true; });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
