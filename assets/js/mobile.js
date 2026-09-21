/**
 * Glue for the portrait demo shell (mobile.html).
 *
 * The inference controller is assets/js/demo.js — it is loaded unchanged and
 * owns every number on this page. This file only adds phone-only affordances
 * and never touches the pipeline:
 *
 *   1. mirror #rtState into the status chip (idle / running / ok / error)
 *   2. paint the plate frame with the *classifier's* colour verdict, read from
 *      the badge demo.js already renders into #plateMeta — no extra inference,
 *      no hard-coded guess
 *   3. forward the camera-capture <input capture> into demo.js's single file
 *      input, so a real "拍照" button works without forking the controller
 *   4. wire the fixed bottom action bar
 */

const $ = (id) => document.getElementById(id);

// --------------------------------------------------------- 1. status chip
const stateEl = $('rtState');
const chip = $('mChip');

function paintChip() {
  if (!chip || !stateEl) return;
  const text = (stateEl.textContent || '').trim();
  let state = 'idle';
  if (/running|loading/i.test(text)) state = 'running';
  else if (/error|fail/i.test(text)) state = 'error';
  else if (text === 'no plate' || /·/.test(text)) state = 'ok';
  chip.dataset.state = state;
  chip.title = text;
}

if (stateEl && chip) {
  new MutationObserver(paintChip).observe(stateEl, {
    childList: true, characterData: true, subtree: true,
  });
  paintChip();
}

// --------------------------------------------------------- 2. plate colour
const frame = $('plateFrame');
const meta = $('plateMeta');
const COLOUR_OF = [['蓝牌', 'blue'], ['绿牌', 'green'], ['黄牌', 'yellow']];

function paintPlate() {
  if (!frame || !meta) return;
  const text = meta.textContent || '';
  const hit = COLOUR_OF.find(([cn]) => text.includes(cn));
  if (hit) frame.dataset.plate = hit[1];
  else delete frame.dataset.plate;
}

if (frame && meta) {
  new MutationObserver(paintPlate).observe(meta, {
    childList: true, subtree: true, characterData: true,
  });
  paintPlate();
}

// --------------------------------------------------------- 3. camera input
const fileInput = $('file');
const camInput = $('cam');

/**
 * demo.js wires exactly one <input type=file> (#file) and reads it on `change`.
 * A dedicated capture input gives a real camera button on Android/ArkWeb; copy
 * its File into #file through a DataTransfer and re-dispatch `change` so the
 * controller sees it as an ordinary pick.
 */
function forwardInto(input) {
  if (!input || !fileInput) return;
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    input.value = '';
    if (!file) return;
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event('change'));
    } catch (e) {
      // Older WebViews without the DataTransfer constructor: fall back to the
      // gallery picker rather than failing silently.
      fileInput.click();
    }
  });
}
forwardInto(camInput);

// --------------------------------------------------------- 4. action bar
const on = (id, fn) => { const b = $(id); if (b) b.addEventListener('click', fn); };

on('btnPick', () => fileInput && fileInput.click());
on('btnSamples', () => {
  const rail = $('samples');
  if (rail) rail.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

// --------------------------------------------------------- 5. live camera
/**
 * ArkWeb implements the W3C WebRTC capture path, so the page can hold a real
 * MediaStream — no native camera code needed. Two gates have to be open or
 * getUserMedia rejects with NotAllowedError: the app must hold
 * ohos.permission.CAMERA (requested in Index.ets), and the Web component must
 * grant VIDEO_CAPTURE in its onPermissionRequest callback.
 *
 * Frames are handed to demo.js through window.__LPR.run(), the same entry the
 * PC-side driver uses, so a captured frame and a pushed frame are the same run.
 */
const CAM = {
  stream: null,
  /** Idle gap between automatic grabs. The pipeline is ~85 ms on 6 threads; 1.5 s
   *  keeps the preview smooth and leaves the SoC well clear of thermal limits. */
  intervalMs: 1500,
  timer: 0,
  on: false,
};

const video = $('camVideo');
const camCard = $('camCard');
const camInfo = $('camInfo');
const camBtn = $('btnCam');
const camAuto = $('camAuto');
const inputCard = document.querySelector('.m-input');

function camLabel(text) {
  if (camBtn) camBtn.innerHTML = '<span class="g">&#9678;</span>' + text;
}

async function camStart() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (camInfo) camInfo.textContent = 'WebRTC 不可用';
    return;
  }
  try {
    CAM.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 } },
      audio: false,
    });
  } catch (e) {
    // No MediaStream (permission refused, or a build without the WebRTC path):
    // fall back to the system capture intent so the button still does something.
    if (camInfo) camInfo.textContent = String((e && e.name) || e);
    if (camInput) camInput.click();
    return;
  }
  video.srcObject = CAM.stream;
  await video.play().catch(() => {});
  if (camCard) camCard.hidden = false;
  if (inputCard) inputCard.hidden = true;
  CAM.on = true;
  camLabel('停止');
  if (camBtn) camBtn.classList.remove('primary');
  if (camInfo) {
    const s = CAM.stream.getVideoTracks()[0].getSettings();
    camInfo.textContent = `${s.width}×${s.height}`;
  }
  camSetAuto(true);
}

function camStop() {
  camSetAuto(false);
  if (CAM.stream) {
    for (const t of CAM.stream.getTracks()) t.stop();
    CAM.stream = null;
  }
  if (video) video.srcObject = null;
  if (camCard) camCard.hidden = true;
  if (inputCard) inputCard.hidden = false;
  CAM.on = false;
  camLabel('相机');
  if (camBtn) camBtn.classList.add('primary');
  if (camInfo) camInfo.textContent = '';
}

/** Grab one frame at the sensor's native resolution and run the pipeline on it. */
async function camGrab() {
  const v = video;
  if (!v || !v.videoWidth || window.__DEMO.running) return;
  const cv = document.createElement('canvas');
  cv.width = v.videoWidth;
  cv.height = v.videoHeight;
  cv.getContext('2d').drawImage(v, 0, 0);
  const blob = await new Promise((r) => cv.toBlob(r, 'image/jpeg', 0.92));
  if (blob && window.__LPR) window.__LPR.run(blob, `camera ${cv.width}×${cv.height}`);
}

function camSetAuto(on) {
  if (CAM.timer) {
    clearInterval(CAM.timer);
    CAM.timer = 0;
  }
  if (camAuto) camAuto.classList.toggle('on', !!on);
  if (on) CAM.timer = setInterval(camGrab, CAM.intervalMs);
}

on('btnCam', () => (CAM.on ? camStop() : camStart()));
on('camShoot', () => camGrab());
on('camAuto', () => camSetAuto(!CAM.timer));
if (video) video.addEventListener('click', () => camGrab());
