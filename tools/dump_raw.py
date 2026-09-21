"""Dump raw end2end outputs (all rows, all confidences) for every sample x every model."""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"
SAMPLES = ROOT / "assets" / "samples"


def letterbox(img, new_shape, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


for m in sorted(MODELS.glob("*.onnx")):
    size = int(m.stem.split("-")[3])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(m), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    print(f"\n=== {m.name} (input {size}) ===")
    for ip in sorted(SAMPLES.glob("*")):
        try:
            img = np.array(Image.open(ip).convert("RGB"))
        except Exception as e:
            print(f"  {ip.name}: OPEN FAIL {e}")
            continue
        lb, r, dw, dh = letterbox(img, size)
        x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out = sess.run(None, {iname: x})[0]
        rows = np.asarray(out).reshape(-1, np.asarray(out).shape[-1])
        print(f"  {ip.name} {img.shape[1]}x{img.shape[0]} -> {len(rows)} rows")
        for row in rows:
            print("      batch=%d box=(%.1f,%.1f,%.1f,%.1f) cls=%d score=%.4f"
                  % (row[0], row[1], row[2], row[3], row[4], int(row[5]), row[6]))
