"""Dump the same intermediates as tools/_verify.html so the browser port can be
diffed against the Python reference field by field.

Writes _evidence/det_dump.json with, for every full image:
  * letterbox params (r / left / top)
  * the raw detector row carrying the strongest objectness (15 floats)
  * the full pipeline result rows (rect / marks / layer / code / conf / crop / cls)
  * a crop checksum so rectification differences are visible directly

Run:  python tools/dump_det.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "_evidence"
EVID.mkdir(exist_ok=True)

IMAGES = ["hlpr-1.jpg", "hlpr-test.jpg", "scene-2.jpg", "plate-yellow-320x48.jpg"]


def top_raw_row(raw: np.ndarray) -> list[float]:
    flat = raw.reshape(-1, raw.shape[-1])
    i = int(np.argmax(flat[:, 4]))
    return [round(float(v), 6) for v in flat[i]]


def tensor_stats(x: np.ndarray) -> dict:
    flat = x.ravel()
    return {
        "sum": round(float(flat.sum()), 6),
        "min": round(float(flat.min()), 8),
        "max": round(float(flat.max()), 8),
        "head": [round(float(v), 8) for v in flat[:24]],
        "mid": [round(float(v), 8) for v in flat[153600:153624]],
    }


def main() -> None:
    det = H.sess("y5fu_320x_sim.onnx")
    rec = H.sess("rpv3_mdict_160_r3.onnx")
    cls = H.sess("litemodel_cls_96x_r1.onnx")

    out = {"ort_version": __import__("onnxruntime").__version__, "images": []}
    for name in IMAGES:
        img = H.imread_u(H.SAMPLES / name)
        size = tuple(int(v) for v in det.get_inputs()[0].shape[2:])
        x, r, left, top = H.detect_pre_precessing(img, size)
        raw = np.asarray(det.run([det.get_outputs()[0].name], {det.get_inputs()[0].name: x})[0])
        flat = raw.reshape(-1, raw.shape[-1])
        i = int(np.argmax(flat[:, 4]))

        plates = H.run_pipeline(img, det, rec, cls, full=True)
        crops = []
        for row in H.detect(det, img):
            marks = row[5:13].reshape(4, 2).astype(int)
            crop = H.get_rotate_crop_image(img, marks)
            crops.append({
                "shape": list(crop.shape[:2]),
                "sum": int(crop.astype(np.int64).sum()),
            })

        out["images"].append({
            "name": name,
            "size": list(img.shape[:2]),
            "detect": {
                "size": list(size),
                "r": r, "left": left, "top": top,
                "rows": int(flat.shape[0]),
                "input": tensor_stats(x),
                "argmaxObjIndex": i,
                "topRawRow": [round(float(v), 6) for v in flat[i]],
            },
            "crops": crops,
            "plates": plates,
        })
        print(f"{name:16s} r={r:.6f} left={left} top={top} argmaxObj={i} top={flat[i][:5]}")

    p = EVID / "det_dump.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", p)


if __name__ == "__main__":
    main()
