"""Render the reference-pipeline result so a human can eyeball it.

Produces _evidence/viz_<name>.png : left = source with box + 4 corner points,
right = the rectified crop actually fed to the recogniser (upscaled).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import hlpr_reference as R

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "samples"
EVID = ROOT / "_evidence"
EVID.mkdir(exist_ok=True)


def panel(src, crop, title):
    h = 420
    sw = int(src.shape[1] * h / src.shape[0])
    src_r = cv2.resize(src, (sw, h))
    if crop is None:
        return src_r
    ch, cw = crop.shape[:2]
    scale = min(560 / max(cw, 1), 380 / max(ch, 1))
    crop_r = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, sw + 20 + crop_r.shape[1], 3), 245, dtype=np.uint8)
    canvas[:src_r.shape[0], :sw] = src_r
    y0 = (h - crop_r.shape[0]) // 2
    canvas[y0:y0 + crop_r.shape[0], sw + 20:sw + 20 + crop_r.shape[1]] = crop_r
    return canvas


def main():
    det = R.sess("y5fu_320x_sim.onnx")
    rec = R.sess("rpv3_mdict_160_r3.onnx")
    cls = R.sess("litemodel_cls_96x_r1.onnx")

    summary = []
    for p in sorted(SAMPLES.glob("hlpr-*.jpg")):
        img = R.imread_u(p)
        dets = R.detect(det, img)
        vis = img.copy()
        crop_show = None
        for row in dets:
            rect = row[:4].astype(int)
            marks = row[5:13].reshape(4, 2).astype(int)
            layer = int(row[13])
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
            for k, (px, py) in enumerate(marks):
                cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(vis, str(k), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            crop = R.get_rotate_crop_image(img, marks)
            if layer == R.DOUBLE:
                h = crop.shape[0]
                line = int(h * 0.4)
                t, tc = R.recognize(rec, crop[:line, :, ])
                b, bc = R.recognize(rec, crop[line:, :])
                code, conf = t + b, (tc + bc) / 2
            else:
                code, conf = R.recognize(rec, crop)
            colors = R.classify(cls, crop)
            crop_show = crop
            summary.append({"file": p.name, "src": list(img.shape[:2]), "code": code,
                            "rec_conf": round(conf, 4), "det_score": round(float(row[4]), 4),
                            "layer": layer, "cls": [round(float(v), 4) for v in colors],
                            "crop": list(crop.shape[:2])})
            print(f"{p.name} -> {code} conf={conf:.3f} layer={layer} cls={np.round(colors,3)}")
        out = panel(vis, crop_show, p.stem)
        dst = EVID / f"viz_{p.stem}.png"
        cv2.imencode(".png", out)[1].tofile(str(dst))
        print("  wrote", dst.name)

    (EVID / "viz_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
