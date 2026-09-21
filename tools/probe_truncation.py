"""Probe the keypoint-truncation defect described in ADR-002 §2.2.

Claim under test: measuring the rectification quad from *float* keypoints (instead of
truncating them to int first) displaces the crop by up to one pixel and can flip the
decoded string on a small plate.

This script does not assume the claim. It measures, per sample, the crop shape and the
decoded string produced by both orderings, and reports which samples actually flip.

Run:  .venv\\Scripts\\python.exe tools\\probe_truncation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2

import hlpr_reference as R

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "samples"
EVID = ROOT / "_evidence"


def crop_float_marks(img, marks_float):
    """Variant V1 — the quad is *measured* from the untruncated keypoints, i.e. the
    port kept `row[5:13]` as float all the way through. `w`/`h` therefore come from
    float norms and can differ by one pixel from the reference."""
    pts = marks_float.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts, pts_std)
    dst = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dh, dw = dst.shape[0:2]
    if dw and dh * 1.0 / dw >= 1.5:
        dst = np.rot90(dst)
    return dst


def crop_int_size_float_transform(img, marks_int, marks_float):
    """Variant V2 — the output size is taken from the truncated keypoints (as the
    reference does) but the homography is solved from the float ones. Same output
    dimensions, sub-pixel displacement of the sampling grid."""
    pts_i = marks_int.astype(np.float32)
    pts_f = marks_float.astype(np.float32)
    w = int(max(np.linalg.norm(pts_i[0] - pts_i[1]), np.linalg.norm(pts_i[2] - pts_i[3])))
    h = int(max(np.linalg.norm(pts_i[0] - pts_i[3]), np.linalg.norm(pts_i[1] - pts_i[2])))
    pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts_f, pts_std)
    dst = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dh, dw = dst.shape[0:2]
    if dw and dh * 1.0 / dw >= 1.5:
        dst = np.rot90(dst)
    return dst


def decode_like_reference(rec, crop, layer):
    """Mirror run_pipeline's decode path exactly, including the double-layer split.

    Skipping this is an easy mistake: on hlpr-1.jpg the detector predicts layer=1, so
    the reference decodes the strip as two stacked rows and concatenates them. Decoding
    it as a single row yields a different (shorter) string and would make the defect
    look like it does not exist.
    """
    if layer == R.DOUBLE:
        h, _ = crop.shape[:2]
        line = int(h * 0.4)
        top_code, top_conf = R.recognize(rec, crop[:line, :, ])
        bot_code, bot_conf = R.recognize(rec, crop[line:, :])
        return top_code + bot_code, (top_conf + bot_conf) / 2
    return R.recognize(rec, crop)


def main() -> None:
    det = R.sess("y5fu_320x_sim.onnx")
    rec = R.sess("rpv3_mdict_160_r3.onnx")

    rows = []
    for p in sorted(SAMPLES.glob("*.jpg")):
        img = R.imread_u(p)
        if img is None:
            continue
        dets = R.detect(det, img)
        for k, row in enumerate(dets):
            raw = np.asarray(row[5:13], dtype=np.float64).reshape(4, 2)
            marks_int = raw.astype(int)
            layer = int(row[13])

            c_int = R.get_rotate_crop_image(img, marks_int)
            c_float = crop_float_marks(img, raw)
            c_mixed = crop_int_size_float_transform(img, marks_int, raw)

            code_int, conf_int = decode_like_reference(rec, c_int, layer)
            code_float, conf_float = decode_like_reference(rec, c_float, layer)
            code_mixed, conf_mixed = decode_like_reference(rec, c_mixed, layer)

            rows.append({
                "file": p.name,
                "plate_index": k,
                "layer": layer,
                "marks_float": raw.tolist(),
                "marks_int": marks_int.tolist(),
                "crop_shape_int": list(c_int.shape[:2]),
                "crop_shape_float": list(c_float.shape[:2]),
                "crop_shape_mixed": list(c_mixed.shape[:2]),
                "code_int": code_int,
                "code_float": code_float,
                "code_mixed": code_mixed,
                "conf_int": round(conf_int, 4),
                "conf_float": round(conf_float, 4),
                "conf_mixed": round(conf_mixed, 4),
                "shape_flip": list(c_int.shape[:2]) != list(c_float.shape[:2]),
                "code_flip": code_int != code_float,
                "code_flip_mixed": code_int != code_mixed,
            })
            print(f"{p.name:28s} p{k} layer={layer}  ref={code_int:10s} {str(list(c_int.shape[:2])):10s}"
                  f"  V1float={code_float:10s} {str(list(c_float.shape[:2])):10s}"
                  f"  V2mixed={code_mixed:10s} {str(list(c_mixed.shape[:2])):10s}")

    flips = [r for r in rows if r["code_flip"]]
    print(f"\n{len(rows)} plate(s) probed, {len(flips)} with a flipped decoded string")
    for r in flips:
        print(f"  FLIP {r['file']} p{r['plate_index']}: {r['code_float']} -> {r['code_int']}")

    (EVID / "probe_truncation.json").write_text(
        json.dumps({"rows": rows, "flips": flips}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", EVID / "probe_truncation.json")


if __name__ == "__main__":
    main()
