"""Is the production recogniser dropping a character on GREEN (new-energy) plates?

⚠️ DEPRECATED -- the plate-colour classifier in this file is BUGGY. Do not cite its
colour breakdowns. It selects "bright" pixels as `V > median(V)` and then measures
their hue; on a BLUE plate the brightest pixels are the WHITE characters, which are
near-unsaturated and whose hue is numeric noise. It therefore measured glyph pixels
instead of the plate face and labelled 86 blue plates "green" -- all 86 carry
7-character ground truth, which is itself proof they are not 8-character new-energy
plates. The "37.2% green hue" figure for the golden plate came from the same broken
selector and is void.

Use `tools/plate_face_colour.py` instead: plate paint = saturated mid-bright pixels
(`S>=90 & 45<=V<=250`), with the plate's own white glyphs as a white-balance control.
See `_evidence/A16-real-dataset-accuracy-20260918.md` section 4.5.

The overall accuracy figure (906/1000) printed by this script IS still valid -- it is
a direct filename comparison and does not depend on the colour classifier.

Context: the golden image `hlpr-test.jpg` carries a green new-energy plate, which
is legally 8 characters. The project's expected string is `苏ED5172` (7), while
LPRNet reads `苏ED51712` (8). One model is wrong. Rather than adjudicate on a
single image, measure the production recogniser's behaviour by plate colour on
the 1000-image third-party annotated set:

  * blue plates  -> 7 characters (province + letter + 5 alphanumerics)
  * green plates -> 8 characters (province + letter + D/F + 5)

If rpv3 is systematically one character short on green plates, the golden
expectation is wrong, not LPRNet. If rpv3 is fine on green plates, LPRNet's extra
`1` is a genuine hallucination.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CROPS = ROOT / "_dataset" / "real" / "crops"
OUT = ROOT / "_dataset" / "real" / "colour_analysis.json"

GREEN = np.array([80, 200, 80])       # BGR of a green plate face
BLUE = np.array([180, 120, 60])       # BGR of a blue plate face


def plate_colour(img):
    """Classify plate background: green / blue / other, by median hue of the bright face."""
    h, w = img.shape[:2]
    core = img[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    H_, S, V = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
    bright = V > np.percentile(V, 50)
    if bright.sum() < 10:
        return "other", 0.0, 0.0
    hue = np.median(H_[bright])
    green_frac = float(((H_[bright] >= 35) & (H_[bright] <= 95)).mean())
    blue_frac = float(((H_[bright] >= 100) & (H_[bright] <= 135)).mean())
    if green_frac > 0.45:
        return "green", green_frac, blue_frac
    if blue_frac > 0.45:
        return "blue", green_frac, blue_frac
    return "other", green_frac, blue_frac


def main():
    files = sorted(CROPS.glob("*.jpg"))
    print(f"images: {len(files)}")
    det = H.sess("y5fu_320x_sim.onnx")
    rec = H.sess("rpv3_mdict_160_r3.onnx")

    per_colour = defaultdict(lambda: {"n": 0, "exact": 0, "short": 0, "long": 0, "same": 0})
    gt_len_by_colour = defaultdict(Counter)
    misses = []
    for i, f in enumerate(files):
        gt = f.stem
        img = H.imread_u(f)
        if img is None:
            continue
        colour, gf, bf = plate_colour(img)
        gt_len_by_colour[colour][len(gt)] += 1

        h, w = img.shape[:2]
        data = H.encode_images(img, w * 1.0 / h, tuple(int(v) for v in rec.get_inputs()[0].shape[2:]))
        out = np.asarray(rec.run([rec.get_outputs()[0].name],
                                 {rec.get_inputs()[0].name: np.expand_dims(data, 0)})[0])[0]
        idx = np.argmax(out, axis=1)
        prob = np.max(out, axis=1)
        code, conf = H.ctc_decode(idx, prob)

        st = per_colour[colour]
        st["n"] += 1
        if code == gt:
            st["exact"] += 1
        else:
            if len(code) < len(gt):
                st["short"] += 1
            elif len(code) > len(gt):
                st["long"] += 1
            if len(code) == len(gt):
                st["same"] += 1
            if len(misses) < 40:
                misses.append({"file": f.name, "gt": gt, "pred": code, "colour": colour,
                               "len_gt": len(gt), "len_pred": len(code), "conf": round(conf, 3)})
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(files)}")

    print("\n=== ground-truth length by plate colour ===")
    for colour in ("blue", "green", "other"):
        c = gt_len_by_colour.get(colour)
        if c:
            print(f"  {colour:6s}: {dict(sorted(c.items()))}")

    print("\n=== rpv3 accuracy by plate colour ===")
    print(f"  {'colour':7s} {'n':>5s} {'exact':>6s} {'acc':>7s} {'too short':>10s} {'too long':>9s} {'same len wrong':>15s}")
    for colour in ("blue", "green", "other"):
        st = per_colour.get(colour)
        if not st or not st["n"]:
            continue
        print(f"  {colour:7s} {st['n']:5d} {st['exact']:6d} {st['exact'] / st['n']:6.1%} "
              f"{st['short']:10d} {st['long']:9d} {st['same']:15d}")

    tot = sum(s["n"] for s in per_colour.values())
    ex = sum(s["exact"] for s in per_colour.values())
    sh = sum(s["short"] for s in per_colour.values())
    lo = sum(s["long"] for s in per_colour.values())
    print(f"  {'ALL':7s} {tot:5d} {ex:6d} {ex / tot:6.1%} {sh:10d} {lo:9d}")

    print("\n=== first 40 misses ===")
    for m in misses:
        print(f"  {m['file']:16s} gt={m['gt']:9s}({m['len_gt']}) pred={m['pred']:9s}({m['len_pred']}) "
              f"{m['colour']:5s} conf={m['conf']}")

    OUT.write_text(json.dumps({
        "n": tot, "exact": ex,
        "by_colour": {k: dict(v) for k, v in per_colour.items()},
        "gt_len_by_colour": {k: dict(v) for k, v in gt_len_by_colour.items()},
        "misses": misses,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
