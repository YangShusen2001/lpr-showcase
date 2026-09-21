"""Audit the plate-colour classifier: are the 86 'green' 7-character plates real?

`plate_colour_analysis.py` classified 86 of the 1000 images as GREEN, yet every one
of them carries a 7-character ground-truth label. If a green plate is legally an
8-character new-energy plate, those 86 cannot be green -- so either the classifier
is unreliable, or the legal argument used to reinterpret the golden image is wrong.

Both matter: the golden-image conclusion currently rests partly on "the plate is
green, therefore it is an 8-character new-energy plate". This script tests that
premise instead of assuming it.

It does three things:
  1. lists the 86 green-classified files and samples them into a contact sheet;
  2. applies the SAME classifier to the golden plate's face, so the comparison is
     apples to apples (previous golden measurement used a different threshold);
  3. reports what a stricter, saturation-aware classifier says about each group.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CROPS = ROOT / "_dataset" / "real" / "crops"
TMP = Path(r"C:\Users\26671\AppData\Local\Temp")
OUT = ROOT / "_dataset" / "real" / "colour_audit.json"


def classify(img, green_frac_thresh=0.45, blue_frac_thresh=0.45, sat_min=0):
    """Same rule as plate_colour_analysis.py, with optional saturation floor."""
    h, w = img.shape[:2]
    core = img[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    H_, S, V = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
    bright = V > np.percentile(V, 50)
    if sat_min:
        bright = bright & (S >= sat_min)
    if bright.sum() < 10:
        return "other", 0.0, 0.0, 0.0
    green_frac = float(((H_[bright] >= 35) & (H_[bright] <= 95)).mean())
    blue_frac = float(((H_[bright] >= 100) & (H_[bright] <= 135)).mean())
    sat_med = float(np.median(S[bright]))
    if green_frac > green_frac_thresh:
        return "green", green_frac, blue_frac, sat_med
    if blue_frac > blue_frac_thresh:
        return "blue", green_frac, blue_frac, sat_med
    return "other", green_frac, blue_frac, sat_med


def main():
    data = json.loads((ROOT / "_dataset" / "real" / "colour_analysis.json").read_text("utf-8"))
    greens = [m["file"] for m in data["misses"]] if "file" in (data["misses"][0] if data["misses"] else {}) else None

    # re-classify everything ourselves so we can vary the rule
    files = sorted(CROPS.glob("*.jpg"))
    rows = []
    for f in files:
        img = H.imread_u(f)
        if img is None:
            continue
        cls, gf, bf, sm = classify(img)
        cls2, gf2, bf2, sm2 = classify(img, sat_min=60)
        rows.append({"file": f.name, "gt": f.stem, "gt_len": len(f.stem),
                     "cls": cls, "green_frac": round(gf, 3), "blue_frac": round(bf, 3),
                     "sat": round(sm, 1),
                     "cls_sat60": cls2, "green_frac_sat60": round(gf2, 3), "sat60": round(sm2, 1)})

    print(f"images: {len(rows)}")
    print("\n1) length x colour (default rule)")
    tab = Counter((r["gt_len"], r["cls"]) for r in rows)
    for (L, c), n in sorted(tab.items()):
        print(f"   gt_len={L}  colour={c:6s} n={n}")

    print("\n2) length x colour (rule with saturation floor 60 -- excludes washed-out/greyish plates)")
    tab2 = Counter((r["gt_len"], r["cls_sat60"]) for r in rows)
    for (L, c), n in sorted(tab2.items()):
        print(f"   gt_len={L}  colour={c:6s} n={n}")

    green_rows = [r for r in rows if r["cls"] == "green"]
    print(f"\n3) the {len(green_rows)} images the default rule calls GREEN, all gt_len=7")
    print(f"   their green_frac: min={min(r['green_frac'] for r in green_rows):.2f} "
          f"median={np.median([r['green_frac'] for r in green_rows]):.2f} "
          f"max={max(r['green_frac'] for r in green_rows):.2f}")
    print(f"   their saturation: min={min(r['sat'] for r in green_rows):.1f} "
          f"median={np.median([r['sat'] for r in green_rows]):.1f} "
          f"max={max(r['sat'] for r in green_rows):.1f}")
    reclassified = Counter(r["cls_sat60"] for r in green_rows)
    print(f"   after saturation floor 60 they become: {dict(reclassified)}")
    print("   sample:", [r["file"] for r in green_rows[:10]])

    # saturation of the blue group for comparison
    blue_rows = [r for r in rows if r["cls"] == "blue"]
    print(f"\n   (blue group for reference: n={len(blue_rows)} "
          f"sat median={np.median([r['sat'] for r in blue_rows]):.1f}, "
          f"green_frac median={np.median([r['green_frac'] for r in blue_rows]):.2f})")

    # contact sheet of the green ones
    sheet_rows = []
    picks = green_rows[:24]
    tiles = []
    for r in picks:
        im = H.imread_u(CROPS / r["file"])
        im = cv2.resize(im, (200, int(200 * im.shape[0] / im.shape[1])))
        cv2.putText(im, r["gt"], (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(im, r["cls"], (3, im.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        tiles.append(im)
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)) for t in tiles]
        per = 6
        rowsimg = []
        for i in range(0, len(tiles), per):
            chunk = tiles[i:i + per]
            while len(chunk) < per:
                chunk.append(np.full_like(chunk[0], 255))
            rowsimg.append(np.hstack(chunk))
        sheet = np.vstack(rowsimg)
        p = TMP / "green_audit.png"
        cv2.imwrite(str(p), sheet)
        print(f"\n   contact sheet -> {p}  ({sheet.shape[1]}x{sheet.shape[0]})")

    # 4) golden plate through the SAME classifier
    print("\n4) golden plate through the same classifier")
    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    det = H.sess("y5fu_320x_sim.onnx")
    rows_det = H.detect(det, img)
    x1, y1, x2, y2 = rows_det[0][:4].astype(int)
    plate = img[y1:y2, x1:x2]
    p = TMP / "golden_plate_face.png"
    cv2.imwrite(str(p), cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    for name, rule in (("default", {}), ("sat floor 60", {"sat_min": 60})):
        cls, gf, bf, sm = classify(plate, **rule)
        print(f"   golden plate [{name:12s}] -> {cls:6s} green_frac={gf:.3f} blue_frac={bf:.3f} sat_med={sm:.1f}")
    print(f"   plate face crop -> {p}")

    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
