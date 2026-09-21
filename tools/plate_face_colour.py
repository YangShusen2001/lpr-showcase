"""Correct plate-face colour measurement (replaces the buggy classifier).

Bug in `plate_colour_analysis.py`: it selected "bright" pixels as `V > median(V)`
and then measured their hue. On a BLUE plate the brightest pixels are the WHITE
characters, which are near-unsaturated -- their hue is numeric noise. So the hue
statistic described glyph pixels, not the plate face, and 86 blue plates were
labelled "green". The 86 all carry 7-character ground truth, which is itself proof
they cannot be 8-character new-energy plates.

Correct approach: classify by the DOMINANT SATURATED colour of the plate face.
  1. drop near-white / near-black pixels (S and V floors) -- removes glyphs,
     borders and shadows, leaving plate-face paint;
  2. take the modal hue band of what remains;
  3. green  (new-energy)  -> hue 35..95
     blue   (ordinary)    -> hue 100..135
     yellow (large/taxi)  -> hue 15..34
     other                -> no band holds a majority

Also measures the golden plate with the SAME code, so the golden reading is
directly comparable instead of relying on a differently-thresholded earlier run.
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
OUT = ROOT / "_dataset" / "real" / "colour_audit2.json"

BANDS = [("green", 35, 95), ("blue", 100, 135), ("yellow", 15, 34)]


def face_colour(img, s_min=90, v_min=45, v_max=250):
    """Dominant saturated colour of the plate face."""
    h, w = img.shape[:2]
    core = img[int(h * 0.12):int(h * 0.88), int(w * 0.08):int(w * 0.92)]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # plate paint = saturated, mid-bright. Kills white glyphs (S low) and
    # black glyphs / shadows (V low) and specular highlights (V very high).
    mask = (S >= s_min) & (V >= v_min) & (V <= v_max)
    n = int(mask.sum())
    if n < 40:
        return "other", {}, 0.0, n
    hues = Hh[mask]
    frac = {}
    for name, lo, hi in BANDS:
        frac[name] = float(((hues >= lo) & (hues <= hi)).mean())
    best = max(frac, key=frac.get)
    if frac[best] < 0.55:
        best = "other"
    return best, {k: round(v, 3) for k, v in frac.items()}, float(np.median(S[mask])), n


def main():
    files = sorted(CROPS.glob("*.jpg"))
    rows = []
    for f in files:
        img = H.imread_u(f)
        if img is None:
            continue
        cls, frac, sat, n = face_colour(img)
        rows.append({"file": f.name, "gt": f.stem, "gt_len": len(f.stem),
                     "cls": cls, "frac": frac, "sat": round(sat, 1), "npx": n})

    print(f"images: {len(rows)}")
    print("\n1) ground-truth length x plate face colour (corrected measurement)")
    tab = Counter((r["gt_len"], r["cls"]) for r in rows)
    for (L, c), n in sorted(tab.items()):
        print(f"   gt_len={L}  colour={c:6s} n={n}")

    print("\n2) distribution of green fraction, and how many have green as the top band")
    gf = np.array([r["frac"].get("green", 0.0) for r in rows])
    print(f"   green_frac: p50={np.median(gf):.3f} p90={np.percentile(gf,90):.3f} max={gf.max():.3f}")
    print(f"   images with green_frac > 0.55 : {(gf > 0.55).sum()}")
    print(f"   images with green_frac > 0.80 : {(gf > 0.80).sum()}")

    top_green = sorted(rows, key=lambda r: -r["frac"].get("green", 0))[:12]
    print("\n   top-12 greenest by corrected measure:")
    for r in top_green:
        print(f"     {r['file']:16s} gt={r['gt']:9s} cls={r['cls']:6s} "
              f"green={r['frac'].get('green',0):.2f} blue={r['frac'].get('blue',0):.2f} sat={r['sat']}")

    print("\n3) the 86 files the OLD rule called green -- corrected verdict")
    old = json.loads((ROOT / "_dataset" / "real" / "colour_analysis.json").read_text("utf-8"))
    # recompute old-rule membership
    def old_rule(img):
        h, w = img.shape[:2]
        core = img[int(h*0.15):int(h*0.85), int(w*0.10):int(w*0.90)]
        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        Hh, S, V = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
        bright = V > np.percentile(V, 50)
        return float(((Hh[bright] >= 35) & (Hh[bright] <= 95)).mean()) > 0.45
    by_name = {r["file"]: r for r in rows}
    old_green = []
    for f in files:
        img = H.imread_u(f)
        if img is not None and old_rule(img):
            old_green.append(f.name)
    print(f"   old rule called {len(old_green)} green")
    cc = Counter(by_name[n]["cls"] for n in old_green if n in by_name)
    print(f"   corrected verdict on those: {dict(cc)}")
    gf_old = np.array([by_name[n]["frac"].get("green", 0.0) for n in old_green if n in by_name])
    print(f"   their corrected green_frac: median={np.median(gf_old):.3f} max={gf_old.max():.3f}")

    print("\n4) golden plate, same corrected code")
    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    det = H.sess("y5fu_320x_sim.onnx")
    rows_det = H.detect(det, img)
    x1, y1, x2, y2 = rows_det[0][:4].astype(int)
    plate = img[y1:y2, x1:x2]
    cls, frac, sat, n = face_colour(plate)
    print(f"   detector box {x2-x1}x{y2-y1} -> {cls}  frac={frac}  sat_med={sat}  npx={n}")
    p = TMP / "golden_face_8x.png"
    cv2.imwrite(str(p), cv2.resize(plate, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
    print(f"   8x face crop -> {p}")

    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
