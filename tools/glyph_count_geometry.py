"""Calibrated glyph-count geometry: how many characters does the golden plate hold?

Chinese plate geometry (GB 7258 / GA 36):
  * glyph width 45 mm, gap 12 mm, glyph height 90 mm
  * blue plate  : 440 x 140 mm, 7 characters  -> ink span 7*45+6*12 = 387 mm
  * green (new-energy): 480 x 140 mm, 8 characters -> ink span 8*45+7*12 = 444 mm

So the ratio  ink_span / glyph_height  is 387/90 = 4.30 for a 7-character plate
and 444/90 = 4.93 for an 8-character plate -- a 15% separation.

This script does not assume that model. It MEASURES the ratio on the 1000-image
third-party set, where the filename gives the true character count, and reports
how well the ratio separates 7 from 8. That calibration is then applied to the
golden plate, whose true count is disputed (rpv3 says 7, LPRNet says 8).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CROPS = ROOT / "_dataset" / "real" / "crops"
OUT = ROOT / "_dataset" / "real" / "glyph_geometry.json"


def glyph_metrics(img):
    """Return (ink_span, glyph_height, n_tall_blobs, span_over_height).

    Works on a plate crop: segment the plate face, binarise the glyphs, then
    measure the horizontal span of glyph ink and the typical glyph height.
    """
    h, w = img.shape[:2]
    if h < 12 or w < 12:
        return None
    up = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    hh, ww = up.shape[:2]
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    # plate face = largest saturated blob (green or blue)
    sat = cv2.inRange(hsv, (0, 45, 40), (179, 255, 255))
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, 4]))
    face = cv2.erode((lab == big).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    if (face > 0).sum() < 0.05 * hh * ww:
        return None

    g = cv2.GaussianBlur(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ((th > 0) & (face > 0)).astype(np.uint8)

    # glyph height: median height of tall connected blobs
    nb, lb, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    heights = [st[i][3] for i in range(1, nb) if st[i][4] > 0.002 * hh * ww and st[i][3] > 0.35 * hh]
    if not heights:
        return None
    Hg = float(np.median(heights))

    # ink span: horizontal extent of the tall blobs only (ignores frame/shadow)
    xs = []
    for i in range(1, nb):
        x, y, bw, bh, area = st[i]
        if area > 0.002 * hh * ww and bh > 0.35 * hh:
            xs.append((x, x + bw))
    if len(xs) < 2:
        return None
    span = float(max(e for _, e in xs) - min(s for s, _ in xs))
    return span, Hg, len(xs), span / Hg


def main():
    files = sorted(CROPS.glob("*.jpg"))
    by_len = {}
    rows = []
    for f in files:
        gt = f.stem
        m = glyph_metrics(H.imread_u(f))
        if not m:
            continue
        span, Hg, nblob, ratio = m
        rows.append({"file": f.name, "gt": gt, "n": len(gt), "ratio": round(ratio, 3),
                     "nblob": nblob, "span": round(span, 1), "Hg": round(Hg, 1)})
        by_len.setdefault(len(gt), []).append(ratio)

    print("=== calibration on the 1000-image third-party set ===")
    print(f"  {'chars':>5s} {'n':>5s} {'median ratio':>13s} {'p10':>6s} {'p90':>6s} {'theoretical':>12s}")
    for k in sorted(by_len):
        a = np.array(by_len[k])
        theo = (k * 45 + (k - 1) * 12) / 90.0
        print(f"  {k:5d} {len(a):5d} {np.median(a):13.3f} {np.percentile(a, 10):6.3f} "
              f"{np.percentile(a, 90):6.3f} {theo:12.2f}")

    # a simple nearest-centroid classifier using the measured medians
    cents = {k: float(np.median(np.array(v))) for k, v in by_len.items()}
    correct = sum(1 for r in rows if min(cents, key=lambda k: abs(cents[k] - r["ratio"])) == r["n"])
    print(f"\n  nearest-centroid char-count accuracy on the calibration set: "
          f"{correct}/{len(rows)} = {correct / max(1, len(rows)):.1%}")

    print("\n=== applying the calibration to the golden plate ===")
    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    det = H.sess("y5fu_320x_sim.onnx")
    rows_det = H.detect(det, img)
    pts = rows_det[0][5:13].reshape(4, 2).astype(np.float32)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)
    W, Hh = 880, 280
    M = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float32))
    warp = cv2.warpPerspective(img, M, (W, Hh), flags=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", warp)
    buf.tofile(r"C:\Users\26671\AppData\Local\Temp\golden_warp_cal.png")

    m = glyph_metrics(warp)
    if not m:
        print("  could not measure the golden plate")
        return 1
    span, Hg, nblob, ratio = m
    print(f"  ink span={span:.1f}  glyph height={Hg:.1f}  tall blobs={nblob}  ratio={ratio:.3f}")
    guess = min(cents, key=lambda k: abs(cents[k] - ratio))
    print(f"  measured ratio {ratio:.3f} -> nearest calibration centroid = {guess} characters")
    for k in sorted(cents):
        print(f"     centroid {k} chars = {cents[k]:.3f}   |diff| = {abs(cents[k] - ratio):.3f}")
    print(f"\n  => golden plate carries {guess} characters")

    OUT.write_text(json.dumps({"calibration": {str(k): v for k, v in by_len.items()},
                               "centroids": {str(k): v for k, v in cents.items()},
                               "golden": {"ratio": ratio, "guess": guess},
                               "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
