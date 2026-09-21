"""How many characters does the golden plate carry? Measure the glyph PITCH.

Chinese plate geometry (GA 36): glyph 45 mm wide, 12 mm gap -> pitch 57 mm.
So if we can measure one glyph's width W and the total ink span S:

    S = n*W + (n-1)*gap      and      gap = W*12/45 = 0.2667*W
    => n = (S/W + 0.2667) / 1.2667

Two independent estimates are computed here:
  1. from the width of the leftmost glyph (the province character)
  2. from the dominant period of the ink column profile (autocorrelation)

The frame/shadow is excluded by keeping only the glyph band, which is found as
the rows where the ink density is high and locally flat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
TMP = Path(r"C:\Users\26671\AppData\Local\Temp")


def warp_plate(img):
    det = H.sess("y5fu_320x_sim.onnx")
    rows = H.detect(det, img)
    pts = rows[0][5:13].reshape(4, 2).astype(np.float32)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)
    d = lambda a, b: float(np.hypot(*(a - b)))                      # noqa: E731
    Wm = (d(pts[0], pts[1]) + d(pts[3], pts[2])) / 2
    Hm = (d(pts[0], pts[3]) + d(pts[1], pts[2])) / 2
    S = 10
    W, Hh = int(Wm * S), int(Hm * S)
    M = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float32))
    return cv2.warpPerspective(img, M, (W, Hh), flags=cv2.INTER_CUBIC)


def glyph_band(warp):
    """Return the ink mask restricted to the glyph band (frame/shadow removed)."""
    h, w = warp.shape[:2]
    V = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    # smooth horizontally to suppress the wavy glare, keep glyph structure
    Vs = cv2.GaussianBlur(V, (0, 0), 6)
    ink = (V < Vs - 22).astype(np.uint8)          # locally dark = glyph stroke
    rowd = ink.mean(axis=1)
    # the glyph band is where row density is high; take the largest run above half-max
    thr = rowd.max() * 0.45
    best = (0, 0, -1)
    i = 0
    while i < h:
        if rowd[i] > thr:
            j = i
            while j < h and rowd[j] > thr:
                j += 1
            if j - i > best[2]:
                best = (i, j, j - i)
            i = j
        else:
            i += 1
    y0, y1 = best[0], best[1]
    band = ink[y0:y1]
    return band, ink, rowd, y0, y1


def main():
    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    warp = warp_plate(img)
    band, ink, rowd, y0, y1 = glyph_band(warp)
    print(f"warp {warp.shape[1]}x{warp.shape[0]}   glyph band rows {y0}..{y1} (h={y1 - y0})")
    ok, buf = cv2.imencode(".png", (band * 255).astype(np.uint8))
    buf.tofile(str(TMP / "golden_band.png"))

    proj = band.sum(axis=0).astype(float)
    # trim empty margins
    nz = np.nonzero(proj > 0)[0]
    span = float(nz[-1] - nz[0])
    print(f"ink span = {span:.0f} px over the {band.shape[1]} px plate width "
          f"({span / band.shape[1]:.1%} of the plate)")

    # --- estimate 1: leftmost glyph (province char) width -------------------
    # walk from the left edge; the first glyph is a run of ink, then a real gap
    gapmin = span * 0.012
    x = nz[0]
    run_start, runs = x, []
    inrun = True
    cur = x
    for xx in range(nz[0], nz[-1] + 1):
        if proj[xx] > 0:
            if not inrun:
                runs.append((run_start, xx, xx - run_start))
                inrun = True
            cur = xx
        else:
            if inrun and xx - cur > gapmin:
                run_start = xx
                inrun = False
    runs = [r for r in runs if r[2] > span * 0.01]
    print(f"\nruns after splitting on gaps > {gapmin:.0f} px: {len(runs)}")
    for a, b, wd in runs:
        print(f"   x={a:5d}..{b:5d}  width={wd:5d}")

    Wg = float(runs[0][2]) if runs else 0.0
    if Wg:
        n1 = (span / Wg + 0.2667) / 1.2667
        print(f"\n[estimate 1] leftmost glyph width W = {Wg:.0f} px")
        print(f"             span/W = {span / Wg:.3f}   ->  n = {n1:.2f}")

    # --- estimate 2: dominant pitch by autocorrelation ---------------------
    p = proj[nz[0]:nz[-1] + 1]
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac /= ac[0]
    lo = int(span / 11)
    hi = int(span / 5)
    peak = lo + int(np.argmax(ac[lo:hi]))
    print(f"\n[estimate 2] autocorrelation peak at lag {peak} px  (ac={ac[peak]:.3f})")
    n2 = span / peak + 0.2667 * (span / peak) / 1.2667
    n2 = (span / peak + 0.2667) / 1.2667
    print(f"             span/pitch = {span / peak:.3f}   ->  n = {n2:.2f}")

    print("\n--- theoretical reference ---")
    for k in (7, 8):
        print(f"   {k} chars: span/glyph_width = {k + (k - 1) * 0.2667:.2f}"
              f"   and span/pitch = {k - 0.2667:.2f}")
    print("\n  measured span/W and span/pitch above should be compared with these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
