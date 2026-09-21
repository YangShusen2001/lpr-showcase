"""Settle the golden image: how many characters does the plate actually carry?

The project's red line is the expected string `苏ED5172` (7 chars). But the plate is
GREEN, and a green Chinese plate is a new-energy plate, which is legally 8
characters (province + letter + D/F + 5). LPRNet reads 8 (`苏ED51712`); the
production recogniser reads 7. One of them is dropping or inventing a character,
so count the glyph slots geometrically instead of trusting either model.

Two pitfalls this script is written around:
  * A whole-scene green mask locks onto foliage, not the plate -- so restrict the
    search to the detector's own plate box first.
  * cv2.imwrite fails silently on non-ASCII Windows paths -- use imencode+tofile.
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


def save_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))
    return path


def rectify(roi):
    """Rectify the green plate inside a plate-sized ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * roi.shape[0] * roi.shape[1]:
        return None, mask
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    s, d = box.sum(1), np.diff(box, axis=1).ravel()
    box = np.array([box[np.argmin(s)], box[np.argmin(d)],
                    box[np.argmax(s)], box[np.argmax(d)]], np.float32)
    W, Hh = 880, 280
    M = cv2.getPerspectiveTransform(box, np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float32))
    return cv2.warpPerspective(roi, M, (W, Hh)), mask


def count_slots(rect):
    """Count glyph groups by dark-ink column projection inside the plate border."""
    h, w = rect.shape[:2]
    inner = rect[int(h * 0.12):int(h * 0.88), int(w * 0.04):int(w * 0.96)]
    g = cv2.GaussianBlur(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = (th > 0).sum(axis=0)
    thr = max(3, int(proj.max() * 0.10))
    groups, cur = [], None
    for x, v in enumerate(proj):
        if v > thr and cur is None:
            cur = x
        elif v <= thr and cur is not None:
            groups.append((cur, x, x - cur))
            cur = None
    if cur is not None:
        groups.append((cur, len(proj), len(proj) - cur))
    # merge runs separated by < 3px (antialiasing splits)
    merged = []
    for gp in groups:
        if merged and gp[0] - merged[-1][1] < 4:
            a = merged[-1][0]
            merged[-1] = (a, gp[1], gp[1] - a)
        else:
            merged.append(gp)
    return merged, proj, th, inner


def main():
    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    det = H.sess("y5fu_320x_sim.onnx")
    rows = H.detect(det, img)
    x1, y1, x2, y2 = rows[0][:4].astype(int)
    print(f"detector box: x={x1}..{x2} y={y1}..{y2}  ({x2 - x1}x{y2 - y1})")
    m = 40
    roi = img[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
    print("roi:", roi.shape)
    save_png(TMP / "golden_roi.png", roi)

    rect, mask = rectify(roi)
    if rect is None:
        print("green plate not found inside the detector box")
        return 1
    save_png(TMP / "golden_rect3.png", rect)
    print("green mask pixels:", int((mask > 0).sum()), " rectified:", rect.shape)

    groups, proj, th, inner = count_slots(rect)
    print(f"\nglyph groups from column projection: {len(groups)}")
    for a, b, w in groups:
        print(f"   x={a:3d}..{b:3d}  width={w:3d}")
    print("\ncolumn profile (every 10px):")
    print(" ".join(f"{proj[i]:3d}" for i in range(0, len(proj), 10)))
    save_png(TMP / "golden_thresh.png", 255 - th)

    # how many tall glyph blobs (characters) vs short ones (the separator dot)?
    n, lab, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    Hh = inner.shape[0]
    chars, dots = [], []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 30:
            continue
        (chars if bh > Hh * 0.45 else dots).append((x, y, bw, bh, area))
    print(f"\ntall blobs (bh > {Hh * 0.45:.0f}px) = {len(chars)}  short blobs (dot/noise) = {len(dots)}")
    for x, y, bw, bh, area in sorted(chars):
        print(f"   CHAR x={x:3d} y={y:3d} w={bw:3d} h={bh:3d} area={area:5d}")
    for x, y, bw, bh, area in sorted(dots):
        print(f"   dot  x={x:3d} y={y:3d} w={bw:3d} h={bh:3d} area={area:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
