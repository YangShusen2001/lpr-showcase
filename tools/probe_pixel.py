"""Zoom into the exact pixels where the port disagrees with cv2.resize.

Prints the four source values, their fixed-point weights, and the result produced by
each candidate layout, so the rounding rule can be read off directly.

Run:  python tools/probe_pixel.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "assets" / "samples" / "scene-2.jpg"
SRC_W, SRC_H = 1140, 456
DST_W, DST_H = 320, 128
COEF = 1 << 11


def weights(n_src: int, n_dst: int):
    i = np.arange(n_dst)
    f = (i + 0.5) * (n_src / n_dst) - 0.5
    s = np.floor(f).astype(np.int64)
    w = f - s
    s = np.clip(s, 0, n_src - 2)
    return s, w


def main() -> None:
    img = H.imread_u(SAMPLE)
    ref = cv2.resize(img, (DST_W, DST_H))[:, :, 0].astype(np.int64)
    src = img[:, :, 0].astype(np.int64)

    sx, wx = weights(SRC_W, DST_W)
    sy, wy = weights(SRC_H, DST_H)
    cx0 = np.round((1 - wx) * COEF).astype(np.int64)
    cx1 = np.round(wx * COEF).astype(np.int64)
    cy0 = np.round((1 - wy) * COEF).astype(np.int64)
    cy1 = np.round(wy * COEF).astype(np.int64)

    tmp = src[:, sx] * cx0 + src[:, sx + 1] * cx1                 # unshifted int
    got_a = ((tmp[sy, :] * cy0[:, None] + tmp[sy + 1, :] * cy1[:, None] + (1 << 21)) >> 22)

    th = ((tmp + (1 << 10)) >> 11)                                # H rounded to int
    got_b = ((th[sy, :] * cy0[:, None] + th[sy + 1, :] * cy1[:, None] + (1 << 10)) >> 11)

    th2 = np.round(tmp / COEF).astype(np.int64)                   # H rounded to 8-bit-ish
    got_c = np.round((th2[sy, :] * cy0[:, None] + th2[sy + 1, :] * cy1[:, None]) / (COEF * COEF)).astype(np.int64)

    for name, got in (("A unshiftedH+V22", got_a), ("B H11r+V11r", got_b), ("C Hround8+Vfloat", got_c)):
        d = got - ref
        print(f"{name:18s} mismatch={int(np.count_nonzero(d)):6d}/{d.size} "
              f"mean={float(d.mean()):+.4f}  #(+1)={int((d == 1).sum())}  #(-1)={int((d == -1).sum())}")

    d = got_a - ref
    idx = np.argwhere(d != 0)[:10]
    print("\nper-pixel detail (candidate A):")
    print("  y  x |   s00   s01   s10   s11 |      w00       w01       w10       w11 |  exact   ref  A  d")
    for y, x in idx:
        r0, r1 = sy[y], sy[y] + 1
        c0, c1 = sx[x], sx[x] + 1
        s00, s01 = src[r0, c0], src[r0, c1]
        s10, s11 = src[r1, c0], src[r1, c1]
        w00 = cy0[y] * cx0[x] / (COEF * COEF)
        w01 = cy0[y] * cx1[x] / (COEF * COEF)
        w10 = cy1[y] * cx0[x] / (COEF * COEF)
        w11 = cy1[y] * cx1[x] / (COEF * COEF)
        exact = s00 * w00 + s01 * w01 + s10 * w10 + s11 * w11
        print(f"  {y:2d} {x:3d} | {s00:5d} {s01:5d} {s10:5d} {s11:5d} | "
              f"{w00:9.6f} {w01:9.6f} {w10:9.6f} {w11:9.6f} | {exact:7.4f} {ref[y, x]:4d} "
              f"{got_a[y, x]:3d} {d[y, x]:+d}")

    print("\nweight sums (should be 1):", float((cy0[:, None] + cy1[:, None])[0] / COEF / COEF * COEF + 0))
    print("cx0+cx1 unique:", np.unique(cx0 + cx1).tolist())
    print("cy0+cy1 unique:", np.unique(cy0 + cy1).tolist())


if __name__ == "__main__":
    main()
