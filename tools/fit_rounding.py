"""Brute-force the exact rounding layout of cv2.resize(INTER_LINEAR).

Evidence gathered so far:
  * fixed-point coefficients with COEF_BITS=11 are exact for these scales
    (scale 3.5625 = 57/16 makes every weight a multiple of 64/2048)
  * the full 2-D result equals floor(exact bilinear) on every mismatching pixel
  * yet a 1-row input rounds 7.9688 up to 8

So the shift/rounding placement differs from the first guess. This enumerates the
plausible layouts and cross-checks each against both the 2-D image and a 1-row probe.

Run:  python tools/fit_rounding.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "assets" / "samples" / "scene-2.jpg"
COEF = 1 << 11


def weights(n_src: int, n_dst: int):
    i = np.arange(n_dst)
    f = (i + 0.5) * (n_src / n_dst) - 0.5
    s = np.floor(f).astype(np.int64)
    w = f - s
    s = np.clip(s, 0, max(n_src - 2, 0))
    return s, w


def build(src: np.ndarray, dst_w: int, dst_h: int, hshift: int, hround: int,
          vshift: int, vround: int) -> np.ndarray:
    sh, sw = src.shape
    sx, wx = weights(sw, dst_w)
    sy, wy = weights(sh, dst_h)
    cx0 = np.round((1 - wx) * COEF).astype(np.int64)
    cx1 = np.round(wx * COEF).astype(np.int64)
    cy0 = np.round((1 - wy) * COEF).astype(np.int64)
    cy1 = np.round(wy * COEF).astype(np.int64)

    tmp = src[:, sx] * cx0 + src[:, np.minimum(sx + 1, sw - 1)] * cx1
    tmp = tmp if hshift == 0 else (tmp + hround) >> hshift
    sy1 = np.minimum(sy + 1, sh - 1)
    v = tmp[sy, :] * cy0[:, None] + tmp[sy1, :] * cy1[:, None]
    v = v if vshift == 0 else (v + vround) >> vshift
    return v


def main() -> None:
    img = H.imread_u(SAMPLE)
    src2d = img[:, :, 0].astype(np.int64)
    ref2d = cv2.resize(img, (320, 128))[:, :, 0].astype(np.int64)

    rng = np.random.default_rng(7)
    row = rng.integers(0, 256, size=(1, 1140), dtype=np.uint8)
    ref1d = cv2.resize(row.reshape(1, 1140), (320, 1)).astype(np.int64)[0]

    layouts = []
    for vround in (0, 1 << 21, (1 << 21) - 1):
        layouts.append((f"h>>0   v>>22 r={vround}", 0, 0, 22, vround))
    for hround in (0, 1 << 10):
        for vround in (0, 1 << 10, (1 << 10) - 1):
            layouts.append((f"h>>11 r={hround}  v>>11 r={vround}", 11, hround, 11, vround))

    print(f"{'layout':34s} {'2-D mismatch':>14s} {'1-row mismatch':>16s}")
    best = None
    for name, hs, hr, vs, vr in layouts:
        g2 = build(src2d, 320, 128, hs, hr, vs, vr)
        d2 = int(np.count_nonzero(g2 - ref2d))
        g1 = build(row.astype(np.int64), 320, 1, hs, hr, vs, vr)
        d1 = int(np.count_nonzero(g1 - ref1d))
        flag = '   <== EXACT' if d2 == 0 and d1 == 0 else ''
        print(f"  {name:32s} {d2:14d} {d1:16d}{flag}")
        if best is None or (d2 + d1) < best[0]:
            best = (d2 + d1, name, d2, d1)

    print(f"\nbest: {best[1]}  (2-D {best[2]}, 1-row {best[3]})")

    # If nothing is exact, show whether floor(exact) is the 2-D rule.
    sx, wx = weights(1140, 320)
    sy, wy = weights(456, 128)
    cx0 = np.round((1 - wx) * COEF).astype(np.int64)
    cx1 = np.round(wx * COEF).astype(np.int64)
    cy0 = np.round((1 - wy) * COEF).astype(np.int64)
    cy1 = np.round(wy * COEF).astype(np.int64)
    num = (src2d[:, sx] * cx0 + src2d[:, np.minimum(sx + 1, 1139)] * cx1)
    v = num[sy, :] * cy0[:, None] + num[np.minimum(sy + 1, 455), :] * cy1[:, None]
    fl = v >> 22
    print(f"\nfloor(exact) mismatch on 2-D: {int(np.count_nonzero(fl - ref2d))}/{fl.size}")
    print(f"  #cv2 == floor(exact)  : {int((fl == ref2d).sum())}")
    print(f"  #cv2 == floor+1       : {int((fl + 1 == ref2d).sum())}")
    print(f"  max |floor - cv2|     : {int(np.abs(fl - ref2d).max())}")


if __name__ == "__main__":
    main()
