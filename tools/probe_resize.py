"""Probe cv2.resize(INTER_LINEAR) on a 2-D downscale to recover the exact algorithm.

The fixed-point variant fitted on hlpr-test.jpg (scale exactly 6.0) fails at
scene-2.jpg's scale of 3.5625, so that fit was over-fitted to one scale. This script
compares several candidate two-pass layouts against cv2 on the real failing case and
prints where each one diverges.

Run:  python tools/probe_resize.py
"""
from __future__ import annotations

import numpy as np
import cv2

from pathlib import Path

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "assets" / "samples" / "scene-2.jpg"
COEF_BITS = 11
COEF = 1 << COEF_BITS


def axes(sw: int, sh: int, dw: int, dh: int):
    def one(src_n, dst_n):
        i = np.arange(dst_n)
        f = (i + 0.5) * (src_n / dst_n) - 0.5
        s = np.floor(f).astype(np.int64)
        w = f - s
        s = np.clip(s, 0, src_n - 2)
        return s, w
    sx, wx = one(sw, dw)
    sy, wy = one(sh, dh)
    return sx, wx, sy, wy


def cand_unshifted_h(src, dw, dh):
    sh, sw = src.shape
    sx, wx, sy, wy = axes(sw, sh, dw, dh)
    cx0 = np.round((1 - wx) * COEF).astype(np.int64)
    cx1 = np.round(wx * COEF).astype(np.int64)
    cy0 = np.round((1 - wy) * COEF).astype(np.int64)
    cy1 = np.round(wy * COEF).astype(np.int64)
    tmp = src[:, sx] * cx0 + src[:, sx + 1] * cx1          # (sh, dw) int64
    v = tmp[sy, :] * cy0[:, None] + tmp[sy + 1, :] * cy1[:, None]
    return ((v + (1 << 21)) >> 22).astype(np.uint8)


def cand_shifted_both(src, dw, dh, round_h: bool, round_v: bool):
    sh, sw = src.shape
    sx, wx, sy, wy = axes(sw, sh, dw, dh)
    cx0 = np.round((1 - wx) * COEF).astype(np.int64)
    cx1 = np.round(wx * COEF).astype(np.int64)
    cy0 = np.round((1 - wy) * COEF).astype(np.int64)
    cy1 = np.round(wy * COEF).astype(np.int64)
    h = src[:, sx] * cx0 + src[:, sx + 1] * cx1
    h = (h + (1 << 10)) >> 11 if round_h else h >> 11
    v = h[sy, :] * cy0[:, None] + h[sy + 1, :] * cy1[:, None]
    v = (v + (1 << 10)) >> 11 if round_v else v >> 11
    return v.astype(np.uint8)


def cand_float(src, dw, dh):
    sh, sw = src.shape
    sx, wx, sy, wy = axes(sw, sh, dw, dh)
    top = src[:, sx] * (1 - wx) + src[:, sx + 1] * wx
    v = top[sy, :] * (1 - wy)[:, None] + top[sy + 1, :] * wy[:, None]
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def main() -> None:
    img = H.imread_u(SAMPLE)
    src = img[:, :, 0].astype(np.int64)          # blue channel only, keeps it readable
    sh, sw = src.shape
    dw, dh = 320, 128

    ref = cv2.resize(img, (dw, dh))[:, :, 0]

    print(f"source {sw}x{sh} -> {dw}x{dh}   scale_x={sw/dw!r}  scale_y={sh/dh!r}")
    print(f"integer scales? sx={sw/dw == round(sw/dw)} sy={sh/dh == round(sh/dh)}")
    print(f"cv2 ref: min={ref.min()} max={ref.max()} mean={ref.mean():.4f}\n")

    cands = {
        "unshifted H, V>>22": cand_unshifted_h(src, dw, dh),
        "H>>11(r), V>>11(r)": cand_shifted_both(src, dw, dh, True, True),
        "H>>11(t), V>>11(r)": cand_shifted_both(src, dw, dh, False, True),
        "H>>11(r), V>>11(t)": cand_shifted_both(src, dw, dh, True, False),
        "float bilinear": cand_float(src, dw, dh),
    }
    for name, got in cands.items():
        d = got.astype(np.int64) - ref.astype(np.int64)
        nz = int(np.count_nonzero(d))
        print(f"  {name:22s} mismatch={nz:7d}/{d.size}  maxAbs={int(np.abs(d).max())}  "
              f"mean={float(d.mean()):+.4f}")

    # where does the best candidate diverge, and how?
    best = cand_unshifted_h(src, dw, dh)
    d = best.astype(np.int64) - ref.astype(np.int64)
    idx = np.argwhere(d != 0)
    print(f"\nfirst 12 mismatches of 'unshifted H' (row, col, got, ref, delta):")
    for r, c in idx[:12]:
        print(f"  ({r:3d},{c:3d})  got={best[r, c]:3d}  ref={ref[r, c]:3d}  d={d[r, c]:+d}")

    rows = sorted(set(int(r) for r, _ in idx))
    cols = sorted(set(int(c) for _, c in idx))
    print(f"\nmismatching rows: {rows[:12]}{' ...' if len(rows) > 12 else ''}  (total {len(rows)})")
    print(f"mismatching cols: {cols[:12]}{' ...' if len(cols) > 12 else ''}  (total {len(cols)})")


if __name__ == "__main__":
    main()
