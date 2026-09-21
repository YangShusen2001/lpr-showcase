"""Measure cv2.resize(INTER_LINEAR)'s effective rounding threshold and pass order.

Round-to-nearest explains 90.5% of pixels and floor 66.6%, so neither is the rule.
This bins pixels by the fractional part of the exact fixed-point numerator to read the
threshold off the data, and tests whether OpenCV runs the vertical pass first.

Run:  python tools/fit_rounding2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "assets" / "samples" / "scene-2.jpg"
COEF = 1 << 11
SX, SY = 1140, 456
DW, DH = 320, 128


def weights(n_src: int, n_dst: int):
    i = np.arange(n_dst)
    f = (i + 0.5) * (n_src / n_dst) - 0.5
    s = np.floor(f).astype(np.int64)
    w = f - s
    s = np.clip(s, 0, n_src - 2)
    return s, w


def main() -> None:
    img = H.imread_u(SAMPLE)
    src = img[:, :, 0].astype(np.int64)
    ref = cv2.resize(img, (DW, DH))[:, :, 0].astype(np.int64)

    sx, wx = weights(SX, DW)
    sy, wy = weights(SY, DH)
    cx0 = np.round((1 - wx) * COEF).astype(np.int64)
    cx1 = np.round(wx * COEF).astype(np.int64)
    cy0 = np.round((1 - wy) * COEF).astype(np.int64)
    cy1 = np.round(wy * COEF).astype(np.int64)

    sx1 = np.minimum(sx + 1, SX - 1)
    sy1 = np.minimum(sy + 1, SY - 1)

    # H then V, single rounding at the very end (22-bit numerator)
    h = src[:, sx] * cx0 + src[:, sx1] * cx1
    num_hv = h[sy, :] * cy0[:, None] + h[sy1, :] * cy1[:, None]

    # V then H, single rounding at the very end
    v = src[sy, :] * cy0[:, None] + src[sy1, :] * cy1[:, None]
    num_vh = v[:, sx] * cx0 + v[:, sx1] * cx1

    for name, num in (("H-then-V", num_hv), ("V-then-H", num_vh)):
        base = num >> 22
        d = base - ref
        plus1 = int((d == -1).sum())
        zero = int((d == 0).sum())
        print(f"{name:9s} floor matches {zero:6d}/{num.size} ({zero / num.size:.1%}), "
              f"needs +1 on {plus1:6d}, other {int(((d != 0) & (d != -1)).sum())}")

    print("\nthreshold analysis on H-then-V:")
    frac = (num_hv & ((1 << 22) - 1)) / float(1 << 22)
    need = (num_hv >> 22) + 1 == ref
    base_ok = (num_hv >> 22) == ref
    bins = np.linspace(0, 1, 21)
    print(f"  {'frac range':>16s} {'n':>8s} {'P(+1)':>8s}")
    for a, b in zip(bins[:-1], bins[1:]):
        m = (frac >= a) & (frac < b)
        n = int(m.sum())
        if n:
            print(f"  [{a:.2f},{b:.2f})        {n:8d} {float(need[m].mean()):8.3f}")

    # A clean way to see the rule: what fraction is needed for +1?
    ok = base_ok | need
    print(f"\n  explainable by floor-or-floor+1: {int(ok.sum())}/{frac.size}")
    lo = frac[need].min() if need.any() else None
    hi = frac[base_ok].max() if base_ok.any() else None
    print(f"  min frac among +1 pixels: {lo}")
    print(f"  max frac among floor pixels: {hi}")


if __name__ == "__main__":
    main()
