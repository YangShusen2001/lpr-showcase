"""Recover cv2.resize(INTER_LINEAR)'s effective filter by impulse response.

Feeding a single non-zero pixel makes the rounding irrelevant and exposes the exact
fixed-point taps, which pins down the coefficient scale and the shift layout that the
black-box behaviour implies.

Run:  python tools/probe_kernel.py
"""
from __future__ import annotations

import numpy as np
import cv2


def impulse_probe(src_w: int, dst_w: int, at: int, value: int = 255) -> np.ndarray:
    img = np.zeros((1, src_w, 3), dtype=np.uint8)
    img[0, at, :] = value
    out = cv2.resize(img, (dst_w, 1))
    return out[0, :, 0].astype(np.int64)


def taps(src_w: int, dst_w: int, at: int) -> list[tuple[int, float]]:
    """The fractional weights my formula predicts for source pixel `at`."""
    res = []
    for x in range(dst_w):
        f = (x + 0.5) * (src_w / dst_w) - 0.5
        s = int(np.floor(f))
        w = f - s
        s = min(max(s, 0), src_w - 2)
        if at == s:
            res.append((x, 1 - w))
        elif at == s + 1:
            res.append((x, w))
    return res


def main() -> None:
    for src_w, dst_w in ((1140, 320), (1920, 320)):
        scale = src_w / dst_w
        print(f"\n=== {src_w} -> {dst_w}   scale={scale!r} ===")
        for at in (100, 101, 102, 103):
            out = impulse_probe(src_w, dst_w, at)
            nz = np.nonzero(out)[0]
            if len(nz) == 0:
                print(f"  impulse @{at}: (no output pixel responded)")
                continue
            pred = dict(taps(src_w, dst_w, at))
            desc = ", ".join(
                f"x={int(x)}: out={int(out[x])} pred_w={pred.get(int(x), 0.0):.6f} "
                f"pred={pred.get(int(x), 0.0) * 255:.3f}"
                for x in nz
            )
            print(f"  impulse @{at}: {desc}")

    # A flat mid-grey field reveals any DC bias / rounding direction.
    print("\n=== flat field ===")
    for src_w, dst_w in ((1140, 320), (1920, 320)):
        for val in (1, 2, 3, 127, 128, 255):
            img = np.full((1, src_w, 3), val, dtype=np.uint8)
            out = cv2.resize(img, (dst_w, 1))[0, :, 0]
            uniq = np.unique(out)
            print(f"  {src_w}->{dst_w} in={val:3d} -> out={uniq.tolist()}")


if __name__ == "__main__":
    main()
