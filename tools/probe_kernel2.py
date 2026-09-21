"""2-D impulse probe: isolate the vertical pass of cv2.resize(INTER_LINEAR).

The horizontal pass already matches (see tools/probe_kernel.py), yet the full 2-D
resize still disagrees by +1 on ~10% of pixels with a systematic upward bias. A single
non-zero pixel makes rounding unambiguous, so the 2x2 response pins down the vertical
layout.

Run:  python tools/probe_kernel2.py
"""
from __future__ import annotations

import numpy as np
import cv2

SRC_W, SRC_H = 1140, 456
DST_W, DST_H = 320, 128
COEF = 1 << 11


def w1(n_src: int, n_dst: int):
    i = np.arange(n_dst)
    f = (i + 0.5) * (n_src / n_dst) - 0.5
    s = np.floor(f).astype(np.int64)
    w = f - s
    s = np.clip(s, 0, n_src - 2)
    return s, w


def main() -> None:
    sx, wx = w1(SRC_W, DST_W)
    sy, wy = w1(SRC_H, DST_H)

    print(f"H: {SRC_W}->{DST_W}  scale={SRC_W / DST_W!r}")
    print(f"V: {SRC_H}->{DST_H}  scale={SRC_H / DST_H!r}\n")

    # 2-D impulse: single non-zero source pixel -> 2x2 output neighbourhood
    for py, px in ((100, 100), (101, 101), (200, 300), (255, 500)):
        img = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
        img[py, px, :] = 255
        out = cv2.resize(img, (DST_W, DST_H))[..., 0].astype(np.int64)
        nz = np.argwhere(out > 0)
        print(f"impulse @(y={py},x={px}):")
        for y, x in nz:
            # which source taps feed this output pixel?
            hs = [k for k in range(DST_W) if sx[k] <= px <= sx[k] + 1]
            pred = None
            for k in hs:
                if k != x:
                    continue
                hw = (1 - wx[k]) if sx[k] == px else wx[k]
                vw = (1 - wy[y]) if sy[y] == py else (wy[y] if sy[y] + 1 == py else 0.0)
                pred = 255 * hw * vw
            print(f"    out[{y},{x}] = {out[y, x]:4d}   pred={pred if pred is None else round(pred, 4)}")

    # vertical-only impulse (single column) to remove the horizontal pass entirely
    print("\nvertical-only (456x1 -> 128x1):")
    for py in (100, 101, 102, 200):
        img = np.zeros((SRC_H, 1, 3), dtype=np.uint8)
        img[py, 0, :] = 255
        out = cv2.resize(img, (1, DST_H))[0, :, 0].astype(np.int64)
        nz = np.nonzero(out)[0]
        for y in nz:
            vw = (1 - wy[y]) if sy[y] == py else (wy[y] if sy[y] + 1 == py else 0.0)
            print(f"  impulse @{py}: out[{int(y)}]={int(out[y]):4d}  pred={255 * vw:.4f}  "
                  f"pred_round={round(255 * vw)}")

    # horizontal-only impulse (single row)
    print("\nhorizontal-only (1x1140 -> 1x320):")
    for px in (100, 101, 102, 200):
        img = np.zeros((1, SRC_W, 3), dtype=np.uint8)
        img[0, px, :] = 255
        out = cv2.resize(img, (DST_W, 1))[0, :, 0].astype(np.int64)
        nz = np.nonzero(out)[0]
        for x in nz:
            hw = (1 - wx[x]) if sx[x] == px else (wx[x] if sx[x] + 1 == px else 0.0)
            print(f"  impulse @{px}: out[{int(x)}]={int(out[x]):4d}  pred={255 * hw:.4f}  "
                  f"pred_round={round(255 * hw)}")


if __name__ == "__main__":
    main()
