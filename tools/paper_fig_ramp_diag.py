"""Diagnostic: who deviates from exact bilinear -- the port, or cv2?

Compares, for a few destination sizes, the port's dumped resize output against
(a) cv2.resize(INTER_LINEAR) and (b) an exact double-precision bilinear resize.

Run:  .venv\\Scripts\\python.exe tools\\paper_fig_ramp_diag.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "_evidence" / "ramp"


def make_source(w: int, h: int) -> np.ndarray:
    s = 0x12345678
    out = np.zeros((w * h, 4), dtype=np.uint8)
    for i in range(w * h):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i, 0] = s & 0xFF
        out[i, 1] = (s >> 8) & 0xFF
        out[i, 2] = (s >> 16) & 0xFF
        out[i, 3] = 255
    return out.reshape(h, w, 4)


def exact_bilinear(src: np.ndarray, dw: int, dh: int) -> np.ndarray:
    sh, sw = src.shape[:2]
    s = src.astype(np.float64)
    fx = (np.arange(dw) + 0.5) * sw / dw - 0.5
    sx = np.clip(np.floor(fx).astype(int), 0, sw - 2)
    ax = (fx - np.floor(fx))[None, :, None]
    fy = (np.arange(dh) + 0.5) * sh / dh - 0.5
    sy = np.clip(np.floor(fy).astype(int), 0, sh - 2)
    ay = (fy - np.floor(fy))[:, None, None]
    tmp = s[:, sx] * (1 - ax) + s[:, sx + 1] * ax
    out = tmp[sy, :] * (1 - ay) + tmp[sy + 1, :] * ay
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main() -> None:
    meta = json.loads((EVID / "port_resized.json").read_text(encoding="utf-8"))
    blob = (EVID / "port_resized.bin").read_bytes()
    src = make_source(meta["srcW"], meta["srcH"])
    rgb = np.ascontiguousarray(src[:, :, :3])

    rows = []
    for cfg in meta["configs"]:
        dw, dh = cfg["dstW"], cfg["dstH"]
        if (dw, dh) not in [(70, 72), (130, 108), (200, 72), (300, 216), (70, 144)]:
            continue
        raw = np.frombuffer(blob, dtype=np.uint8, count=cfg["len"], offset=cfg["offset"])
        got = raw.reshape(dh, dw, 4)[:, :, :3]
        ref = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        exa = exact_bilinear(rgb, dw, dh)

        def stats(a, b, label):
            d = a.astype(np.int16) - b.astype(np.int16)
            return (f"{label}: n={d.size} diff={np.count_nonzero(d)} "
                    f"mean={d.mean():+.4f} min={d.min()} max={d.max()}")

        rows.append(f"--- {dw}x{dh} (src {meta['srcW']}x{meta['srcH']})")
        rows.append("  " + stats(got, ref, "port - cv2 "))
        rows.append("  " + stats(got, exa, "port - exact"))
        rows.append("  " + stats(ref, exa, "cv2  - exact"))

    out = "\n".join(rows)
    (EVID / "diag.txt").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
