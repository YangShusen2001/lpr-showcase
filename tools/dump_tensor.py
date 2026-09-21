"""Dump the detector's letterboxed input tensor as ground truth for the resize fitter.

Writes, into _evidence/:
  tensor_src.rgba  — the source image as raw RGBA (BGR swapped, alpha forced to 255)
  tensor_lb.f32    — the letterboxed, RGB, /255, NCHW tensor as little-endian float32
  tensor_meta.json — shapes and the letterbox parameters

tools/fit_resize.mjs consumes these to find the exact resize variant that reproduces
cv2's INTER_LINEAR bit for bit.

Run:  python tools/dump_tensor.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "_evidence"
EVID.mkdir(exist_ok=True)

IMAGES = ["hlpr-test.jpg", "hlpr-1.jpg", "scene-2.jpg"]
SIZE = 320


def main() -> None:
    meta = {"size": SIZE, "images": []}
    for name in IMAGES:
        bgr = H.imread_u(H.SAMPLES / name)
        h, w = bgr.shape[:2]

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = bgr[:, :, 2]
        rgba[:, :, 1] = bgr[:, :, 1]
        rgba[:, :, 2] = bgr[:, :, 0]
        rgba[:, :, 3] = 255

        lb, r, left, top = H.letter_box(bgr, (SIZE, SIZE))
        rgb = lb[:, :, ::-1]
        nchw = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)

        stem = Path(name).stem
        (EVID / f"tensor_src_{stem}.rgba").write_bytes(rgba.tobytes())
        (EVID / f"tensor_lb_{stem}.f32").write_bytes(nchw.astype("<f4").tobytes())

        meta["images"].append({
            "name": name, "stem": stem, "srcW": w, "srcH": h,
            "r": r, "left": left, "top": top,
            "lbShape": list(nchw.shape), "nchwBytes": int(nchw.size * 4),
        })
        print(f"{name:16s} {w}x{h} -> r={r:.6f} left={left} top={top} lb={nchw.shape}")

    (EVID / "tensor_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nwrote tensor_meta.json + per-image .rgba/.f32")


if __name__ == "__main__":
    main()
