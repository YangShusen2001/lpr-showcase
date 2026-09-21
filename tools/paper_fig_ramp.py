"""OpenCV side of the scale sweep behind the paper's "+1 LSB" ramp claim.

Reads the port's resized buffers dumped by tools/paper_fig_ramp.mjs, reproduces
the identical source pattern, runs cv2.resize(INTER_LINEAR) on the same
destination sizes, and bins P(difference) by the fractional part of the source
coordinate.

Writes _evidence/paper_ramp.json.

Run:  .venv\\Scripts\\python.exe tools\\paper_fig_ramp.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "_evidence" / "ramp"

NBINS = 20


def make_source(w: int, h: int) -> np.ndarray:
    """Same 32-bit LCG as tools/paper_fig_ramp.mjs."""
    s = 0x12345678
    out = np.zeros((w * h, 4), dtype=np.uint8)
    for i in range(w * h):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i, 0] = s & 0xFF
        out[i, 1] = (s >> 8) & 0xFF
        out[i, 2] = (s >> 16) & 0xFF
        out[i, 3] = 255
    return out.reshape(h, w, 4)


def main() -> None:
    meta = json.loads((EVID / "port_resized.json").read_text(encoding="utf-8"))
    blob = (EVID / "port_resized.bin").read_bytes()
    sw, sh = meta["srcW"], meta["srcH"]
    src = make_source(sw, sh)
    src_rgb = np.ascontiguousarray(src[:, :, :3])

    n_pix = np.zeros(NBINS, dtype=np.int64)
    n_diff = np.zeros(NBINS, dtype=np.int64)
    n_pos = np.zeros(NBINS, dtype=np.int64)

    # split by whether the vertical scale is exactly 1: the residual turns out to
    # be governed by that, not by the fractional coordinate
    ve_pix = np.zeros(NBINS, dtype=np.int64)
    ve_diff = np.zeros(NBINS, dtype=np.int64)
    vf_pix = np.zeros(NBINS, dtype=np.int64)
    vf_diff = np.zeros(NBINS, dtype=np.int64)

    # diagnostic: does OpenCV's channel count change the SIMD path?
    diag4_pix = 0
    diag4_diff = 0

    per_config = []
    for cfg in meta["configs"]:
        dw, dh = cfg["dstW"], cfg["dstH"]
        raw = np.frombuffer(blob, dtype=np.uint8, count=cfg["len"], offset=cfg["offset"])
        got = raw.reshape(dh, dw, 4)[:, :, :3]

        ref = cv2.resize(src_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
        ref4 = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)[:, :, :3]
        diag4_pix += ref4.size
        diag4_diff += int(np.count_nonzero(ref4 != ref))

        diff = got.astype(np.int16) - ref.astype(np.int16)      # (dh, dw, 3)
        any_d = np.any(diff != 0, axis=2)                        # (dh, dw)
        pos_d = np.any(diff == 1, axis=2) & ~np.any(diff < 0, axis=2)

        # fractional part of the source coordinate for each destination column
        fx = (np.arange(dw) + 0.5) * sw / dw - 0.5
        sx = np.floor(fx).astype(int)
        frac = fx - sx
        interior = (sx >= 0) & (sx <= sw - 2)
        idx = np.clip((frac * NBINS).astype(int), 0, NBINS - 1)

        # rows are restricted the same way, so the vertical pass is also interior
        fy = (np.arange(dh) + 0.5) * sh / dh - 0.5
        sy = np.floor(fy).astype(int)
        rows_ok = (sy >= 0) & (sy <= sh - 2)
        any_d = any_d[rows_ok, :]
        pos_d = pos_d[rows_ok, :]
        nrows = int(rows_ok.sum())

        for b in np.unique(idx[interior]):
            m = interior & (idx == b)
            n_pix[b] += int(m.sum()) * nrows
            n_diff[b] += int(any_d[:, m].sum())
            n_pos[b] += int(pos_d[:, m].sum())
            if dh == sh:
                ve_pix[b] += int(m.sum()) * nrows
                ve_diff[b] += int(any_d[:, m].sum())
            else:
                vf_pix[b] += int(m.sum()) * nrows
                vf_diff[b] += int(any_d[:, m].sum())

        per_config.append({
            "dstW": dw,
            "dstH": dh,
            "nDiff": int(any_d.sum()),
            "nPix": int(any_d.size),
        })

    p_diff = np.divide(n_diff, n_pix, out=np.zeros(NBINS), where=n_pix > 0)
    p_pos = np.divide(n_pos, n_pix, out=np.zeros(NBINS), where=n_pix > 0)

    def series(pix, dif):
        r = np.divide(dif, pix, out=np.zeros(NBINS), where=pix > 0)
        return [
            {"lo": b / NBINS, "hi": (b + 1) / NBINS, "mid": (b + 0.5) / NBINS,
             "nPix": int(pix[b]), "nDiff": int(dif[b]), "pDiff": float(r[b])}
            for b in range(NBINS)
        ]

    out = {
        "srcW": sw,
        "srcH": sh,
        "nConfigs": len(meta["configs"]),
        "totalPix": int(n_pix.sum()),
        "totalDiff": int(n_diff.sum()),
        "totalPos": int(n_pos.sum()),
        "buckets": [
            {
                "lo": b / NBINS,
                "hi": (b + 1) / NBINS,
                "mid": (b + 0.5) / NBINS,
                "nPix": int(n_pix[b]),
                "nDiff": int(n_diff[b]),
                "nPos": int(n_pos[b]),
                "pDiff": float(p_diff[b]),
                "pPos": float(p_pos[b]),
            }
            for b in range(NBINS)
        ],
        "vertExact": {
            "nPix": int(ve_pix.sum()), "nDiff": int(ve_diff.sum()),
            "pDiff": float(ve_diff.sum() / ve_pix.sum()) if ve_pix.sum() else 0.0,
            "buckets": series(ve_pix, ve_diff),
        },
        "vertFrac": {
            "nPix": int(vf_pix.sum()), "nDiff": int(vf_diff.sum()),
            "pDiff": float(vf_diff.sum() / vf_pix.sum()) if vf_pix.sum() else 0.0,
            "buckets": series(vf_pix, vf_diff),
        },
        "exactConfigs": [f'{c["dstW"]}x{c["dstH"]}' for c in per_config if c["nDiff"] == 0],
        "diag4": {"pix": diag4_pix, "diff": diag4_diff,
                  "pDiff": diag4_diff / diag4_pix if diag4_pix else 0.0},
    }
    dst = ROOT / "_evidence" / "paper_ramp.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"configs={out['nConfigs']} pixels={out['totalPix']} "
          f"diff={out['totalDiff']} ({100*out['totalDiff']/out['totalPix']:.3f}%) "
          f"pos={out['totalPos']} exact={len(out['exactConfigs'])}")
    print(f"  diag: 4-ch vs 3-ch OpenCV reference differ on "
          f"{100*out['diag4']['pDiff']:.4f}% of pixels")
    print(f"  vertical scale == 1 : {100*out['vertExact']['pDiff']:.3f}% "
          f"({out['vertExact']['nPix']} px)")
    print(f"  vertical scale != 1 : {100*out['vertFrac']['pDiff']:.3f}% "
          f"({out['vertFrac']['nPix']} px)")
    for b in out["buckets"]:
        print(f"  frac {b['lo']:.2f}-{b['hi']:.2f}  P(diff)={b['pDiff']*100:6.2f}%  "
              f"P(+1)={b['pPos']*100:6.2f}%  n={b['nPix']}")
    print("wrote", dst)


if __name__ == "__main__":
    main()
