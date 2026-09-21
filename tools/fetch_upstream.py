"""Fetch the remaining upstream HyperLPR test photos and score them for demo use.

Downloads into assets/samples/upstream/ then runs the full pipeline on each and
prints the detected plate size as a fraction of the frame, so the best demo
material can be chosen on evidence rather than guesswork.

Run:  python tools/fetch_upstream.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "assets" / "samples" / "upstream"
DEST.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/szad670401/HyperLPR/master/"
WANTED = [
    "resource/images/1.jpg",
    "resource/images/2.jpg",
    "resource/images/a.jpg",
    "resource/images/test_img.jpg",
    "resource/images/align/1.jpg",
    "resource/images/align/3.jpg",
    "resource/images/align/5.jpg",
]

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(rel: str) -> Path:
    out = DEST / Path(rel).name
    if out.exists() and out.stat().st_size > 0:
        return out
    with opener.open(BASE + rel, timeout=120) as r:
        data = r.read()
    out.write_bytes(data)
    return out


def main() -> None:
    det = H.sess("y5fu_320x_sim.onnx")
    rec = H.sess("rpv3_mdict_160_r3.onnx")
    cls = H.sess("litemodel_cls_96x_r1.onnx")

    rows = []
    for rel in WANTED:
        try:
            p = fetch(rel)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {rel}: {e}")
            continue
        img = H.imread_u(p)
        if img is None:
            print(f"[skip] {rel}: undecodable")
            continue
        h, w = img.shape[:2]
        plates = H.run_pipeline(img, det, rec, cls, full=True)
        print(f"\n{rel}  {w}x{h}  {p.stat().st_size // 1024}KB  -> {len(plates)} plate(s)")
        best = 0.0
        for pl in plates:
            ch, cw = pl["crop_shape"]
            frac = (cw * ch) / float(w * h)
            best = max(best, frac)
            print(f"    code={pl['code']:12s} conf={pl['rec_conf']:.3f} det={pl['det_score']:.3f} "
                  f"layer={pl['layer']} crop={ch}x{cw} area={frac * 100:.2f}%  cls={pl['cls']}")
        rows.append((rel, w, h, len(plates), best))

    print("\n=== ranked by plate area fraction ===")
    for rel, w, h, n, frac in sorted(rows, key=lambda r: -r[4]):
        print(f"  {frac * 100:6.2f}%  {w}x{h}  plates={n}  {rel}")


if __name__ == "__main__":
    main()
