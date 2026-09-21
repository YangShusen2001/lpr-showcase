"""Evaluate the fetched upstream images, record the verdict as UTF-8 JSON, and
promote the ones worth demoing into assets/samples/ under ASCII names.

Run:  python tools/promote_samples.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import hlpr_reference as H

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "_evidence"
STAGE = ROOT / "assets" / "samples" / "upstream"
SAMPLES = ROOT / "assets" / "samples"

# staged file -> (destination name, role)
PROMOTE = {
    "2.jpg": ("scene-2.jpg", "full scene, yellow plate, ~8% of frame"),
    "a.jpg": ("plate-yellow-320x48.jpg", "tight yellow plate crop"),
}


def main() -> None:
    det = H.sess("y5fu_320x_sim.onnx")
    rec = H.sess("rpv3_mdict_160_r3.onnx")
    cls = H.sess("litemodel_cls_96x_r1.onnx")

    report = {"images": []}
    for p in sorted(STAGE.glob("*.jpg")):
        img = H.imread_u(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        plates = H.run_pipeline(img, det, rec, cls, full=True)
        entry = {"file": p.name, "size": [w, h], "plates": plates}
        report["images"].append(entry)
        print(f"{p.name:16s} {w}x{h} -> {[pl['code'] for pl in plates]}")

    (EVID / "upstream_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- promoting ---")
    promoted = []
    for src_name, (dst_name, role) in PROMOTE.items():
        src = STAGE / src_name
        if not src.exists():
            print(f"  MISSING {src_name}")
            continue
        dst = SAMPLES / dst_name
        shutil.copy2(src, dst)
        promoted.append({"file": dst_name, "role": role, "bytes": dst.stat().st_size})
        print(f"  {src_name} -> {dst_name}  ({role})")

    (EVID / "promoted_samples.json").write_text(
        json.dumps(promoted, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
