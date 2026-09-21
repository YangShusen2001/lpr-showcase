"""One-off: compute the rectified-crop RGB checksum from the Python reference.

The native C++ port reports `cropSum` (sum of R,G,B over the rectified crop, alpha
excluded). pipeline.js documents the same field as comparable to numpy's BGR sum.
Matching checksums prove detection + rectification agree; a mismatch localises the
divergence to those two stages instead of the recogniser.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import hlpr_reference as R  # noqa: E402

det = R.sess("y5fu_320x_sim.onnx")

for name in ["hlpr-test.jpg", "scene-2.jpg"]:
    img = R.imread_u(R.SAMPLES / name)
    dets = R.detect(det, img, 0.25, 0.5)
    print(f"== {name}  img={img.shape[1]}x{img.shape[0]}  dets={len(dets)}")
    for row in dets:
        marks = row[5:13].reshape(4, 2).astype(int)
        crop = R.get_rotate_crop_image(img, marks)
        print(f"   rect={row[:4].astype(int).tolist()} score={float(row[4]):.4f} "
              f"layer={int(row[13])}")
        print(f"   marks={marks.tolist()}")
        print(f"   crop_shape={list(crop.shape[:2])}  crop_sum={int(crop.sum())}")
