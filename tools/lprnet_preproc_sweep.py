"""Is LPRNet's miss a model limit or my preprocessing? Sweep the standard variants.

The head-to-head gave LPRNet 3/5 against rpv3's 5/5, including a miss on the golden
expected string. Before writing "LPRNet cannot replace rpv3" that has to be
attributed: a stretch-resize to 94x24 distorts the aspect ratio, and LPRNet's own
recipe is aspect-preserving padding. If a standard variant fixes the golden string,
the earlier miss was the harness, not the model.

Variants: {stretch, letterbox} x {BGR, RGB} x {(x-127.5)/128, (x-127.5)/127.5, x/255}
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402
import lprnet_reference as L          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SAMPLES = ROOT / "assets" / "samples"
GOLDEN = "苏ED5172"
CASES = [("crop-0-津B6H920.jpg", "津B6H920"), ("crop-1-皖KD01833.jpg", "皖KD01833"),
         ("crop-6-蒙B023H6.jpg", "蒙B023H6"), ("crop-8-冀D5L690.jpg", "冀D5L690")]


def build(crop, fit, order, norm, W=94, Hh=24):
    if fit == "stretch":
        x = cv2.resize(crop, (W, Hh))
    else:                                    # letterbox: keep aspect, pad with 0
        h, w = crop.shape[:2]
        r = min(Hh / h, W / w)
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        x = np.zeros((Hh, W, 3), np.uint8)
        y0, x0 = (Hh - nh) // 2, (W - nw) // 2
        x[y0:y0 + nh, x0:x0 + nw] = cv2.resize(crop, (nw, nh))
    if order == "RGB":
        x = x[:, :, ::-1]
    x = x.astype(np.float32)
    if norm == "pm127.5_div128":
        x = (x - 127.5) / 128.0
    elif norm == "pm127.5":
        x = (x - 127.5) / 127.5
    else:
        x = x / 255.0
    return x.transpose(2, 0, 1)[None, ...]


def main():
    net = L.load()
    iname, oname = net.get_inputs()[0].name, net.get_outputs()[0].name

    # the golden crop, produced exactly the way the device pipeline produces it
    img = H.imread_u(SAMPLES / "hlpr-test.jpg")
    det = H.sess("y5fu_320x_sim.onnx")
    rows = H.detect(det, img)
    golden_crop = H.get_rotate_crop_image(img, rows[0][5:13].reshape(4, 2).astype(int))
    print(f"golden crop shape (h,w) = {golden_crop.shape[:2]}, expected {GOLDEN}")

    cases = list(CASES) + [("__golden__", GOLDEN)]
    crops = {name: (golden_crop if name == "__golden__" else H.imread_u(SAMPLES / name))
             for name, _ in cases}

    best = None
    for fit in ("stretch", "letterbox"):
        for order in ("BGR", "RGB"):
            for norm in ("pm127.5_div128", "pm127.5", "div255"):
                hits, preds = 0, []
                for name, gt in cases:
                    x = build(crops[name], fit, order, norm)
                    out = np.asarray(net.run([oname], {iname: x})[0])[0]
                    code = L.ctc_greedy(np.argmax(out, axis=0))
                    preds.append(code)
                    hits += (code == gt)
                gold = preds[-1]
                flag = "  <-- GOLDEN OK" if gold == GOLDEN else ""
                print(f"  fit={fit:9s} order={order} norm={norm:14s} "
                      f"{hits}/{len(cases)}  golden={gold}{flag}")
                if best is None or hits > best[0]:
                    best = (hits, fit, order, norm, gold)
    print(f"\nBEST: {best[0]}/{len(cases)}  fit={best[1]} order={best[2]} "
          f"norm={best[3]}  golden={best[4]}")

    # Wider input? The ONNX input is fixed at 24x94, so also record the true aspect
    # ratio of each crop -- a plate much wider than 94/24 = 3.92 is being squashed.
    print("\naspect ratios (w/h): input 94/24 = 3.92")
    for name, _ in cases:
        h, w = crops[name].shape[:2]
        print(f"  {name:22s} {w}x{h}  ratio={w / h:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
