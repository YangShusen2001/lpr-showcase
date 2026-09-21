"""Golden-string acceptance for LPRNet: does it get `苏ED5172` right?

⚠️⚠️ DEPRECATED JUDGEMENT (2026-09-18) — read before citing any PASS/FAIL here.
The "red line" this script checks is NOT a human label: `苏ED5172` is what rpv3
itself outputs on `hlpr-test.jpg`. Validating a model against that string is
validating the system against its own output. A16 established the correct reading
is the eight-character `苏ED51712` (the plate is yaw-compressed ~2x; a standard-aspect
re-read gives 8 chars at conf 0.987; the n=200 control in tools/stretch_control.py
shows stretching never invents a character). Accuracy is now judged on 1000
human-labelled crops -- see tools/lprnet_real_accuracy.py and ADR-015.

The project's red line is not an aggregate accuracy figure -- it is that the
golden image `hlpr-test.jpg` must decode to exactly `苏ED5172`. ADR-007 records
that the NPU's 0.08% numerical drift flips that string (苏E05172), which is why
`det=CPU / rec=NPU / cls=CPU` is the production assignment.

So LPRNet is judged the same way: run it on the golden crop and print the string.
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

GOLDEN = "苏ED5172"


def lprnet(crop, net, iname, oname):
    x = cv2.resize(crop, (94, 24)).astype(np.float32)
    x = (x - 127.5) * 0.0078125
    x = x.transpose(2, 0, 1)[None, ...]
    out = np.asarray(net.run([oname], {iname: x})[0])[0]
    return L.ctc_greedy(np.argmax(out, axis=0)), out


def main():
    net = L.load()
    iname, oname = net.get_inputs()[0].name, net.get_outputs()[0].name
    rec = H.sess("rpv3_mdict_160_r3.onnx")

    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    det = H.sess("y5fu_320x_sim.onnx")
    rows = H.detect(det, img)
    print(f"detections: {len(rows)}")
    for i, r in enumerate(rows):
        pts = r[5:13].reshape(4, 2).astype(int)
        crop = H.get_rotate_crop_image(img, pts)
        code, logits = lprnet(crop, net, iname, oname)
        rp, conf = H.recognize(rec, crop)
        h, w = crop.shape[:2]
        print(f"  plate {i}: crop={w}x{h} det_score={r[4]:.4f}")
        print(f"    LPRNet : {code}   {'OK' if code == GOLDEN else 'MISMATCH (want ' + GOLDEN + ')'}")
        print(f"    rpv3   : {rp} (conf {conf:.3f})   {'OK' if rp == GOLDEN else 'MISMATCH'}")
        # per-step argmax around the failure point, for attribution
        idx = np.argmax(logits, axis=0)
        print(f"    LPRNet argmax row ({logits.shape[1]} steps): {list(idx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
