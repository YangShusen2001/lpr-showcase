"""Head-to-head: LPRNet vs the production recogniser (rpv3 / SVTR) on the SAME crops.

⚠️ SUPERSEDED (2026-09-18) — two defects in this script's framing, see ADR-015:
  1. Its ground truth is `EXPECTED = 苏ED5172`, which is rpv3's OWN output on the
     golden image (n=5 samples total). Circular. The real accuracy acceptance is
     tools/lprnet_real_accuracy.py on 1000 human-labelled crops:
     rpv3 906/1000 = 90.6% vs LPRNet 888/1000 = 88.8%, McNemar p=0.1788.
  2. The claim below that "LPRNet's CTC character set is not the project's 77-entry
     table" is FALSE -- upstream `_load_data.py` matches the recovered 77-entry map
     item for item (A16 §3). ADR-007 §5's alignment TODO has been withdrawn.
Kept only as the historical n=5 step.

This is the accuracy acceptance A13 §5 left open. ADR-007 §5 and A13 §5 both say
the same thing: LPRNet has only ever been shown to *run* on the NPU, never to
*read correctly*, and its CTC character set is not the project's 77-entry table.
Until that is measured, "LPRNet can replace rpv3" is not a claim anyone may make.

Method (device-free, onnxruntime CPU, so it is about the models, not the backend):

  * full photos  -> the project's own reference detector + rectifier produce the
    crop, then BOTH recognisers read that exact same crop;
  * labelled crops -> the file itself is the crop.

Ground truth is the expected string `苏ED5172` for the golden image
`assets/samples/hlpr-test.jpg`, and the filename for `crop-*.jpg`.

Run:  ./.venv/Scripts/python.exe tools/lprnet_vs_rpv3.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import hlpr_reference as H            # noqa: E402  (project reference implementation)
import lprnet_reference as L          # noqa: E402  (recovered LPRNet alphabet + preprocess)

SAMPLES = ROOT / "assets" / "samples"
EVID = ROOT / "_evidence"

EXPECTED = "苏ED5172"                 # golden expected string for hlpr-test.jpg

# Photos whose ground truth is known independently of the model.
FULL_PHOTOS = {
    "hlpr-test.jpg": EXPECTED,
    "hlpr-1.jpg": None,               # unlabelled upstream demo photo
    "plate-1.jpg": None,
    "plate-2.jpg": None,
    "plate-3.jpg": None,
}


def lprnet_read(sess, iname, oname, crop):
    """Run LPRNet on one already-rectified plate crop."""
    x = L.preprocess(crop, "pm127.5_div128")
    out = np.asarray(sess.run([oname], {iname: x})[0])[0]
    return L.ctc_greedy(np.argmax(out, axis=0))


def main():
    det = H.sess("y5fu_320x_sim.onnx")
    rpv3 = H.sess("rpv3_mdict_160_r3.onnx")
    net = L.load()
    iname = net.get_inputs()[0].name
    oname = net.get_outputs()[0].name

    report = {"expected": EXPECTED, "rows": []}

    def score(tag, crop, gt):
        t0 = time.perf_counter()
        code_r, conf_r = H.recognize(rpv3, crop)
        t1 = time.perf_counter()
        code_l = lprnet_read(net, iname, oname, crop)
        t2 = time.perf_counter()
        row = {"tag": tag, "gt": gt, "rpv3": code_r, "rpv3_conf": round(conf_r, 4),
               "lprnet": code_l,
               "rpv3_ms": round((t1 - t0) * 1000, 2),
               "lprnet_ms": round((t2 - t1) * 1000, 2),
               "rpv3_ok": (code_r == gt) if gt else None,
               "lprnet_ok": (code_l == gt) if gt else None,
               "crop_shape": list(crop.shape[:2])}
        report["rows"].append(row)
        mark_r = "" if gt is None else ("  ✅" if code_r == gt else "  ❌")
        mark_l = "" if gt is None else ("  ✅" if code_l == gt else "  ❌")
        print(f"  {tag:34s} gt={str(gt):10s} "
              f"rpv3={code_r:12s}{mark_r}   lprnet={code_l:12s}{mark_l}")
        return row

    # ---- 1) the four ground-truth crops (pure recogniser comparison) ----------
    print("=== labelled crops (recogniser only) ===")
    for p in sorted(SAMPLES.glob("crop-*.jpg")):
        gt = p.stem.split("-", 2)[-1]
        score(p.name, H.imread_u(p), gt)

    # ---- 2) full photos: detect + rectify, then both recognisers ---------------
    for name, gt in FULL_PHOTOS.items():
        p = SAMPLES / name
        if not p.exists():
            continue
        print(f"\n=== {name} ({gt or 'unlabelled'}) ===")
        img = H.imread_u(p)
        dets = H.detect(det, img)
        if len(dets) == 0:
            print("  (no plate detected)")
            continue
        for i, row in enumerate(dets):
            crop = H.get_rotate_crop_image(img, row[5:13].reshape(4, 2).astype(int))
            score(f"{name}#{i}", crop, gt)

    # ---- 3) verdict ----------------------------------------------------------
    scored = [r for r in report["rows"] if r["gt"]]
    n = len(scored)
    ok_r = sum(1 for r in scored if r["rpv3_ok"])
    ok_l = sum(1 for r in scored if r["lprnet_ok"])
    print(f"\n=== accuracy on labelled inputs ===")
    print(f"  rpv3   {ok_r}/{n}")
    print(f"  lprnet {ok_l}/{n}")
    report["summary"] = {"labelled": n, "rpv3_ok": ok_r, "lprnet_ok": ok_l}

    golden = [r for r in report["rows"] if r["tag"].startswith("hlpr-test")]
    if golden:
        g = golden[0]
        print(f"\n=== golden expected string {EXPECTED} ===")
        print(f"  rpv3   -> {g['rpv3']:12s} {'MATCH' if g['rpv3'] == EXPECTED else 'MISMATCH'}")
        print(f"  lprnet -> {g['lprnet']:12s} {'MATCH' if g['lprnet'] == EXPECTED else 'MISMATCH'}")

    (EVID / "lprnet_vs_rpv3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", EVID / "lprnet_vs_rpv3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
