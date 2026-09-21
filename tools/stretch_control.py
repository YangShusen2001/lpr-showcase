"""Control experiment: can horizontal stretching INVENT a character?

Finding under test: the golden plate reads 7 characters when left at its observed
(yaw-compressed) aspect, but 8 characters -- `苏ED51712` -- when stretched to a
canonical plate aspect. Before concluding the plate really has 8 characters, rule
out the trivial explanation that stretching makes the recogniser hallucinate an
extra glyph.

Control: take known 7-character plates from the third-party set, stretch each
horizontally by the same factor, and count how often the recogniser's output grows
from 7 to 8 characters. If stretching never invents characters on plates whose
truth is known to be 7, then the golden plate's 8-character reading is real.

Also measures the reverse: how often a *compressed* view LOSES a character, by
squeezing known 7-character plates the same way the golden plate is squeezed.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CROPS = ROOT / "_dataset" / "real" / "crops"
OUT = ROOT / "_dataset" / "real" / "stretch_control.json"

GOLDEN_ASPECT = 123.5 / 77.3          # the golden plate's observed (compressed) aspect
CANONICAL_7 = 440 / 140               # blue plate
CANONICAL_8 = 480 / 140               # green new-energy plate
N = 200


def rec(rec_sess, img):
    h, w = img.shape[:2]
    data = H.encode_images(img, w * 1.0 / h,
                           tuple(int(v) for v in rec_sess.get_inputs()[0].shape[2:]))
    out = np.asarray(rec_sess.run([rec_sess.get_outputs()[0].name],
                                  {rec_sess.get_inputs()[0].name: np.expand_dims(data, 0)})[0])[0]
    idx = np.argmax(out, axis=1)
    return H.ctc_decode(idx, np.max(out, axis=1))


def stretch_to(img, target_aspect):
    """Resample so the image has the given width/height aspect, preserving height."""
    h, w = img.shape[:2]
    nw = max(8, int(round(h * target_aspect)))
    return cv2.resize(img, (nw, h), interpolation=cv2.INTER_CUBIC)


def main():
    files = sorted(CROPS.glob("*.jpg"))
    random.seed(0)
    random.shuffle(files)
    files = [f for f in files if len(f.stem) == 7][:N]
    print(f"control set: {len(files)} known 7-character plates")

    rec_sess = H.sess("rpv3_mdict_160_r3.onnx")

    stats = {"orig": Counter(), "stretched7": Counter(), "stretched8": Counter(),
             "compressed": Counter()}
    examples = []
    for f in files:
        gt = f.stem
        img = H.imread_u(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        variants = {
            "orig": img,
            "stretched7": stretch_to(img, CANONICAL_7),
            "stretched8": stretch_to(img, CANONICAL_8),
            "compressed": stretch_to(img, GOLDEN_ASPECT),
        }
        got = {}
        for k, im in variants.items():
            code, conf = rec(rec_sess, im)
            got[k] = (code, conf)
            stats[k][len(code)] += 1
        if len(got["stretched7"][0]) != 7 or len(got["compressed"][0]) != 7:
            if len(examples) < 25:
                examples.append({
                    "file": f.name, "gt": gt,
                    **{k: {"pred": v[0], "len": len(v[0]), "conf": round(v[1], 3)}
                       for k, v in got.items()},
                })

    print("\n=== output LENGTH distribution (truth is 7 for every image) ===")
    for k in ("orig", "stretched7", "stretched8", "compressed"):
        c = stats[k]
        n = sum(c.values())
        print(f"  {k:12s} n={n:4d}  lengths={dict(sorted(c.items()))}")
        inv = c.get(8, 0) + c.get(9, 0)
        lost = c.get(6, 0) + c.get(5, 0)
        print(f"               invented a character (len>7): {inv} ({inv / max(1, n):.1%})"
              f"   lost a character (len<7): {lost} ({lost / max(1, n):.1%})")

    print("\n=== images whose length changed ===")
    for e in examples:
        print(f"  {e['file']:16s} gt={e['gt']:9s} "
              f"orig={e['orig']['pred']:9s}({e['orig']['len']}) "
              f"str7={e['stretched7']['pred']:9s}({e['stretched7']['len']}) "
              f"str8={e['stretched8']['pred']:9s}({e['stretched8']['len']}) "
              f"comp={e['compressed']['pred']:9s}({e['compressed']['len']})")

    OUT.write_text(json.dumps({"stats": {k: dict(v) for k, v in stats.items()},
                               "examples": examples}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("\nCONCLUSION: if 'stretched7' invents a character in ~0% of cases, then horizontal")
    print("stretching does NOT hallucinate glyphs, and the golden plate's 8-character reading")
    print("is a real character that the compressed view hides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
