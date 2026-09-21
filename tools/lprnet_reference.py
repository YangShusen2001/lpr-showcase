"""LPRNet reference: preprocessing + CTC decode, and an *empirical* dictionary check.

Why this exists
---------------
LPRNet is the candidate recogniser the project's CANN-friendliness criterion scores
highest (81.2 % NPU utilisation upper bound, ADR-014). It has been converted
(`om_lprnet.om`) and really runs on the device (`build_rc=0 / run_rc=0`), but it has
**never** been scored for accuracy: it is a CTC network whose character set is not
the project's 77-entry table, so "it runs" says nothing about "it reads correctly".
A13 §5 states this red line explicitly: 「只测了能跑，没测准」.

This script closes that gap, device-free, against the four ground-truth crops in
assets/samples/ whose filenames carry the expected string.

How the dictionary was recovered (not assumed)
----------------------------------------------
A first pass with a *guessed* 77-style dictionary missed 4/4, but the raw argmax
indices showed structure, and that structure is what the dictionary below encodes:

  * index 67 sits BETWEEN characters in every sample -> 67 is the CTC **blank**,
    so the model's character set is 0..66 = **67 characters** (matching ADR-007's
    "CTC 67 字符字典"). The blank is at the END, not at 0.
  * digits: 0->31, 1->32, 2->33, 3->34, 5->36, 6->37, 8->39, 9->40
    i.e. **index = 31 + digit**.
  * letters: B->42, D->44, H->48, K->50  =>  A=41, and the alphabet skips I and O
    (24 letters, the standard Chinese-plate letter set) => 41..64.
  * provinces: 津->2, 蒙->6, 皖->12 => 京1 津2 冀3 晋4 辽5 蒙6 吉7 黑8 沪9 苏10 浙11
    皖12 ... consistent with the standard plate-province order, i.e. 1..30.

Indices 0, 65 and 66 were never observed in the four samples, so they are left
unresolved rather than guessed.

Run:  ./.venv/Scripts/python.exe tools/lprnet_reference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "samples"
EVID = ROOT / "_evidence"
EVID.mkdir(exist_ok=True)

ONNX = Path(r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet.onnx")

# --- recovered dictionary (see module docstring for the evidence) -------------
PROVINCES = "京津冀晋辽蒙吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁"   # idx 1..30
DIGITS = "0123456789"                                                     # idx 31..40
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"                                      # idx 41..64, no I/O
BLANK = 67

CHARS = [None] * 68
for i, ch in enumerate(PROVINCES, start=1):
    CHARS[i] = ch
for i, ch in enumerate(DIGITS, start=31):
    CHARS[i] = ch
for i, ch in enumerate(LETTERS, start=41):
    CHARS[i] = ch
# CHARS[0], CHARS[65], CHARS[66] stay None: unobserved in the labelled samples.
CHARS[BLANK] = ""   # blank


def load():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])


def imread_u(path: Path):
    """cv2.imread cannot open non-ASCII paths on Windows; decode from a buffer."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def preprocess(img, mode: str):
    """LPRNet wants (1,3,24,94) NCHW. `mode` selects the normalisation convention."""
    h, w = 24, 94
    x = cv2.resize(img, (w, h)).astype(np.float32)
    if mode == "pm127.5_div128":     # (x - 127.5) / 128   -- upstream LPRNet
        x = (x - 127.5) / 128.0
    elif mode == "pm127.5":          # (x - 127.5) / 127.5 -- project convention
        x = (x - 127.5) / 127.5
    elif mode == "div255":           # x / 255
        x = x / 255.0
    else:
        raise ValueError(mode)
    return x.transpose(2, 0, 1)[None, ...]


def ctc_greedy(idx_row):
    """Blank is index 67 and sits at the END of the alphabet."""
    out = []
    for i, k in enumerate(idx_row):
        if k == BLANK:
            continue
        if i > 0 and idx_row[i - 1] == k:
            continue
        out.append(CHARS[k] if CHARS[k] else f"<{k}>")
    return "".join(out)


def main():
    sess = load()
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    crops = sorted(SAMPLES.glob("crop-*.jpg"))
    report = {"model": str(ONNX), "blank": BLANK,
              "provinces": PROVINCES, "digits": DIGITS, "letters": LETTERS,
              "runs": []}

    for mode in ("pm127.5_div128", "pm127.5", "div255"):
        print(f"\n=== preprocess = {mode} ===")
        for p in crops:
            gt = p.stem.split("-", 2)[-1]
            img = imread_u(p)
            out = np.asarray(sess.run([oname], {iname: preprocess(img, mode)})[0])[0]
            idx = np.argmax(out, axis=0)
            code = ctc_greedy(idx)
            print(f"  {p.name:26s} gt={gt:10s} pred={code:10s} "
                  f"{'OK' if code == gt else 'MISS'}")
            report["runs"].append({"mode": mode, "file": p.name, "gt": gt,
                                   "pred": code, "match": code == gt,
                                   "raw_idx": [int(v) for v in idx]})

    # Cross-check the recovered alphabet against the data: for every labelled crop,
    # pair the collapsed non-blank indices with the ground-truth characters.
    print("\n=== recovered alphabet vs ground truth ===")
    pairs: dict[int, set] = {}
    for r in report["runs"]:
        if r["mode"] != "pm127.5_div128":
            continue
        seq = []
        for i, k in enumerate(r["raw_idx"]):
            if k == BLANK:
                continue
            if i > 0 and r["raw_idx"][i - 1] == k:
                continue
            seq.append(k)
        for k, ch in zip(seq, r["gt"]):
            pairs.setdefault(k, set()).add(ch)
    conflicts = 0
    for k in sorted(pairs):
        guess = CHARS[k] or f"<{k}>"
        ok = sorted(pairs[k]) == [guess]
        if not ok:
            conflicts += 1
        print(f"  idx={k:3d}  observed={''.join(sorted(pairs[k])):12s} "
              f"recovered='{guess}'  {'consistent' if ok else 'CONFLICT'}")
    report["alphabet_conflicts"] = conflicts

    (EVID / "lprnet_reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", EVID / "lprnet_reference.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
