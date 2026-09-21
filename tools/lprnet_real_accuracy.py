"""Head-to-head accuracy: LPRNet vs the production recogniser (rpv3/SVTR).

Input: real, third-party, human-annotated plate crops downloaded from
sirius-ai/LPRNet_Pytorch `data/test/` -- the filenames ARE the ground truth, so
there is zero annotation error and no synthesised data in the number.

Dictionary: the official LPRNet `CHARS` list (data/load_data.py), 68 entries,
blank = 67.  Preprocessing is the official recipe: stretch to 94x24, BGR,
(x - 127.5) / 128.

Output: per-sample verdicts + a confusion summary, written to
_evidence/lprnet_accuracy_<date>.md so the numbers can be cited from the paper.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402
import lprnet_reference as L          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CROPS = ROOT / "_dataset" / "real" / "crops"
OUT = ROOT / "_evidence"


def gt_of(p: Path) -> str:
    return p.stem.split("-")[0].split("_")[0]


def main():
    files = sorted(CROPS.glob("*.jpg"))
    if not files:
        print("no crops in", CROPS)
        return 1
    net = L.load()
    iname, oname = net.get_inputs()[0].name, net.get_outputs()[0].name

    rec = H.sess("rpv3_mdict_160_r3.onnx")

    l_ok = r_ok = 0
    rows = []
    t0 = time.time()
    for p in files:
        gt = gt_of(p)
        img = H.imread_u(p)
        # --- LPRNet, official recipe ---
        x = cv2.resize(img, (94, 24)).astype(np.float32)
        x = (x - 127.5) * 0.0078125
        x = x.transpose(2, 0, 1)[None, ...]
        out = np.asarray(net.run([oname], {iname: x})[0])[0]
        lp = L.ctc_greedy(np.argmax(out, axis=0))
        # --- production recogniser ---
        rp = H.recognize(rec, img)[0]
        l_hit, r_hit = (lp == gt), (rp == gt)
        l_ok += l_hit
        r_ok += r_hit
        rows.append((p.name, gt, lp, l_hit, rp, r_hit))

    dt = time.time() - t0
    n = len(rows)
    print(f"n={n}  elapsed={dt:.1f}s")
    print(f"  LPRNet  : {l_ok}/{n} = {l_ok / n:.1%}")
    print(f"  rpv3    : {r_ok}/{n} = {r_ok / n:.1%}")
    print()
    print("--- disagreements (LPRNet wrong) ---")
    for name, gt, lp, lh, rp, rh in rows:
        if not lh:
            mark = "rpv3 OK" if rh else "both wrong"
            print(f"  {name:22s} gt={gt:9s} lpr={lp:9s} rpv3={rp:9s} [{mark}]")
    print()
    print("--- disagreements (rpv3 wrong) ---")
    for name, gt, lp, lh, rp, rh in rows:
        if not rh:
            mark = "LPRNet OK" if lh else "both wrong"
            print(f"  {name:22s} gt={gt:9s} lpr={lp:9s} rpv3={rp:9s} [{mark}]")

    # ---- paired significance (McNemar) -------------------------------
    # With a few dozen samples a raw accuracy gap is not evidence; only the
    # discordant pairs carry information about which model is better.
    b = sum(1 for r in rows if r[3] and not r[5])   # LPRNet right, rpv3 wrong
    c = sum(1 for r in rows if r[5] and not r[3])   # rpv3 right, LPRNet wrong
    disc = b + c
    if disc:
        # exact binomial two-sided p under H0: p=0.5
        from math import comb
        k = min(b, c)
        pval = sum(comb(disc, i) for i in range(0, k + 1)) * 2 / (2 ** disc)
        pval = min(1.0, pval)
    else:
        pval = 1.0
    print()
    print(f"paired: LPRNet-only-correct={b}  rpv3-only-correct={c}  "
          f"discordant={disc}  McNemar exact p={pval:.4f}")

    # ---- per-position error profile ----------------------------------
    # A recogniser that loses the province (position 0) has a different failure
    # mode from one that inserts/drops a digit: the province slot is the part
    # the project's 77-token dictionary exists to get right.
    print()
    print("--- LPRNet error profile by character position ---")
    pos_err = {}
    for name, gt, lp, lh, rp, rh in rows:
        if lh or len(lp) != len(gt):
            continue
        for i, (a, b_) in enumerate(zip(gt, lp)):
            if a != b_:
                pos_err[i] = pos_err.get(i, 0) + 1
    for i in sorted(pos_err):
        print(f"  position {i} (0=province): {pos_err[i]} errors")

    OUT.mkdir(exist_ok=True)
    dst = OUT / "lprnet_accuracy_20260918.md"
    with dst.open("w", encoding="utf-8") as f:
        f.write("# LPRNet accuracy acceptance on real third-party data\n\n")
        f.write("- Source: `sirius-ai/LPRNet_Pytorch` `data/test/` -- real photographs,\n")
        f.write("  filenames are the human ground truth, no synthetic data.\n")
        f.write("- This is LPRNet's *own* test split: the comparison is biased in\n")
        f.write("  LPRNet's favour, and rpv3 was never trained on it.\n")
        f.write(f"- n = {n} crops\n")
        f.write("- LPRNet preprocessing: official recipe (stretch 94x24, BGR, (x-127.5)/128)\n")
        f.write("- LPRNet dictionary: official `CHARS` from the upstream repo (68 entries, blank=67)\n\n")
        f.write("| recogniser | correct | accuracy |\n|---|---|---|\n")
        f.write(f"| LPRNet | {l_ok}/{n} | {l_ok / n:.1%} |\n")
        f.write(f"| rpv3 (production) | {r_ok}/{n} | {r_ok / n:.1%} |\n\n")
        f.write(f"Paired McNemar: LPRNet-only-correct = {b}, rpv3-only-correct = {c}, "
                f"exact p = {pval:.4f}\n\n")
        f.write("## LPRNet errors\n\n| file | gt | LPRNet | rpv3 |\n|---|---|---|---|\n")
        for name, gt, lp, lh, rp, rh in rows:
            if not lh:
                f.write(f"| {name} | {gt} | {lp} | {rp} |\n")
        f.write("\n## rpv3 errors\n\n| file | gt | LPRNet | rpv3 |\n|---|---|---|---|\n")
        for name, gt, lp, lh, rp, rh in rows:
            if not rh:
                f.write(f"| {name} | {gt} | {lp} | {rp} |\n")
    print(f"\nwritten {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
