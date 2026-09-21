"""Diff the browser port against the Python reference, field by field.

Reads _evidence/web__verify_html.json (written by tools/web_check.mjs) and
_evidence/det_dump.json (written by tools/dump_det.py) and prints a comparison
table, flagging every numeric deviation so fidelity claims stay honest.

Run:  python tools/compare_dumps.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "_evidence"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def rel(a: float, b: float) -> str:
    if a == b:
        return "exact"
    d = abs(a - b)
    scale = max(abs(a), abs(b), 1e-12)
    return f"Δabs={d:.3e}  Δrel={d / scale:.2e}"


def compare_vec(label: str, pa, pb) -> list[str]:
    rows = []
    if len(pa) != len(pb):
        rows.append(f"  {label}: LENGTH MISMATCH py={len(pa)} js={len(pb)}")
        return rows
    worst = 0.0
    worst_i = -1
    for i, (a, b) in enumerate(zip(pa, pb)):
        d = abs(float(a) - float(b))
        if d > worst:
            worst, worst_i = d, i
    tag = "exact" if worst == 0 else f"max Δabs={worst:.3e} @[{worst_i}]"
    rows.append(f"  {label}: {tag}")
    if worst_i >= 0 and worst > 0:
        rows.append(f"      py={pa[worst_i]!r}  js={pb[worst_i]!r}")
    return rows


def main() -> None:
    web = load(EVID / "web__verify_html.json")["result"]
    py = load(EVID / "det_dump.json")

    js_by_name = {im["name"]: im for im in web["images"]}
    py_by_name = {im["name"]: im for im in py["images"]}

    print(f"Python onnxruntime {py['ort_version']}  vs  onnxruntime-web (browser)\n")
    lines: list[str] = []

    for name, pi in py_by_name.items():
        ji = js_by_name.get(name)
        lines.append(f"=== {name} ===")
        if ji is None:
            lines.append("  MISSING on the JS side")
            continue

        pd, jd = pi["detect"], ji["detect"]
        lines.append(f"  letterbox: py r={pd['r']!r} left={pd['left']} top={pd['top']}"
                     f"  |  js r={jd['r']!r} left={jd['left']} top={jd['top']}")
        lines += compare_vec("input tensor sum", [pd["input"]["sum"]], [jd["input"]["sum"]])
        lines += compare_vec("input tensor head", pd["input"]["head"], jd["input"]["head"])
        lines += compare_vec("input tensor mid", pd["input"]["mid"], jd["input"]["mid"])
        lines += compare_vec("detector raw row", pd["topRawRow"], jd["topRawRow"])

        if not pi["plates"] or not ji["plates"]:
            lines.append(f"  plates: py={len(pi['plates'])} js={len(ji['plates'])}"
                         f"  {'OK (both empty)' if not pi['plates'] and not ji['plates'] else 'DIFF'}")
            lines.append("")
            continue

        p0 = pi["plates"][0]
        j0 = ji["plates"][0]
        lines.append(f"  rect     : py={p0['rect']}  js={j0['rect']}"
                     f"  {'OK' if p0['rect'] == j0['rect'] else 'DIFF'}")
        lines.append(f"  marks    : py={p0['marks']}  js={j0['marks']}"
                     f"  {'OK' if p0['marks'] == j0['marks'] else 'DIFF'}")
        lines.append(f"  cropShape: py={p0['crop_shape']}  js={j0['cropShape']}"
                     f"  {'OK' if p0['crop_shape'] == j0['cropShape'] else 'DIFF'}")
        lines.append(f"  cropSum  : py={pi['crops'][0]['sum']}  js={j0['cropSum']}"
                     f"  {'OK' if pi['crops'][0]['sum'] == j0['cropSum'] else 'DIFF'}")
        lines.append(f"  code     : py={p0['code']!r}  js={j0['code']!r}"
                     f"  {'OK' if p0['code'] == j0['code'] else 'DIFF'}")
        lines.append(f"  recConf  : {rel(p0['rec_conf'], j0['recConf'])}")
        lines.append(f"  detScore : {rel(p0['det_score'], j0['detScore'])}")
        lines += compare_vec("cls", p0["cls"], j0["cls"])
        lines.append("")

    # recogniser-only agreement on the labelled crops
    lines.append("=== recogniser on labelled crops ===")
    for ji in web["images"]:
        if "gt" in ji:
            lines.append(f"  {ji['gt']:12s} py=— js={ji['code']:12s} conf={ji['conf']} "
                         f"match={ji['match']}")

    text = "\n".join(lines)
    print(text)
    (EVID / "fidelity_report.txt").write_text(text, encoding="utf-8")
    print("\nwrote", EVID / "fidelity_report.txt")


if __name__ == "__main__":
    main()
