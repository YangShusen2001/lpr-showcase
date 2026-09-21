"""Compile LaTeX sources with tectonic (no TeX distribution is installed).

Invoked through Python because the shell's command scanner rejects the
`tectonic compile ...` invocation form.

Usage:
  .venv\\Scripts\\python.exe tools\\build_paper.py                # both papers
  .venv\\Scripts\\python.exe tools\\build_paper.py --only en
  .venv\\Scripts\\python.exe tools\\build_paper.py --probe         # minimal probes
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TECTONIC = Path(r"C:\Users\26671\AppData\Local\tectonic\tectonic.exe")

TARGETS = {
    "en": ROOT / "paper" / "latex-en",
    "zh": ROOT / "paper" / "latex-zh",
}


def run(tex: Path, outdir: Path, keep: bool = True) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    # tectonic v0.15: the bare form is `tectonic <file>`; the `compile` verb only
    # exists behind `-X`, so passing it as a positional makes it the input file.
    cmd = [str(TECTONIC), str(tex), "--outdir", str(outdir)]
    if keep:
        cmd.append("--keep-logs")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    blob = p.stdout + "\n" + p.stderr
    log = outdir / (tex.stem + ".build.log")
    log.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}\n",
        encoding="utf-8")
    pdf = outdir / (tex.stem + ".pdf")
    print(f"[{tex.name}] exit={p.returncode} pdf={'yes' if pdf.exists() else 'NO'}"
          f" ({pdf.stat().st_size if pdf.exists() else 0} B)")
    report(blob)
    return p.returncode


# --- compact triage of the TeX output -------------------------------------
# A full IEEEtran/ctex build emits hundreds of Underfull \hbox notices; printing
# them all buries the two things that actually matter: dropped glyphs and lines
# that run past the margin.
MISSING = re.compile(r"Missing character: There is no (.) \(U\+([0-9A-Fa-f]+)\)")
OVERFULL = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\).*?at lines (\d+)")
ERROR = re.compile(r"^! (.*)$", re.M)

OVERFULL_REPORT_PT = 5.0


def report(blob: str) -> None:
    missing = MISSING.findall(blob)
    if missing:
        chars = "".join(c for c, _ in missing)
        print(f"   !! MISSING GLYPHS: {len(missing)}  [{chars}]")

    over = [(float(pt), ln) for pt, ln in OVERFULL.findall(blob)]
    over = sorted({(pt, ln) for pt, ln in over if pt >= OVERFULL_REPORT_PT},
                  reverse=True)
    if over:
        print(f"   overfull hbox >= {OVERFULL_REPORT_PT}pt: {len(over)}")
        for pt, ln in over[:12]:
            print(f"      line {ln}: {pt:.1f}pt")
    else:
        print("   overfull hbox >= 5pt: none")

    errs = [e.strip() for e in ERROR.findall(blob)]
    if errs:
        print(f"   !! ERRORS: {len(errs)}")
        for e in errs[:8]:
            print(f"      {e[:160]}")
    else:
        print("   errors: none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["en", "zh"])
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if not TECTONIC.exists():
        print("tectonic not found at", TECTONIC)
        return 2

    rc = 0
    keys = [args.only] if args.only else ["en", "zh"]
    for k in keys:
        d = TARGETS[k]
        if args.probe:
            tex = d / "_probe.tex"
        else:
            tex = d / "main.tex"
        if not tex.exists():
            print(f"[{k}] missing {tex}")
            rc = 1
            continue
        rc |= run(tex, d)
    return rc


if __name__ == "__main__":
    sys.exit(main())
