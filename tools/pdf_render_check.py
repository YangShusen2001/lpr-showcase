"""Verify that a compiled PDF really contains the expected CJK glyphs.

Motivation: a LaTeX run can *succeed* while silently dropping CJK characters
when they are routed through a Latin-only font (classic case: a Chinese
province character inside \\texttt under IEEEtran).  The compile log warns
"Missing character", but it is easy to miss among hundreds of \\hbox warnings.

This tool answers the question directly:
  1. which PDF backends are available;
  2. whether the expected code points appear in the extracted text;
  3. optionally render page N to a PNG for eyeball inspection.

Usage (always via Python, never the shell):
  python tools/pdf_render_check.py            # check every paper PDF it can find
  python tools/pdf_render_check.py --only en
Writes a JSON report next to each PDF and prints a one-line verdict per file.

Note: the shell on this machine refuses commands containing non-ASCII, so the
target PDF paths are resolved internally from ROOT rather than passed as
arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# province characters used on the sample plates in both papers
DEFAULT_EXPECT = "藏苏粤港津皖蒙冀"


def targets(only: str | None, probes: bool = False) -> list[Path]:
    """Deliverables are main.pdf; the _probe.pdf files are toolchain smoke
    tests that legitimately contain none of the sample glyphs, so they are
    only checked on request."""
    found: list[Path] = []
    keys = [only] if only else ["en", "zh"]
    for k in keys:
        d = ROOT / "paper" / f"latex-{k}"
        p = d / "main.pdf"
        if p.exists():
            found.append(p)
        if probes:
            q = d / "_probe.pdf"
            if q.exists():
                found.append(q)
    return found


def backends() -> dict[str, bool]:
    import importlib.util
    return {m: bool(importlib.util.find_spec(m))
            for m in ("fitz", "pypdf", "PyPDF2", "pdfminer")}


def extract_text(pdf: Path) -> tuple[str, str]:
    """Return (text, backend_used)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf))
        return "\n".join(p.get_text() for p in doc), "fitz"
    except Exception:
        pass
    for mod in ("pypdf", "PyPDF2"):
        try:
            m = __import__(mod)
            r = m.PdfReader(str(pdf))
            return "\n".join((pg.extract_text() or "") for pg in r.pages), mod
        except Exception:
            continue
    try:
        from pdfminer.high_level import extract_text as _et
        return _et(str(pdf)), "pdfminer"
    except Exception:
        pass
    return "", "none"


def render(pdf: Path, page: int, out: Path) -> str | None:
    try:
        import fitz
    except Exception:
        return None
    doc = fitz.open(str(pdf))
    if page < 1 or page > doc.page_count:
        return None
    pix = doc[page - 1].get_pixmap(dpi=200)
    out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out))
    return str(out)


def pdf_pages(pdf: Path) -> int:
    try:
        import fitz
        return fitz.open(str(pdf)).page_count
    except Exception:
        return 0


def page_with(pdf: Path, chars: str) -> int:
    """1-based number of the page holding the most of `chars`, else 1."""
    try:
        import fitz
    except Exception:
        return 1
    doc = fitz.open(str(pdf))
    best, best_n = 1, -1
    for i, p in enumerate(doc, start=1):
        t = p.get_text()
        n = sum(1 for c in chars if c in t)
        if n > best_n:
            best, best_n = i, n
    return best


def check_one(pdf: Path, expect: str, render_page: int, outdir: Path | None,
              find: str = "") -> dict:
    outdir = outdir or pdf.parent
    outdir.mkdir(parents=True, exist_ok=True)

    text, used = extract_text(pdf)
    want = [c for c in expect]
    present = [c for c in want if c in text]
    absent = [c for c in want if c not in text]

    report = {
        "pdf": str(pdf),
        "size_bytes": pdf.stat().st_size,
        "pages": pdf_pages(pdf),
        "text_backend": used,
        "text_chars": len(text),
        "expected": want,
        "present": present,
        "absent": absent,
        "verdict": ("PASS" if want and not absent else
                    "NO_EXPECTATION" if not want else "FAIL"),
    }
    if render_page:
        # 0 means "auto": the page where the expected glyphs actually live
        pg = page_with(pdf, expect) if render_page < 0 else render_page
        report["rendered_page"] = pg
        report["rendered_png"] = render(
            pdf, pg, outdir / f"{pdf.stem}_p{pg}.png")

    if find:
        hits = [ln.strip() for ln in text.splitlines() if find in ln]
        report["find"] = find
        report["find_hits"] = hits

    rep = outdir / f"{pdf.stem}.glyphcheck.json"
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    extra = f" page={report.get('rendered_page')}" if render_page else ""
    print(f"[{pdf.name}] {report['verdict']} pages={report['pages']} "
          f"backend={used} chars={len(text)} absent={''.join(absent) or '-'}"
          f"{extra} -> {rep.name}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["en", "zh"])
    ap.add_argument("--expect", default=DEFAULT_EXPECT,
                    help="characters that must be present")
    ap.add_argument("--render", type=int, default=0,
                    help="page to render; 0 = none, -1 = the page holding the glyphs")
    ap.add_argument("--find", default="",
                    help="ASCII needle; matching text lines are recorded in the report")
    ap.add_argument("--probes", action="store_true",
                    help="also check the minimal _probe.pdf toolchain smoke tests")
    args = ap.parse_args()

    pdfs = targets(args.only, args.probes)
    if not pdfs:
        print("no PDFs found")
        return 2

    print("backends:", backends())
    worst = 0
    for pdf in pdfs:
        r = check_one(pdf, args.expect, args.render, None, args.find)
        if r["verdict"] == "FAIL":
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
