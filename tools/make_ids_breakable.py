"""Make long typewriter identifiers breakable so they stop overflowing margins.

Why this is needed
------------------
The TeX logs show the real defect behind the worst overfull hboxes:

    Overfull \\hbox (146.87732pt too wide) in paragraph at lines 538--543
    []... 用 converter_lite --fp16=on 转换的模型，... 其声明 dtype 为
    \\TU/lmtt/m/n/10.95 OH_AI_DATATYPE_NUMBERTYPE_FLOAT32

`OH_AI_DATATYPE_NUMBERTYPE_FLOAT32` is 33 characters of monospace (~217pt at
10.95pt).  Because xeCJK welds a CJK run to the following Latin run (ctex sets
CJKspace=true, so the source space between them is dropped), TeX sees one long
unbreakable unit and pushes it 5.2cm past the right margin instead of breaking
before it.

Fix: route identifier-like strings through \\path (the `url` package), which
permits a line break after `_ . / = -`.  These strings are only ever used in
running text and table cells -- never in a caption or section title -- so the
familiar fragility of \\path in moving arguments does not apply here.

The script prints every \\texttt body it did NOT convert, so anything unusual
can be reviewed instead of silently skipped.

Usage:  python tools/make_ids_breakable.py [--write]
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "paper" / "latex-en" / "main.tex",
    ROOT / "paper" / "latex-zh" / "main.tex",
]

TT = re.compile(r"\\texttt\{([^{}]*)\}")
# Anything made only of these characters, once \_ is un-escaped, is a
# machine identifier / file path and is safe to hand to \path.
IDENT_CHARS = re.compile(r"^[A-Za-z0-9_.\-=*/]+$")
# Strings with no internal punctuation at all (MU4158, resizeLinear) are short
# enough to fit and stay as \texttt; a long slash-path is not, so allow those.
LONG_PLAIN = 24
MIN_LEN = 12

# identifiers that carry a space (a command line, or a version number) and so
# cannot be handled by the generic rule
SPACED = [
    (r"\texttt{converter\_lite --fp16=on}",
     r"\path{converter_lite} \path{--fp16=on}"),
    (r"\texttt{converter\_lite 2.6}", r"\path{converter_lite} 2.6"),
]


def is_ident(body: str) -> str | None:
    """Return the \\path-ready form of `body`, or None to leave it alone."""
    if not body or any(c in body for c in " \t{}(),<>$%#&'\"~^"):
        return None
    plain = body.replace(r"\_", "_")
    if not IDENT_CHARS.match(plain):
        return None
    if len(plain) < MIN_LEN:
        return None
    has_punct = any(c in plain for c in "_.-/=")
    if not has_punct and len(plain) < LONG_PLAIN:
        return None
    return plain


def convert(text: str) -> tuple[str, list[str], list[str]]:
    converted: list[str] = []
    skipped: list[str] = []

    for old, new in SPACED:
        if old in text:
            text = text.replace(old, new)
            converted.append(new)

    def sub(m: re.Match) -> str:
        body = m.group(1)
        plain = is_ident(body)
        if plain is not None:
            converted.append(r"\path{" + plain + "}")
            return r"\path{" + plain + "}"
        skipped.append(body)
        return m.group(0)

    return TT.sub(sub, text), converted, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for f in FILES:
        src = f.read_text(encoding="utf-8")
        out, conv, skip = convert(src)
        print(f"\n=== {f.relative_to(ROOT)} ===")
        print(f"  converted {len(conv)} identifiers")
        for c in conv:
            print(f"    -> {c}")
        if skip:
            print(f"  left as \\texttt{{}} ({len(skip)}):")
            for s in skip:
                print(f"    - {s}")
        if args.write and out != src:
            bak = f.with_suffix(f".tex.bak-{stamp}")
            shutil.copy2(f, bak)
            f.write_text(out, encoding="utf-8")
            print(f"  WRITTEN (backup {bak.name})")
        elif out == src:
            print("  no change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
