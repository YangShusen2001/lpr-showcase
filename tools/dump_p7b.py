"""Dump the printable text inside a HarmonyOS .p7b profile so we can see its
structure (the embedded JSON is usually stored verbatim inside the PKCS#7).

Usage:  python dump_p7b.py <out.txt> <file.p7b> [...]
"""
import re
import sys


def main() -> int:
    out_path = sys.argv[1]
    lines = []
    for path in sys.argv[2:]:
        with open(path, "rb") as fh:
            blob = fh.read()
        lines.append("=" * 78)
        lines.append(f"{path}  ({len(blob)} bytes)")
        lines.append(f"  head: {blob[:32].hex(' ')}")
        # printable runs of >= 8 chars
        text = blob.decode("latin-1")
        runs = re.findall(r"[\x20-\x7e\t\r\n]{8,}", text)
        lines.append(f"  printable runs: {len(runs)}")
        for run in runs:
            lines.append("  | " + run.replace("\n", "\\n")[:2000])
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
