"""Summarise every HarmonyOS .p7b provisioning profile in a directory.

Prints, per profile: bundle-name, app-identifier, type, uuid, validity window
(human readable), and the UDIDs the profile is bound to.

Usage:  python summarize_profiles.py <out.txt> <dir-with-p7b>
"""
import datetime as dt
import json
import os
import re
import sys

FIELDS = ("bundle-name", "app-identifier", "type", "uuid", "developer-id", "apl")


def extract_json(blob: bytes):
    text = blob.decode("utf-8", "replace")
    for match in re.finditer(r'\{"version-name"', text):
        start = match.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception as exc:  # pragma: no cover
                        print(f"  json parse failed: {exc}")
                        break
    return None


def ts(value):
    try:
        return dt.datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return repr(value)


def main() -> int:
    out_path, src_dir = sys.argv[1], sys.argv[2]
    lines = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".p7b"):
            continue
        path = os.path.join(src_dir, name)
        with open(path, "rb") as fh:
            blob = fh.read()
        doc = extract_json(blob)
        lines.append("-" * 78)
        lines.append(f"FILE : {name}")
        if not doc:
            lines.append("  !! could not extract profile JSON")
            continue
        info = doc.get("bundle-info", {})
        for key in FIELDS:
            if key in doc:
                lines.append(f"  {key:<16} = {doc[key]}")
            elif key in info:
                lines.append(f"  {key:<16} = {info[key]}")
        validity = doc.get("validity", {})
        if validity:
            lines.append(
                f"  validity         = {ts(validity.get('not-before'))} -> {ts(validity.get('not-after'))}"
            )
        dbg = doc.get("debug-info", {})
        for udid in dbg.get("device-ids", []):
            lines.append(f"  udid             = {udid}")
        if not dbg:
            lines.append("  debug-info       = <none> (release profile, not device bound)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
