"""Extract the JSON payload embedded in HarmonyOS .p7b provisioning profiles.

A HarmonyOS debug/release profile is a PKCS#7 (DER) container whose content is a
plain JSON document.  We do not need a full ASN.1 parser: the JSON is stored
verbatim, so we locate the outermost {...} run of printable text and decode it.

Usage:  python read_profile.py <out.txt> <profile.p7b> [<profile.p7b> ...]
"""
import json
import re
import sys

KEYS = (
    "bundle-name",
    "uuid",
    "validity",
    "app-identifier",
    "type",
    "issuer",
    "subject",
    "not-before",
    "not-after",
    "development-certificate",
    "distribution-certificate",
    "apl",
    "version-name",
    "version-code",
)


def extract_json(blob: bytes):
    """Return the first balanced {...} JSON object found in blob."""
    text = blob.decode("utf-8", "replace")
    for start in (m.start() for m in re.finditer(r"\{", text)):
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
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def walk(obj, path=""):
    """Yield (path, value) for every scalar in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def main() -> int:
    out_path = sys.argv[1]
    lines = []
    for profile in sys.argv[2:]:
        with open(profile, "rb") as fh:
            blob = fh.read()
        doc = extract_json(blob)
        lines.append("=" * 78)
        lines.append(f"PROFILE: {profile}  ({len(blob)} bytes)")
        if doc is None:
            lines.append("  !! no embedded JSON found")
            continue
        for path, value in walk(doc):
            leaf = path.split(".")[-1].split("[")[0]
            if leaf in KEYS or "device" in path.lower() or "udid" in path.lower():
                lines.append(f"  {path} = {value!r}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
