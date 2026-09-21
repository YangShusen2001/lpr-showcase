"""Fetch additional real car photos from the HyperLPR repository."""
from pathlib import Path
from urllib.parse import quote
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "assets" / "samples"

RAW = "https://raw.githubusercontent.com/szad670401/HyperLPR/master"
ITEMS = [
    ("resource/images/2.jpg", "hlpr-2.jpg"),
]

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for rel, name in ITEMS:
    dest = SAMP / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP {dest.stat().st_size/1024:8.1f}KB  {name}")
        continue
    for attempt in range(3):
        try:
            with opener.open(f"{RAW}/{quote(rel)}", timeout=120) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"OK   {len(data)/1024:8.1f}KB  {name}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry{attempt} {name}: {e}")
