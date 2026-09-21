"""Fetch the real Chinese-plate test images shipped in the HyperLPR repository.

Source: https://github.com/szad670401/HyperLPR (Apache-2.0), resource/images/.
Used here as *real* verification inputs for the detection + recognition models.

Proxies are bypassed explicitly: the local Clash tunnel returns 502 for these paths.
"""
from pathlib import Path
from urllib.parse import quote
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "assets" / "samples"
SAMP.mkdir(parents=True, exist_ok=True)

RAW = "https://raw.githubusercontent.com/szad670401/HyperLPR/master"

ITEMS = [
    ("resource/images/1.jpg", "hlpr-1.jpg"),
    ("resource/images/test_img.jpg", "hlpr-test.jpg"),
    ("resource/images/rec_crop/_0_津B6H920.jpg", "crop-0-津B6H920.jpg"),
    ("resource/images/rec_crop/_1_皖KD01833.jpg", "crop-1-皖KD01833.jpg"),
    ("resource/images/rec_crop/_6_蒙B023H6.jpg", "crop-6-蒙B023H6.jpg"),
    ("resource/images/rec_crop/_8_冀D5L690.jpg", "crop-8-冀D5L690.jpg"),
]

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

for rel, name in ITEMS:
    dest = SAMP / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP {dest.stat().st_size/1024:8.1f}KB  {name}")
        continue
    url = f"{RAW}/{quote(rel)}"
    ok = False
    for attempt in range(3):
        try:
            with opener.open(url, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"OK   {len(data)/1024:8.1f}KB  {name}")
            ok = True
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry{attempt} {name}: {e}")
    if not ok:
        print(f"FAIL {name}")
