"""List candidate test images in the upstream HyperLPR repository.

Keeps the demo's provenance consistent: the ONNX models already come from
szad670401/HyperLPR, so the test photos should too.

Run:  python tools/list_upstream.py
"""
from __future__ import annotations

import json
import urllib.request

API = "https://api.github.com/repos/szad670401/HyperLPR/git/trees/master?recursive=1"

# The local Clash proxy answers 502 for some raw.githubusercontent paths; go direct.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

with opener.open(API, timeout=60) as r:
    tree = json.load(r)

IMGS = (".jpg", ".jpeg", ".png", ".webp")
hits = [n["path"] for n in tree["tree"]
        if n["type"] == "blob" and n["path"].lower().endswith(IMGS)]

print(f"total blobs: {len(tree['tree'])}, images: {len(hits)}")
for p in hits:
    print(" ", p)
