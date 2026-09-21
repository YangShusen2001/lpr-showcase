"""Decisive checks on the golden image.

Three independent questions:

  1. Is the plate really green?  A blue plate photographed under a green cast
     would fake it. Compare the plate face colour against the car body, which is
     known to be white -- if the body is neutral and the plate is green, the
     plate is genuinely green (and therefore a new-energy plate = 8 characters).

     ⚠️ TWO BUGS in `colour_report()` below -- read the corrected numbers in
     `tools/plate_face_colour.py` / `_evidence/A16-real-dataset-accuracy-20260918.md`
     section 4.5 instead of trusting this function's output:
       (a) `bright = V > percentile(V, 60)` picks the WHITE glyphs on a plate, not
           the plate paint, so the hue statistic describes characters, not colour;
       (b) the "car body" patch at `y1-60:y1-10, x1-200:x1-60` is not the car body --
           on `hlpr-test.jpg` it lands on saturated red/blue scenery, not a neutral.
     The corrected method uses saturated mid-bright plate paint (`S>=90 & 45<=V<=250`)
     with the plate's own white glyphs as the white-balance control. It confirms the
     plate IS green (G-B = +39 with neutral glyphs at +6), so the section-1 conclusion
     survives -- but the "37.2% green hue" figure printed here does not.

  2. What does upstream say?  `hlpr-test.jpg` came from the HyperLPR project.
     Look for the image and any accompanying ground-truth label upstream.

  3. What does the production recogniser say when fed a clean, aspect-preserving
     perspective rectification of the plate (instead of its own internal crop)?
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hlpr_reference as H            # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
TMP = Path(r"C:\Users\26671\AppData\Local\Temp")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept": "application/vnd.github+json"})
    with OPENER.open(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def colour_report(img):
    """Report hue/HSV of the plate face and of the car body."""
    # plate face: detector box
    det = H.sess("y5fu_320x_sim.onnx")
    rows = H.detect(det, img)
    x1, y1, x2, y2 = rows[0][:4].astype(int)
    plate = img[y1:y2, x1:x2]
    # car body: a patch just left of the plate on the bumper/boot
    body = img[y1 - 60:y1 - 10, x1 - 200:x1 - 60]

    def stat(patch, name):
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        Hh, S, V = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
        bright = V > np.percentile(V, 60)
        bgr = patch.reshape(-1, 3).mean(0)
        print(f"  {name:12s} mean BGR = ({bgr[0]:5.1f},{bgr[1]:5.1f},{bgr[2]:5.1f})   "
              f"hue median = {np.median(Hh[bright]):5.1f}   sat median = {np.median(S[bright]):5.1f}   "
              f"val median = {np.median(V[bright]):5.1f}")
        return float(np.median(Hh[bright])), float(np.median(S[bright]))

    print("1) colour check (OpenCV hue 0-179: 60=yellow, 90=cyan-green, 120=blue)")
    hp, sp = stat(plate, "plate face")
    hb, sb = stat(body, "car body")
    print(f"   -> plate hue {hp:.0f} with saturation {sp:.0f};  body hue {hb:.0f} with saturation {sb:.0f}")
    if sp > 60 and 35 <= hp <= 100:
        print("   -> plate is SATURATED GREEN => new-energy plate => 8 characters expected")
    else:
        print("   -> plate is not a saturated green")

    # green fraction inside the plate face
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
    sat = ss > 60
    gf = float(((hh >= 35) & (hh <= 100) & sat).mean())
    bf = float(((hh >= 100) & (hh <= 135) & sat).mean())
    print(f"   green-hue fraction = {gf:.1%}   blue-hue fraction = {bf:.1%}")
    return gf, bf


def upstream():
    print("\n2) upstream provenance of hlpr-test.jpg")
    for repo, branch in (("szad670401/HyperLPR", "master"), ("szad670401/HyperLPR", "main")):
        try:
            tree = api(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")
        except Exception as e:                                     # noqa: BLE001
            print(f"   {repo}@{branch}: {e}")
            continue
        hits = [t["path"] for t in tree.get("tree", [])
                if "hlpr-test" in t["path"].lower() or "hlpr_test" in t["path"].lower()]
        print(f"   {repo}@{branch}: {len(tree.get('tree', []))} entries; hlpr-test hits = {hits}")
        # also look for any label / ground-truth files
        labels = [t["path"] for t in tree.get("tree", [])
                  if any(k in t["path"].lower() for k in ("label", "ground", "truth", "gt_", "test.txt"))]
        print(f"      label-ish files: {labels[:12]}")
        break
    return None


def corrected_recognition(img):
    print("\n3) rpv3 on a clean aspect-preserving rectification of the plate")
    det = H.sess("y5fu_320x_sim.onnx")
    rec = H.sess("rpv3_mdict_160_r3.onnx")
    rows = H.detect(det, img)
    pts = rows[0][5:13].reshape(4, 2).astype(np.float32)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)
    d = lambda a, b: float(np.hypot(*(a - b)))                      # noqa: E731
    Wm = (d(pts[0], pts[1]) + d(pts[3], pts[2])) / 2
    Hm = (d(pts[0], pts[3]) + d(pts[1], pts[2])) / 2

    variants = {}
    # (a) aspect preserving (what the plate really looks like)
    S = 6
    W, Hh = int(Wm * S), int(Hm * S)
    M = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float32))
    variants["aspect-preserving"] = cv2.warpPerspective(img, M, (W, Hh), flags=cv2.INTER_CUBIC)
    # (b) stretched to a canonical 7-char plate aspect 440x140
    M2 = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [440, 0], [440, 140], [0, 140]], np.float32))
    variants["stretched-440x140"] = cv2.warpPerspective(img, M2, (440, 140), flags=cv2.INTER_CUBIC)
    # (c) stretched to the new-energy aspect 480x140
    M3 = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [480, 0], [480, 140], [0, 140]], np.float32))
    variants["stretched-480x140"] = cv2.warpPerspective(img, M3, (480, 140), flags=cv2.INTER_CUBIC)
    # (d) the recogniser's own internal crop, for reference
    variants["rpv3 internal crop"] = H.get_rotate_crop_image(img, pts.astype(int))

    for name, im in variants.items():
        h, w = im.shape[:2]
        data = H.encode_images(im, w * 1.0 / h, tuple(int(v) for v in rec.get_inputs()[0].shape[2:]))
        out = np.asarray(rec.run([rec.get_outputs()[0].name],
                                 {rec.get_inputs()[0].name: np.expand_dims(data, 0)})[0])[0]
        idx = np.argmax(out, axis=1)
        prob = np.max(out, axis=1)
        code, conf = H.ctc_decode(idx, prob)
        nz = [(H.TOKEN[int(k)] if int(k) < len(H.TOKEN) else "?") for k in idx if prob[idx.tolist().index(k)] > 0] if False else None
        steps = " ".join((H.TOKEN[int(k)] if int(k) < len(H.TOKEN) else "?") for k in idx)
        print(f"   {name:22s} {w:4d}x{h:<4d} -> {code!r:14s} len={len(code)}  conf={conf:.3f}")
        print(f"        CTC steps: {steps}")


def main():
    img = H.imread_u(ROOT / "assets" / "samples" / "hlpr-test.jpg")
    print(f"image {img.shape[1]}x{img.shape[0]}\n")
    colour_report(img)
    upstream()
    corrected_recognition(img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
