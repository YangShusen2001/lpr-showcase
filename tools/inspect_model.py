"""Inspect the downloaded ONNX license-plate detectors: IO spec + a real inference run.

Run with the system Python that already has onnxruntime + numpy + PIL + cv2.
"""
import sys
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"
OUT = ROOT / "_evidence"
OUT.mkdir(exist_ok=True)

report = {}


def describe(path: Path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 1024 / 1024, 2), "inputs": [], "outputs": []}
    for i in sess.get_inputs():
        info["inputs"].append({"name": i.name, "shape": i.shape, "type": i.type})
    for o in sess.get_outputs():
        info["outputs"].append({"name": o.name, "shape": o.shape, "type": o.type})
    return sess, info


def letterbox(img: np.ndarray, new_shape: int, color=(114, 114, 114)):
    """Ultralytics-style letterbox to a square of side new_shape."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def run(sess, img_path: Path, size: int, tag: str):
    img = np.array(Image.open(img_path).convert("RGB"))
    lb, r, dw, dh = letterbox(img, size)
    x = lb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    iname = sess.get_inputs()[0].name
    outs = sess.run(None, {iname: x})
    res = {"image": img_path.name, "model": tag, "input_size": size,
           "out_shapes": [list(o.shape) for o in outs],
           "out_sample": [np.asarray(o).reshape(-1)[:16].tolist() for o in outs]}

    # end2end row layout (open_image_models yolo_v9 postprocess, verified 2026-09-17):
    #   [batch_idx, x1, y1, x2, y2, class_id, score]
    arr = np.asarray(outs[0])
    rows = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(1, -1)
    dets = []
    for row in rows:
        if row.shape[0] < 7:
            continue
        score = float(row[6])
        if score < 0.25:
            continue
        x1, y1, x2, y2 = [float(v) for v in row[1:5]]
        dets.append({"box": [x1, y1, x2, y2], "conf": score, "cls": float(row[5])})
    res["num_dets_gt025"] = len(dets)
    res["dets"] = dets[:10]

    pil = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(pil)
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        # model space -> original space
        x1 = (x1 - dw) / r
        x2 = (x2 - dw) / r
        y1 = (y1 - dh) / r
        y2 = (y2 - dh) / r
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        d.text((x1 + 2, max(0, y1 - 12)), f"{det['conf']:.2f}", fill=(255, 0, 0))
    dst = OUT / f"probe_{img_path.stem}_{tag}.png"
    pil.save(dst)
    res["annotated"] = dst.name
    return res


def main():
    imgs = sorted((ROOT / "assets" / "samples").glob("*.jpg"))
    for m in sorted(MODELS.glob("*.onnx")):
        sess, info = describe(m)
        size = int(m.stem.split("-")[3])
        info["runs"] = []
        for ip in imgs:
            info["runs"].append(run(sess, ip, size, m.stem))
        report[m.name] = info
    (OUT / "model_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("images found:", [p.name for p in imgs])
    for k, v in report.items():
        print(k, "inputs", v["inputs"], "outputs", v["outputs"])
        for r in v["runs"]:
            print("   ", r["image"], "size", r["input_size"], "shapes", r["out_shapes"], "dets", r["num_dets_gt025"])


if __name__ == "__main__":
    main()
