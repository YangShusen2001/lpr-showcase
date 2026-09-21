"""Bare-head full-pipeline cross-check on PC (CPU ONNX).

Answers ONE question: with the decode verified element-wise
(tools/verify_head_decode.py, MATCH), does the bare-head path reproduce the
original model's end-to-end result (苏ED5172) on the SAME image?

  baseline  = original y5fu_320x_sim (in-graph decode) full pipeline
  bare_head = y5fu_320x_head + numpy decode (ADR-006 §5 formulas)

If baseline == bare_head, the C++ port and the formulas are fine and any
on-device mismatch is NPU numeric drift; if they differ, the decode/port is
wrong regardless of the element-wise check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hlpr_reference as ref  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEAD = Path(r"C:\Users\26671\lpr-harmony\models_ms\head\y5fu_320x_head.onnx")
OUT = ROOT / "_evidence" / "bare_head_pipeline_check.json"

SCALES = [(40, 8), (20, 16), (10, 32)]
ANCHORS = {
    40: [(4.0, 5.0), (8.0, 10.0), (13.0, 16.0)],
    20: [(23.0, 29.0), (43.0, 55.0), (73.0, 105.0)],
    10: [(146.0, 217.0), (231.0, 300.0), (335.0, 433.0)],
}


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def decode_heads(heads):
    rows = []
    for (H, stride), h in zip(SCALES, heads):
        t = h.reshape(1, 3, 15, H, H).transpose(0, 1, 3, 4, 2)
        cx = sigmoid(t[..., 0]) * 2.0 - 0.5
        cy = sigmoid(t[..., 1]) * 2.0 - 0.5
        bw = (sigmoid(t[..., 2]) * 2.0) ** 2
        bh = (sigmoid(t[..., 3]) * 2.0) ** 2
        obj = sigmoid(t[..., 4])
        kpt = t[..., 5:13]
        cls = sigmoid(t[..., 13:15])
        gy, gx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
        gx = (gx[None, None] * stride).astype(np.float32)
        gy = (gy[None, None] * stride).astype(np.float32)
        anch = ANCHORS[H]
        aw = np.array([a[0] for a in anch], np.float32)[None, :, None, None]
        ah = np.array([a[1] for a in anch], np.float32)[None, :, None, None]
        px = cx * stride + gx
        py = cy * stride + gy
        pw = bw * aw
        ph = bh * ah
        kx = kpt[..., 0::2] * aw[..., None] + gx[..., None]
        ky = kpt[..., 1::2] * ah[..., None] + gy[..., None]
        kpts = np.stack([kx, ky], axis=-1).reshape(1, 3, H, H, 8)
        row = np.concatenate([px[..., None], py[..., None], pw[..., None],
                              ph[..., None], obj[..., None], kpts, cls], axis=-1)
        rows.append(row.reshape(1, 3 * H * H, 15))
    return np.concatenate(rows, axis=1)


def main():
    img = ref.imread_u(ref.SAMPLES / "hlpr-test.jpg")
    rec = ref.sess("rpv3_mdict_160_r3.onnx")
    cls = ref.sess("litemodel_cls_96x_r1.onnx")

    # a) baseline: original detector with in-graph decode
    det0 = ref.sess("y5fu_320x_sim.onnx")
    res0 = ref.run_pipeline(img, det0, rec, cls)

    # b) bare head + numpy decode, same downstream
    so = ort.SessionOptions()
    so.log_severity_level = 3
    head = ort.InferenceSession(str(HEAD), so, providers=["CPUExecutionProvider"])
    x, r, left, top = ref.detect_pre_precessing(img, (320, 320))
    hs = head.run(None, {head.get_inputs()[0].name: x})
    raw = decode_heads(hs)
    dets = ref.detect_post(np.asarray(raw), r, left, top)
    bare = []
    for row in dets:
        marks = row[5:13].reshape(4, 2).astype(int)
        crop = ref.get_rotate_crop_image(img, marks)
        code, conf = ref.recognize(rec, crop)
        bare.append({"code": code, "conf": round(float(conf), 4),
                     "marks": marks.tolist(), "crop": list(crop.shape[:2]),
                     "det_score": round(float(row[4]), 4)})

    result = {"baseline": res0, "bare_head": bare,
              "codes_match": [p["code"] for p in res0] == [p["code"] for p in bare]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
