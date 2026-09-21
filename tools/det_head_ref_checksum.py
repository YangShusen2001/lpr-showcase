"""PC reference for the CPU-diff protocol on the bare-head detector.

Mirrors MsBenchRun's protocol exactly (ms_engine.cpp):
  input fill  x[j] = ((j * 2654435761) % 1000) / 1000.0
  checksum    L2 = sqrt(sum(v^2)) over output[0] read as fp32
  maxAbs      max |v| over output[0]

Run this, then compare against the on-device `DET BENCH` hilog line
(checksum / maxAbs fields). The gap quantifies Kirin 8020 NNRT numeric drift
on the rank-4 bare head, independent of any image content.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HEAD = Path(r"C:\Users\26671\lpr-harmony\models_ms\head\y5fu_320x_head.onnx")
OUT = Path(r"C:\Users\26671\Desktop\车牌识别\_evidence\det_head_ref_checksum.json")


def main():
    n = 3 * 320 * 320
    j = np.arange(n, dtype=np.uint64)
    x = ((j * np.uint64(2654435761)) % np.uint64(1000)).astype(np.float32) / 1000.0
    x = x.reshape(1, 3, 320, 320)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(str(HEAD), so, providers=["CPUExecutionProvider"])
    outs = s.run(None, {s.get_inputs()[0].name: x})

    rows = []
    for i, o in enumerate(outs):
        a = np.asarray(o, dtype=np.float64)
        rows.append({
            "output": i,
            "shape": list(o.shape),
            "l2": round(float(np.sqrt((a ** 2).sum())), 4),
            "maxAbs": round(float(np.abs(a).max()), 6),
        })

    result = {"protocol": "x[j]=((j*2654435761)%1000)/1000", "inputElems": n,
              "outputs": rows}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
