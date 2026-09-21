"""PC reference for the ncnn port: same image, same NCHW layout, ONNX runtime.

The device-side ncnn run (lpr.ncnnRun via hilog `NCNN RUN`) prints the L2/maxAbs of
the three bare-head outputs for hlpr-test.jpg. This script computes the SAME numbers
with the SAME preprocessing (letterbox -> NCHW RGB /255, both via the project's
canonical implementation) on ONNX runtime, so the two are directly comparable.

Any gap is the ncnn port's fidelity (or its fp16/threading defaults), not a
layout artefact — the layouts match by construction here.
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
OUT = ROOT / "_evidence" / "ncnn_ref_check.json"


def main():
    img = ref.imread_u(ref.SAMPLES / "hlpr-test.jpg")
    x, r, left, top = ref.detect_pre_precessing(img, (320, 320))  # [1,3,320,320] RGB /255

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
            "maxAbs": round(float(np.abs(a).max()), 4),
        })

    result = {"image": "hlpr-test.jpg", "preprocess": "letterbox 320, NCHW RGB /255",
              "r": round(float(r), 6), "left": int(left), "top": int(top),
              "outputs": rows}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
