"""Inspect the HyperLPR3 ONNX models: exact input/output specs."""
from pathlib import Path

import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"

for name in ("y5fu_320x_sim.onnx", "y5fu_640x_sim.onnx", "rpv3_mdict_160_r3.onnx", "litemodel_cls_96x_r1.onnx"):
    p = MODELS / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    print(f"\n=== {name} ({p.stat().st_size/1024/1024:.2f} MB) ===")
    for i in sess.get_inputs():
        print(f"  IN  {i.name:20s} shape={i.shape} type={i.type}")
    for o in sess.get_outputs():
        print(f"  OUT {o.name:20s} shape={o.shape} type={o.type}")
