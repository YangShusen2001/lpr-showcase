# -*- coding: utf-8 -*-
"""用 onnxruntime 读三个模型的输入/输出规格（venv 里的 onnx 包已损坏，不依赖它）。"""
import json
import os

import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "assets", "models")
OUT = os.path.join(ROOT, "_evidence", "onnx_io.json")
NAMES = ["y5fu_320x_sim", "rpv3_mdict_160_r3", "litemodel_cls_96x_r1"]

report = {"ort_version": ort.__version__, "models": {}}
for n in NAMES:
    entry = {}
    for ext in (".onnx", ".onnx.json"):
        p = os.path.join(MODELS, n + ext)
        if os.path.isfile(p):
            entry["path"] = p
            entry["size"] = os.path.getsize(p)
            break
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        s = ort.InferenceSession(entry["path"], so, providers=["CPUExecutionProvider"])
        entry["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in s.get_inputs()]
        entry["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in s.get_outputs()]
        entry["ok"] = True
    except Exception as e:
        entry["ok"] = False
        entry["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
    report["models"][n] = entry

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("ok")
