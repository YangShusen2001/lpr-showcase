# -*- coding: utf-8 -*-
"""侦察三个车牌模型的 ONNX 结构：输入名/形状、动态维、算子直方图、可疑算子。"""
import json
import os
from collections import Counter

import onnx

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models")
NAMES = ["y5fu_320x_sim", "rpv3_mdict_160_r3", "litemodel_cls_96x_r1"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_evidence", "onnx_inspect.json")

# 已知在 NNAcl/MindSpore Lite 上容易出问题的算子
SUSPECT = {
    "Resize", "Upsample", "Transpose", "LSTM", "GRU", "RNN", "Gather", "GatherND",
    "ScatterND", "NonMaxSuppression", "TopK", "Einsum", "Trilu", "Where", "Range",
    "DynamicQuantizeLinear", "GridSample", "RoiAlign", "Expand", "Tile", "Pad",
}


def dims_of(vi):
    out = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append(None)
    return out


def find_model(name):
    for ext in (".onnx", ".onnx.json"):
        p = os.path.join(MODELS_DIR, name + ext)
        if os.path.isfile(p):
            return p
    return None


report = {}
for name in NAMES:
    p = find_model(name)
    if not p:
        report[name] = {"error": "model not found"}
        continue
    m = onnx.load(p)
    g = m.graph
    ops = Counter(n.op_type for n in g.node)
    dyn_inputs = []
    inputs = []
    for vi in g.input:
        d = dims_of(vi)
        inputs.append({"name": vi.name, "dims": d, "has_dynamic": any(x is None or isinstance(x, str) for x in d)})
        if inputs[-1]["has_dynamic"]:
            dyn_inputs.append(vi.name)
    report[name] = {
        "path": p,
        "file_size": os.path.getsize(p),
        "ir_version": m.ir_version,
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import],
        "producer": m.producer_name + " " + (m.producer_version or ""),
        "inputs": inputs,
        "outputs": [{"name": vi.name, "dims": dims_of(vi)} for vi in g.output],
        "dynamic_inputs": dyn_inputs,
        "op_count": len(g.node),
        "ops": dict(ops.most_common()),
        "suspect_ops_present": sorted(set(ops) & SUSPECT),
        "initializer_count": len(g.initializer),
    }

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("ok ->", OUT)
