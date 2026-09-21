"""Locate the exact decode step by exporting intermediate tensors as graph outputs.

The first decode attempt (tools/verify_head_decode.py) mismatched, and the node dump
showed the keypoint slices get multiplied by the ANCHOR constant rather than a stride —
so the split between "what the graph already did" and "what I must redo" was drawn in
the wrong place.

This script settles it empirically instead of by reading node names: it re-exports the
original graph with 982 / 1095 / 1208 (the tensors right after the 981-add) promoted to
graph outputs, runs them, and compares against the raw head tensors pushed through
sigmoid alone. If they match, those tensors are pre-decode and the whole decode is mine
to write; if not, part of the decode is already baked in.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from onnx import ModelProto, TensorProto, helper

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"C:\Users\26671\Desktop\车牌识别"
SRC = os.path.join(ROOT, r"assets\models\y5fu_320x_sim.onnx.json")
IMG = os.path.join(ROOT, r"assets\samples\hlpr-test.jpg")
WORK = r"C:\Users\26671\lpr-harmony\models_ms\head"

TARGETS = ["982", "1095", "1208"]  # post-Add tensors, one per scale


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def letterbox(path, size=320):
    import cv2
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def cut(src, targets, dst):
    m = ModelProto()
    with open(src, "rb") as f:
        m.ParseFromString(f.read())
    g = m.graph

    init_names = {i.name for i in g.initializer}
    input_names = {i.name for i in g.input}
    producer = {}
    for i, n in enumerate(g.node):
        for o in n.output:
            producer[o] = i

    keep = set()
    stack = list(targets)
    while stack:
        t = stack.pop()
        if t in init_names or t in input_names:
            continue
        i = producer.get(t)
        if i is None or i in keep:
            continue
        keep.add(i)
        stack.extend(g.node[i].input)

    kept = [n for i, n in enumerate(g.node) if i in keep]
    del g.node[:]
    g.node.extend(kept)

    del g.output[:]
    for t in targets:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))

    used = set()
    for n in g.node:
        used.update(n.input)
    ki = [x for x in g.initializer if x.name in used]
    del g.initializer[:]
    g.initializer.extend(ki)
    g.value_info.clear()

    with open(dst, "wb") as f:
        f.write(m.SerializeToString())
    return dst


def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    x = letterbox(IMG)

    # 1. raw head tensors
    head = ort.InferenceSession(os.path.join(WORK, "y5fu_320x_head.onnx"), so,
                                providers=["CPUExecutionProvider"])
    raw = head.run(None, {head.get_inputs()[0].name: x})

    # 2. the post-Add tensors, straight from the original graph
    stage_path = cut(SRC, TARGETS, os.path.join(WORK, "y5fu_320x_stage.onnx"))
    st = ort.InferenceSession(stage_path, so, providers=["CPUExecutionProvider"])
    got = st.run(None, {st.get_inputs()[0].name: x})

    for tname, r, s in zip(TARGETS, raw, got):
        H = r.shape[-1]
        sig = sigmoid(r.reshape(1, 3, 15, H, H).transpose(0, 1, 3, 4, 2))
        d = np.abs(sig - s)
        print(f"{tname}: stage{s.shape}  maxAbs(sigmoid(raw) - stage) = {float(d.max()):.6f}")
        # per-channel, to see which parts are pre- vs post-decode
        for c in [0, 1, 2, 4, 5, 13]:
            dc = float(np.abs(sig[..., c] - s[..., c]).max())
            print(f"    ch{c:2d}: sigmoid-range=[{float(sig[...,c].min()):.3f},"
                  f"{float(sig[...,c].max()):.3f}]  stage-range=[{float(s[...,c].min()):.3f},"
                  f"{float(s[...,c].max()):.3f}]  diff={dc:.6f}")


if __name__ == "__main__":
    sys.exit(main())
