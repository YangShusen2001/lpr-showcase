"""Export a bare-head y5fu: cut the anchor-grid decode out of the graph.

Why: the NPU rejects y5fu because of 12 rank-5 CONSTANTS (ADR-004 §4.1), and those
constants are anchor-grid/stride tensors produced by the Detect head's decode step.
Remove the decode and the rank-5 constants go with it.

Cut points come from _evidence/head-anatomy.json — the three head convs:
    Conv_500 -> "948"  [1,45,40,40]   (weight model.21.m.0.weight)
    Conv_596 -> "1061" [1,45,20,20]
    Conv_692 -> "1174" [1,45,10,10]
45 = 3 anchors x 15 (4 box + 1 obj + 8 kpt + 2 cls). Everything after these tensors
is decode (Reshape -> Transpose -> Slice/Sigmoid -> Mul/Add with the grid constants),
so keeping only the subgraph that PRODUCES them drops the whole decode.

This is a *feasibility* export: it answers "does the NPU accept the model once the
rank-5 constants are gone?" — decoding is NOT implemented here. If the NPU accepts
it, the decode has to be reimplemented in C++ using the constants extracted from the
original graph, and that decode must be verified against the current end-to-end path
before it can replace anything.
"""
import os
import sys

from onnx import ModelProto, TensorProto, helper

ROOT = r"C:\Users\26671\Desktop\车牌识别"
SRC = os.path.join(ROOT, r"assets\models\y5fu_320x_sim.onnx.json")
OUT_DIR = r"C:\Users\26671\lpr-harmony\models_ms\head"
DST = os.path.join(OUT_DIR, "y5fu_320x_dec.onnx")

# tensor name -> shape (None = let ONNX infer). Two candidate cut points:
#
#   A) the 1x1 head convs themselves — 948/1061/1174 [1,45,H,W]. Rank-4, safest for
#      the NPU, but then ALL of the decode (reshape/transpose/sigmoid/slice) is mine.
#   B) the post-Add tensors — 980/1093/1206 [1,3,H,W,15]. Verified identical to
#      982/1095/1208 because the rank-5 constant 981/1094/1207 is all zeros, so the
#      Add is a no-op. Cutting here keeps sigmoid + the Reshape/Transpose inside the
#      graph (they were NOT what the NPU complained about) and leaves only the cheap
#      per-channel decode for C++. Preferred: fewer moving parts on our side.
#
# NOTE: cutting at B makes the graph emit a rank-5 tensor. Whether the NPU accepts a
# rank-5 *activation* (as opposed to a rank-5 *constant*) is exactly what we are
# testing — the constants are what broke it before.
HEADS = {
    "980": None,   # 40x40 scale, [1,3,40,40,15]
    "1093": None,  # 20x20 scale
    "1206": None,  # 10x10 scale
}

sys.stdout.reconfigure(encoding="utf-8")


def main():
    m = ModelProto()
    with open(SRC, "rb") as f:
        m.ParseFromString(f.read())
    g = m.graph

    print(f"source: {len(g.node)} nodes, {len(g.initializer)} initializers")
    print(f"  rank-5 constants: {sum(1 for i in g.initializer if len(i.dims) == 5)}")

    init_names = {i.name for i in g.initializer}
    input_names = {i.name for i in g.input}

    producer = {}
    for i, n in enumerate(g.node):
        for o in n.output:
            producer[o] = i

    # Keep exactly the nodes needed to produce the three head tensors: walk backwards
    # from each head output until we reach graph inputs or constants (both are leaves).
    keep = set()
    stack = list(HEADS)
    missing = []
    while stack:
        t = stack.pop()
        if t in init_names or t in input_names:
            continue
        i = producer.get(t)
        if i is None:
            missing.append(t)
            continue
        if i in keep:
            continue
        keep.add(i)
        stack.extend(g.node[i].input)

    if missing:
        print(f"  !! tensors with no producer: {sorted(set(missing))[:6]}")

    print(f"  kept {len(keep)} / {len(g.node)} nodes "
          f"(dropped {len(g.node) - len(keep)} decode nodes)")
    # protobuf repeated fields reject slice assignment; clear + extend instead.
    kept_nodes = [n for i, n in enumerate(g.node) if i in keep]
    del g.node[:]
    g.node.extend(kept_nodes)

    # Re-point the graph outputs at the head tensors.
    del g.output[:]
    for name, dims in HEADS.items():
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, dims))

    # Drop initializers nothing references any more (this is where the rank-5
    # constants actually leave).
    used = set()
    for n in g.node:
        used.update(n.input)
    before = len(g.initializer)
    kept_init = [x for x in g.initializer if x.name in used]
    del g.initializer[:]
    g.initializer.extend(kept_init)
    left5 = sum(1 for x in g.initializer if len(x.dims) == 5)
    print(f"  initializers {before} -> {len(g.initializer)}  (rank-5 left: {left5})")

    # Stale value_info entries would reference dropped tensors.
    g.value_info.clear()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DST, "wb") as f:
        f.write(m.SerializeToString())
    print(f"wrote {DST}  ({os.path.getsize(DST)} bytes)")

    # Self-check: the cut graph must still run.
    import onnxruntime as ort
    import numpy as np
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(DST, so, providers=["CPUExecutionProvider"])
    print("  outputs:", [(o.name, o.shape) for o in s.get_outputs()])
    x = np.zeros((1, 3, 320, 320), dtype=np.float32)
    ys = s.run(None, {s.get_inputs()[0].name: x})
    for o, y in zip(s.get_outputs(), ys):
        print(f"  ran {o.name}: {y.shape} sum={float(np.abs(y).sum()):.2f}")


if __name__ == "__main__":
    sys.exit(main())
