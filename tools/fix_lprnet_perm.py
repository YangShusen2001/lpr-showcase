"""Rewrite LPRNet's six Transpose[0,3,2,1] away so the model can reach the NPU.

Why: the NPU only accepts Transpose perm == [0,1,3,2]
(`permute_matmul_fusion_pass: only support order 0132`), and LPRNet has six
Transpose nodes all using [0,3,2,1]. Result: LANDED=CPU (ADR-007 §3.2).

The six Transpose come in three pairs, each of the shape

    MaxPool(3x3) -> T[0,3,2,1] -> MaxPool(1x1, stride S) -> T[0,3,2,1] -> Conv

Two facts make each pair removable:

  1. T[0,3,2,1] is SELF-INVERSE — applying it twice returns to the original layout.
  2. The middle MaxPool has a 1x1 window, so it does not aggregate neighbours at all;
     it only samples along the stride. A 1x1-window pool with stride S is exactly
     `Slice(step=S)`.

So each pair collapses to either nothing (S = 1) or a Slice along the sampled axis.
The [0,3,2,1] transpose moves C to the end, making the layout [N,W,H,C]; the pool then
runs over the leading two dims (W,H), and with stride [1,S] it samples every S-th
element of the SECOND of those (which is the original W axis).

Verification is the point: after rewriting, the ONNX output must match the original
LPRNet elementwise. `maxAbsDiff` near zero is the only acceptable outcome — shape
agreement means nothing.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from onnx import ModelProto, TensorProto, helper, numpy_helper

sys.stdout.reconfigure(encoding="utf-8")

SRC = r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet.onnx"
DST = r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet_npufix.onnx"


def find_pair(g, pool_name):
    """Locate T -> MaxPool(1x1) -> T following `pool_name`; return node indices."""
    by_out = {}
    for i, n in enumerate(g.node):
        for o in n.output:
            by_out[o] = i
    consumers = {}
    for i, n in enumerate(g.node):
        for inp in n.input:
            consumers.setdefault(inp, []).append(i)

    i0 = next(i for i, n in enumerate(g.node) if n.name == pool_name)
    out = g.node[i0].output[0]
    t1 = consumers[out][0]
    mp = consumers[g.node[t1].output[0]][0]
    t2 = consumers[g.node[mp].output[0]][0]
    return i0, t1, mp, t2


def attr_ints(node, name):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints)
    return None


def main():
    model = ModelProto()
    with open(SRC, "rb") as f:
        model.ParseFromString(f.read())
    g = model.graph

    print("before: nodes =", len(g.node))
    for n in g.node:
        if n.op_type == "Transpose":
            print(f"  {n.name} perm={attr_ints(n, 'perm')}")

    # Each entry: (first MaxPool of the pair, stride, channel count before the pair).
    # Channel counts are from tools/lprnet_shape_probe.py (runtime shapes):
    #   pair2: 79 is (1,128,18,44) -> 81 is (1,44,18,64)   => C 128 -> 64, step 2
    #   pair3: 101 is (1,256,16,21) -> 103 is (1,21,16,64) => C 256 -> 64, step 4
    # stride 1 -> the whole pair is the identity.
    pairs = [
        ("MaxPool_2", 1, 0),
        ("MaxPool_14", 2, 128),
        ("MaxPool_34", 4, 256),
    ]

    drop = set()
    inserts = []
    rewire = {}   # old tensor name -> new tensor name
    for pool_name, stride, channels in pairs:
        i0, t1, mp, t2 = find_pair(g, pool_name)
        pool = g.node[i0]
        t1n, mpn, t2n = g.node[t1], g.node[mp], g.node[t2]
        print(f"  pair {pool_name}: T={t1n.name} pool={mpn.name} "
              f"k={attr_ints(mpn, 'kernel_shape')} s={attr_ints(mpn, 'strides')} T={t2n.name}")

        k = attr_ints(mpn, "kernel_shape")
        s = attr_ints(mpn, "strides")
        assert k == [1, 1], f"middle pool window is not 1x1: {k}"
        assert s[0] == 1 and s[1] == stride, f"stride mismatch {s} vs {stride}"

        pool_out = pool.output[0]   # the pair's input tensor
        after = t2n.output[0]       # the name consumed downstream

        if stride == 1:
            # T -> MaxPool(1x1, s=1) -> T is the identity (verified: 66:(1,64,20,90)
            # and 69:(1,64,20,90) are the same shape). Drop all three.
            rewire[after] = pool_out
            drop.update({t1, mp, t2})
        else:
            # Verified by runtime shape: in the [N,W,H,C] view produced by [0,3,2,1],
            # ONNX MaxPool pools dims 2,3, and with strides [1,S] the CHANGED axis is
            # the last one — the original CHANNEL axis (128->64, 256->64). So the pair
            # is a channel-axis subsample: Gather(axis=1, indices=0,S,2S,...).
            # Gather is in the official operator list; the 5-input Slice form is not
            # usable here because this graph's opset predates it.
            idx = list(range(0, channels, stride))
            assert len(idx) * stride >= channels, f"gather coverage wrong for {channels}/{stride}"
            idx_name = "gidx_" + pool_name
            # INT32, not INT64: the NPU delegate reports
            # `op_supported_format.cc GetOpSupportedFormat: The type of Data not find`
            # and fails to init the session when Gather carries int64 indices.
            g.initializer.append(
                helper.make_tensor(idx_name, TensorProto.INT32, [len(idx)], idx))
            node = helper.make_node(
                "Gather", [pool_out, idx_name], [after],
                name="Gather_" + pool_name, axis=1)
            inserts.append((i0, node))
            drop.update({t1, mp, t2})

    # remove dropped nodes, append inserts
    kept = [n for i, n in enumerate(g.node) if i not in drop]
    kept.extend(node for _, node in inserts)

    # re-point every input that referenced a now-missing tensor
    rewired = 0
    for n in kept:
        for k in range(len(n.input)):
            if n.input[k] in rewire:
                n.input[k] = rewire[n.input[k]]
                rewired += 1
    print(f"  rewired {rewired} input reference(s)")

    del g.node[:]
    g.node.extend(kept)

    # keep only referenced initializers
    used = set()
    for n in g.node:
        used.update(n.input)
    ki = [x for x in g.initializer if x.name in used]
    del g.initializer[:]
    g.initializer.extend(ki)
    g.value_info.clear()

    print("after : nodes =", len(g.node))
    remaining = [(n.name, attr_ints(n, "perm")) for n in g.node if n.op_type == "Transpose"]
    print("  remaining Transpose:", remaining if remaining else "NONE")

    with open(DST, "wb") as f:
        f.write(model.SerializeToString())
    print("wrote", DST, os.path.getsize(DST), "bytes")

    # ---- verification: elementwise match against the original
    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(SRC, so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(DST, so, providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).rand(1, 3, 24, 94).astype(np.float32)
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    yb = b.run(None, {b.get_inputs()[0].name: x})[0]
    print("orig out", ya.shape, " fixed out", yb.shape)
    if ya.shape != yb.shape:
        print("VERDICT: SHAPE MISMATCH")
        return 1
    d = np.abs(ya - yb)
    print(f"maxAbsDiff={float(d.max()):.8f}  meanAbsDiff={float(d.mean()):.8f}")
    print("VERDICT:", "MATCH" if float(d.max()) < 1e-4 else "MISMATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
