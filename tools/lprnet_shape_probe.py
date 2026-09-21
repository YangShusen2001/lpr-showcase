"""Reveal the true shapes flowing through LPRNet's Transpose/MaxPool pairs.

The rewrite in tools/fix_lprnet_perm.py assumed the [0,3,2,1] transpose puts the
layout at [N,W,H,C] and that the following 1x1 MaxPool therefore samples the original
W axis. The shape-inference conflict (inferred 7 vs declared 18) says that assumption
is wrong somewhere.

ONNX MaxPool pools over dims 2 and 3 only. So after [0,3,2,1] on an [N,C,H,W] tensor
the pooled axes are NOT (W,H) — this script prints the actual runtime shapes at every
step of the three pairs so the semantics can be read off instead of guessed.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from onnx import ModelProto, TensorProto, helper

sys.stdout.reconfigure(encoding="utf-8")

SRC = r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet.onnx"
WORK = os.path.dirname(SRC)

# Every tensor on the three Transpose/MaxPool chains, in graph order.
TARGETS = [
    "66", "67", "68", "69",   # pair 1: MaxPool_2 -> T_3 -> MaxPool_4 -> T_5
    "79", "80", "81", "82",   # pair 2
    "101", "102", "103", "104",  # pair 3
]


def main():
    model = ModelProto()
    with open(SRC, "rb") as f:
        model.ParseFromString(f.read())
    g = model.graph

    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n

    print("=== graph-level view of the three pairs ===")
    for t in TARGETS:
        n = by_out.get(t)
        if n is None:
            print(f"  {t}: (no producer)")
            continue
        attrs = {}
        for a in n.attribute:
            attrs[a.name] = list(a.ints) if a.ints else a.i
        print(f"  {t:>4} <- {n.op_type}({n.name})  {attrs}  in={list(n.input)}")

    # Promote the intermediates to graph outputs and read their real shapes.
    del g.output[:]
    for t in TARGETS:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))

    tmp = os.path.join(WORK, "_lprnet_shapes.onnx")
    with open(tmp, "wb") as f:
        f.write(model.SerializeToString())

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).rand(1, 3, 24, 94).astype(np.float32)
    ys = s.run(None, {s.get_inputs()[0].name: x})

    print("\n=== runtime shapes ===")
    for t, y in zip(TARGETS, ys):
        print(f"  {t:>4}: {tuple(y.shape)}")

    print("\nNOTE: ONNX MaxPool pools dims 2,3. Compare the shapes above against the")
    print("      assumed [N,W,H,C] to see which axis the 1x1 pool actually strides over.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
