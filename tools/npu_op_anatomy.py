"""Dissect the NPU-unfriendly ops in the two hard models.

Two different failure modes are in play and they must not be conflated:

  y5fu_320x_sim  -> rejected because of 12 rank-5 CONSTANTS (anchor-grid decode baked
                    into the graph). ADR-004 §4.1 already disproved four graph-surgery
                    fixes; the only remaining route is a structural change (bare head).
  rpv3_mdict_160_r3 -> ACCEPTED by the NPU, but the compiler reports Reshape / Permute /
                    GatherV2D / Swish as unsupported, so those subgraphs fall back to
                    CPU (mixed execution, ADR-005 §6).

This script answers "can the ops be rewritten?" with graph facts rather than opinion:
for every offending node it prints the op, the tensor shapes flowing through it, and
whether its inputs are constants (i.e. whether it can be constant-folded away).
"""
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"

# Ops the NPU compiler flagged, per ADR-005 §6.
SUSPECT = {"Reshape", "Transpose", "Gather", "GatherND", "Sigmoid", "Mul", "Swish"}


def shape_of(value_info):
    if value_info.type.WhichOneof("value") != "tensor_type":
        return "?"
    tt = value_info.type.tensor_type
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def main():
    for name in ["y5fu_320x_sim", "rpv3_mdict_160_r3"]:
        path = MODELS / (name + ".onnx.json")
        if not path.exists():
            path = MODELS / (name + ".onnx")
        m = onnx.load(str(path))
        g = m.graph

        print("=" * 78)
        print(f"## {name}   nodes={len(g.node)}")

        ops = Counter(n.op_type for n in g.node)
        print("   op histogram:", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:14]))

        # initializer name -> shape, for "is this input a constant?" checks
        consts = {}
        for init in g.initializer:
            consts[init.name] = list(init.dims)

        # every tensor's shape, as far as the graph declares it
        shapes = {}
        for vi in list(g.input) + list(g.value_info) + list(g.output):
            shapes[vi.name] = shape_of(vi)

        # rank-5 constants: the y5fu blocker
        r5 = {k: v for k, v in consts.items() if len(v) == 5}
        print(f"   initializers={len(consts)}  rank-5 constants={len(r5)}")
        for k, v in list(r5.items())[:6]:
            print(f"      rank5 {k} {v}")

        # offending ops, with their context
        hits = [n for n in g.node if n.op_type in SUSPECT]
        print(f"   suspect-op nodes={len(hits)}")
        shown = 0
        for n in hits:
            ins = []
            for i in n.input:
                tag = f"C{consts[i]}" if i in consts else f"V{shapes.get(i, '?')}"
                ins.append(f"{i[:34]}{tag}")
            outs = [f"{o[:30]}{shapes.get(o, '?')}" for o in n.output]
            print(f"      [{n.op_type}] {n.name[:46]}")
            print(f"         in : {' | '.join(ins)}")
            print(f"         out: {' | '.join(outs)}")
            shown += 1
            if shown >= 10:
                print(f"      ... ({len(hits) - shown} more)")
                break


if __name__ == "__main__":
    sys.exit(main())
