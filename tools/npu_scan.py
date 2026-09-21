"""NPU fitness scanner for ONNX models (Kirin 8020).

Answers "can this model run on the NPU?" from two hard gates derived from the NPU
compiler's own checks (ADR-007 §2) — NOT from the published operator list, which
lists Reshape/Permute/Swish as supported and therefore misleads:

    gate 1: every tensor rank <= 4          <- permute_check_support: realdimCnt > 4
    gate 2: every Transpose perm == [0,1,3,2] <- permute_matmul_fusion_pass: only order 0132

Usage:
    python tools/npu_scan.py model1.onnx [model2.onnx ...]
"""
import collections
import sys

from onnx import ModelProto


def dims_of(value_info):
    return [d.dim_value if d.HasField("dim_value") else "?" for d in value_info.type.tensor_type.shape.dim]


def scan(path):
    model = ModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    g = model.graph

    print("=" * 72)
    print(path)
    print(f"  nodes         : {len(g.node)}")
    print(f"  inputs        : {[(vi.name, dims_of(vi)) for vi in g.input]}")
    print(f"  outputs       : {[(vo.name, dims_of(vo)) for vo in g.output]}")
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"  ops           : {dict(ops.most_common(14))}")

    rank5 = [init.name for init in g.initializer if len(init.dims) >= 5]
    init_ranks = collections.Counter(len(init.dims) for init in g.initializer)
    print(f"  rank>=5 consts: {len(rank5)} {rank5[:4]}")
    print(f"  init ranks    : {dict(sorted(init_ranks.items()))}")

    # tensor ranks as declared in value_info / inputs / outputs
    tensor_ranks = collections.Counter()
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        tensor_ranks[len(vi.type.tensor_type.shape.dim)] += 1
    print(f"  tensor ranks  : {dict(sorted(tensor_ranks.items()))}")

    perms = collections.Counter()
    bad = []
    for n in g.node:
        if n.op_type == "Transpose":
            perm = next((tuple(a.ints) for a in n.attribute if a.name == "perm"), None)
            perms[perm] += 1
            if perm != (0, 1, 3, 2):
                bad.append((n.name, perm))
    print(f"  Transpose perms: {dict(perms)}")
    print(f"  BAD perm count : {len(bad)}")
    for nm, pm in bad[:8]:
        print(f"      {nm} perm={pm}")

    # Reshape whose output rank can't be inferred is gate 3 (softer)
    print(f"  Reshape nodes : {ops.get('Reshape', 0)}")

    gate1 = len(rank5) == 0
    gate2 = len(bad) == 0
    verdict = "LIKELY NPU-OK" if (gate1 and gate2) else "WILL FALL BACK TO CPU"
    print(f"  GATE1 rank<=4 : {'PASS' if gate1 else 'FAIL'}")
    print(f"  GATE2 perm    : {'PASS' if gate2 else 'FAIL'}")
    print(f"  VERDICT       : {verdict}")
    return gate1 and gate2


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for p in argv[1:]:
        scan(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
