#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump an ONNX model's operator set (and optionally node order) to judge ncnn convertibility.

Why: ncnn's ONNX importer supports a fixed op list. Before promising "put the recogniser
on the GPU too", we need to know whether the graph only uses ops that ncnn implements
(Conv/BN/ReLU/Pool/MatMul...) or leans on things it does not (LSTM/GRU, Einsum,
LayerNormalization, dynamic shapes...).

Usage:
  python inspect_onnx_ops.py <model.onnx> [--nodes] [--limit N]

Accepts `.onnx` and the project's `.onnx.json` (same bytes, whitelist-suffix rename).
"""
import argparse
import collections
import os
import sys

import onnx


def load(path: str):
    """The project renames .onnx -> .onnx.json to dodge a static-host whitelist;
    onnx.load() cannot infer that, so parse from bytes explicitly."""
    m = onnx.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--nodes", action="store_true", help="print node order (op + name)")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"[x] not found: {args.model}")
        return 1
    m = load(args.model)

    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(f"# {os.path.basename(args.model)}")
    print(f"opset       = {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    print(f"ir_version  = {m.ir_version}")
    print(f"producer    = {m.producer_name} {m.producer_version}")
    print(f"nodes       = {len(m.graph.node)}  initializers = {len(m.graph.initializer)}")

    def dims(v):
        out = []
        for d in v.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                out.append(str(d.dim_value))
            elif d.HasField("dim_param"):
                out.append(d.dim_param or "?")
            else:
                out.append("?")
        return "[" + ",".join(out) + "]"

    print("\ninputs:")
    for v in m.graph.input:
        if v.name in {i.name for i in m.graph.initializer}:
            continue  # some exporters list initializers as inputs
        print(f"  {v.name:24s} {dims(v)}")
    print("outputs:")
    for v in m.graph.output:
        print(f"  {v.name:24s} {dims(v)}")

    print(f"\nop histogram ({len(ops)} distinct):")
    for op, n in ops.most_common():
        print(f"  {op:26s} {n:5d}")

    if args.nodes:
        print("\nnode order:")
        for i, n in enumerate(m.graph.node):
            if i >= args.limit:
                print(f"  ... ({len(m.graph.node) - args.limit} more)")
                break
            print(f"  [{i:4d}] {n.op_type:22s} {n.name}")

    # Dynamic-batch / dynamic-shape warning: ncnn needs fixed shapes.
    dyn = [v.name for v in list(m.graph.input) + list(m.graph.output)
           if any(not d.HasField("dim_value") for d in v.type.tensor_type.shape.dim)]
    if dyn:
        print(f"\n[!] dynamic dims present in: {dyn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())