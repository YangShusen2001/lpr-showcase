"""Verify a hand-written YOLOv5 decode reproduces the original model's output.

The bare-head export (tools/export_bare_head.py) gets the detector onto the NPU, but it
only emits the three raw head tensors [1,45,40,40] / [1,45,20,20] / [1,45,10,10].
Everything the graph used to do after them — Reshape -> Transpose -> Slice -> Sigmoid ->
Concat -> Mul/Add with anchor and grid constants — must be redone in C++.

Before writing any C++ this proves the maths, by decoding the bare-head outputs in numpy
and diffing against the ORIGINAL model's [1,6300,15] output on the same input.

Constants recovered from the graph (they are what made the NPU reject the model):
    1005  [1,3,H,W,2]  anchor w/h          (40x40 scale: 4.0 .. 16.0)
    1014  [1,3,H,W,2]  grid, in pixels     (0 .. 312 = 39*8)
    988   []           scalar 2.0
    981   [1,3,H,W,15] ALL ZERO -> the Add is a no-op, nothing to carry over

Channel layout of the 15: [cx, cy, w, h, obj, kpt0x, kpt0y, ..., kpt3y, cls0, cls1]
(confirmed by the Slice index constants 0,1,2,4,5,7,9,11,13,15).

ADR-006 §5 corrections (2026-09-17, verified by export_decode_stages.py):
  - kpt channels [5:13] are RAW LOGITS — the graph sigmoids only [0:5] and
    [13:15]; ch5 range [-2.465, 1.396] proves no sigmoid.
  - kpts multiply the ANCHOR w/h and add the grid in PIXELS; cx/cy/w/h use the
    (sigmoid*2-0.5+grid_idx)*stride form. Two paths, same grid, different shape.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"C:\Users\26671\Desktop\车牌识别"
ORIG = os.path.join(ROOT, r"assets\models\y5fu_320x_sim.onnx.json")
HEAD = r"C:\Users\26671\lpr-harmony\models_ms\head\y5fu_320x_head.onnx"
IMG = os.path.join(ROOT, r"assets\samples\hlpr-test.jpg")

SCALES = [(40, 8), (20, 16), (10, 32)]  # (grid size, stride) for a 320 input


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def letterbox(path, size=320):
    """Match the pipeline's preprocessing: BGR->RGB, /255, NCHW."""
    import cv2
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, left, top


def decode(heads, anchors):
    """heads: list of [1,45,H,W] for H in 40,20,10. Returns [1, 6300, 15]."""
    rows = []
    for (H, stride), h, anch in zip(SCALES, heads, anchors):
        b, c, hh, ww = h.shape
        assert (c, hh, ww) == (45, H, H), (c, hh, ww)
        # [1,45,H,W] -> [1,3,15,H,W] -> [1,3,H,W,15]
        t = h.reshape(1, 3, 15, H, H).transpose(0, 1, 3, 4, 2)

        cx = (sigmoid(t[..., 0]) * 2.0 - 0.5)
        cy = (sigmoid(t[..., 1]) * 2.0 - 0.5)
        bw = (sigmoid(t[..., 2]) * 2.0) ** 2
        bh = (sigmoid(t[..., 3]) * 2.0) ** 2
        obj = sigmoid(t[..., 4])
        # ADR-006 §5: kpt channels are RAW LOGITS — the graph only sigmoids
        # [0:5] and [13:15]; [5:13] passes through. kpts multiply the ANCHOR
        # (not the stride) and add the grid in PIXELS (1014 = 0..312 = 39*8).
        kpt = t[..., 5:13]
        cls = sigmoid(t[..., 13:15])

        # grid: pixel coordinates of each cell centre
        gy, gx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
        gx = (gx[None, None] * stride).astype(np.float32)
        gy = (gy[None, None] * stride).astype(np.float32)

        # anchors: anch is [3, 2] (w, h) per anchor
        aw = anch[:, 0][None, :, None, None].astype(np.float32)
        ah = anch[:, 1][None, :, None, None].astype(np.float32)

        px = cx * stride + gx
        py = cy * stride + gy
        pw = bw * aw
        ph = bh * ah

        # keypoints: raw logit * anchor + grid pixels (ADR-006 §5, measured)
        kx = kpt[..., 0::2] * aw[..., None] + gx[..., None]
        ky = kpt[..., 1::2] * ah[..., None] + gy[..., None]
        kpts = np.stack([kx, ky], axis=-1).reshape(1, 3, H, H, 8)

        row = np.concatenate(
            [px[..., None], py[..., None], pw[..., None], ph[..., None],
             obj[..., None], kpts, cls], axis=-1)  # [1,3,H,H,15]
        rows.append(row.reshape(1, 3 * H * H, 15))
    return np.concatenate(rows, axis=1)


def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    x, r, left, top = letterbox(IMG)
    print(f"input {x.shape}  r={r:.6f} left={left} top={top}")

    ref = ort.InferenceSession(ORIG, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {ref.get_inputs()[0].name: x})[0]
    print("original output", ref_out.shape)

    head = ort.InferenceSession(HEAD, so, providers=["CPUExecutionProvider"])
    names = [o.name for o in head.get_outputs()]
    hs = head.run(None, {head.get_inputs()[0].name: x})
    print("bare head outputs", [(n, h.shape) for n, h in zip(names, hs)])

    # anchors straight out of the graph. Three scales, one constant each:
    #   1005  -> 40x40  [[4,5],[8,10],[13,16]]
    #   1118  -> 20x20  [[23,29],[43,55],[73,105]]
    #   1231  -> 10x10  [[146,217],[231,300],[335,433]]
    # (The 0-valued rank-5 constants 981/1094/1207 are no-op Adds and carry nothing.)
    from onnx import ModelProto, numpy_helper
    m = ModelProto()
    with open(ORIG, "rb") as f:
        m.ParseFromString(f.read())
    init = {i.name: i for i in m.graph.initializer}
    anchors = []
    for nm in ["1005", "1118", "1231"]:
        if nm in init:
            a = numpy_helper.to_array(init[nm])  # [1,3,H,W,2]
            anchors.append(a[0, :, 0, 0, :])     # [3,2]
            print(f"  anchors {nm}: {a[0, :, 0, 0, :].tolist()}")
    if len(anchors) != 3:
        print("!! could not find all anchor constants")
        return 1

    mine = decode(hs, anchors)
    print("decoded", mine.shape)

    d = np.abs(mine - ref_out)
    print(f"maxAbsDiff={float(d.max()):.6f}  meanAbsDiff={float(d.mean()):.6f}")
    worst = np.unravel_index(np.argmax(d), d.shape)
    print(f"worst at {worst}: mine={float(mine[worst]):.4f} ref={float(ref_out[worst]):.4f}")
    print("VERDICT:", "MATCH" if float(d.max()) < 1e-3 else "MISMATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
