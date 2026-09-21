"""Faithful Python reference implementation of the HyperLPR3 ONNX pipeline.

This is the *ground truth* the browser port must match. Every function below is a
direct transcription of the upstream algorithm:

  * detect  -> open_image_models ... no: HyperLPR3 Prj-Python/hyperlpr3/inference/multitask_detect.py
  * rectify -> hyperlpr3/common/tools_process.py :: get_rotate_crop_image
  * recognise -> hyperlpr3/inference/recognition.py :: encode_images + CTC greedy decode
  * classify -> hyperlpr3/inference/classification.py

Run:  python tools/hlpr_reference.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"
SAMPLES = ROOT / "assets" / "samples"
EVID = ROOT / "_evidence"
EVID.mkdir(exist_ok=True)

TOKEN = ["blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "云", "京", "冀", "吉", "学", "宁",
         "川", "挂", "新", "晋", "桂", "民", "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", "蒙", "藏", "警", "豫",
         "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", "黑", "领", "使", "澳"]

DOUBLE = 1  # plate layer class index meaning "double layer"


def model_path(name: str) -> Path:
    """Resolve a model file, accepting the browser runtime's renamed copy.

    The static preview host this project is demoed on serves an extension
    whitelist and refuses .onnx (HTTP 403), so the very same bytes are stored as
    `<name>.onnx.json` for the browser. Keeping one file and resolving it here is
    better than duplicating 14 MB and letting the two copies drift apart.
    """
    direct = MODELS / name
    if direct.exists():
        return direct
    renamed = MODELS / (name + ".json")
    if renamed.exists():
        return renamed
    raise FileNotFoundError(f"model not found: {direct} or {renamed}")


def sess(name: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(model_path(name)), so, providers=["CPUExecutionProvider"])


def imread_u(path: Path):
    """cv2.imread cannot open non-ASCII paths on Windows; decode from a byte buffer instead."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- detection
def letter_box(img, size=(320, 320)):
    h, w, _ = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)
    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img, r, left, top


def detect_pre_precessing(img, img_size):
    img, r, left, top = letter_box(img, img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
    img = img / 255
    return img.reshape(1, *img.shape), r, left, top


def xywh2xyxy(boxes):
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes, iou_thresh):
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter_area = w * h
        union_area = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                      + (boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1]))
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep


def restore_box(boxes, r, left, top):
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top
    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes


def detect_post(dets, r, left, top, conf_thresh=0.25, iou_thresh=0.5):
    choice = dets[:, :, 4] > conf_thresh
    dets = dets[choice]
    if dets.size == 0:
        return np.zeros((0, 14), dtype=np.float32)
    dets[:, 13:15] *= dets[:, 4:5]
    boxes = xywh2xyxy(dets[:, :4])
    score = np.max(dets[:, 13:15], axis=-1, keepdims=True)
    index = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
    output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)
    keep = nms(output, iou_thresh)
    return restore_box(output[keep], r, left, top)


def detect(det, img, conf_thresh=0.25, iou_thresh=0.5):
    size = tuple(int(v) for v in det.get_inputs()[0].shape[2:])
    x, r, left, top = detect_pre_precessing(img, size)
    raw = det.run([det.get_outputs()[0].name], {det.get_inputs()[0].name: x})[0]
    return detect_post(np.asarray(raw), r, left, top, conf_thresh, iou_thresh)


# ---------------------------------------------------------------- rectification
def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    points = points.astype(np.float32)
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dh, dw = dst.shape[0:2]
    if dw and dh * 1.0 / dw >= 1.5:
        dst = np.rot90(dst)
    return dst


# ---------------------------------------------------------------- recognition
def encode_images(image, max_wh_ratio, target_shape, limited_max_width=160, limited_min_width=48):
    imgC = 3
    imgH, imgW = target_shape
    assert imgC == image.shape[2]
    max_wh_ratio = max(max_wh_ratio, imgW / imgH)
    imgW = int(imgH * max_wh_ratio)
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = image.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    resized_w = imgW if ratio_imgH > imgW else int(ratio_imgH)
    resized_image = cv2.resize(image, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = (resized_image.transpose((2, 0, 1)) - 127.5) / 127.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def ctc_decode(index_row, prob_row):
    chars, confs = [], []
    for i, idx in enumerate(index_row):
        if idx == 0:
            continue
        if i > 0 and index_row[i - 1] == idx:
            continue
        chars.append(TOKEN[int(idx)] if int(idx) < len(TOKEN) else "?")
        confs.append(float(prob_row[i]))
    return "".join(chars), float(np.mean(confs)) if confs else 0.0


def recognize(rec, crop):
    h, w = crop.shape[:2]
    data = encode_images(crop, w * 1.0 / h, tuple(int(v) for v in rec.get_inputs()[0].shape[2:]))
    data = np.expand_dims(data, 0)
    out = np.asarray(rec.run([rec.get_outputs()[0].name], {rec.get_inputs()[0].name: data})[0])
    prod = out[0]
    return ctc_decode(np.argmax(prod, axis=1), np.max(prod, axis=1))


# ---------------------------------------------------------------- classification
def classify(cls, crop):
    """Upstream keeps BGR here (classification.py :: encode_images does no channel swap)."""
    size = int(cls.get_inputs()[0].shape[2])
    x = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None, ...]
    out = np.asarray(cls.run([cls.get_outputs()[0].name], {cls.get_inputs()[0].name: x})[0])[0]
    return out


# ---------------------------------------------------------------- pipeline
def run_pipeline(img, det, rec, cls, conf=0.25, iou=0.5, full=False):
    t0 = time.perf_counter()
    dets = detect(det, img, conf, iou)
    t1 = time.perf_counter()
    results = []
    for row in dets:
        rect = row[:4].astype(int)
        score = float(row[4])
        marks = row[5:13].reshape(4, 2).astype(int)
        layer = int(row[13])
        crop = get_rotate_crop_image(img, marks)
        t2 = time.perf_counter()
        if layer == DOUBLE:
            h, _ = crop.shape[:2]
            line = int(h * 0.4)
            top_code, top_conf = recognize(rec, crop[:line, :, ])
            bot_code, bot_conf = recognize(rec, crop[line:, :])
            code, rconf = top_code + bot_code, (top_conf + bot_conf) / 2
        else:
            code, rconf = recognize(rec, crop)
        t3 = time.perf_counter()
        if code == "":
            continue
        item = {"rect": rect.tolist(), "det_score": round(score, 4), "marks": marks.tolist(),
                "layer": layer, "code": code, "rec_conf": round(rconf, 4),
                "crop_shape": list(crop.shape[:2])}
        if full:
            item["cls"] = [round(float(v), 4) for v in classify(cls, crop)]
            item["t_detect_ms"] = round((t1 - t0) * 1000, 2)
            item["t_rectify_ms"] = round((t2 - t1) * 1000, 2)
            item["t_recog_ms"] = round((t3 - t2) * 1000, 2)
        results.append(item)
    return results


def main():
    det = sess("y5fu_320x_sim.onnx")
    rec = sess("rpv3_mdict_160_r3.onnx")
    cls = sess("litemodel_cls_96x_r1.onnx")

    report = {"recognizer_on_labelled_crops": [], "full_images": []}

    # 1) recogniser sanity check against filenames carrying ground truth
    for p in sorted(SAMPLES.glob("crop-*.jpg")):
        gt = p.stem.split("-", 2)[-1]
        img = imread_u(p)
        code, conf = recognize(rec, img)
        row = {"file": p.name, "gt": gt, "pred": code, "conf": round(conf, 4), "match": code == gt}
        report["recognizer_on_labelled_crops"].append(row)
        print(f"[rec ] {p.name:28s} gt={gt:12s} pred={code:12s} conf={conf:.3f} {'OK' if code == gt else 'MISS'}")

    # 2) full pipeline on real photos
    for p in sorted(SAMPLES.glob("hlpr-*.jpg")):
        img = imread_u(p)
        res = run_pipeline(img, det, rec, cls, full=True)
        report["full_images"].append({"file": p.name, "size": list(img.shape[:2]), "plates": res})
        print(f"[pipe] {p.name:28s} {img.shape[1]}x{img.shape[0]} -> {len(res)} plate(s)")
        for r in res:
            print(f"        code={r['code']:12s} conf={r['rec_conf']:.3f} det={r['det_score']:.3f} "
                  f"layer={r['layer']} crop={r['crop_shape']} "
                  f"t_det={r.get('t_detect_ms')}ms t_rec={r.get('t_recog_ms')}ms cls={r.get('cls')}")

    (EVID / "hlpr_reference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", EVID / "hlpr_reference.json")


if __name__ == "__main__":
    main()
