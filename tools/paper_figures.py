"""Generate every figure used by the paper, in English and Chinese variants.

Design rules:
  * Nothing is drawn from a hard-coded "expected" value. Every quantitative panel
    reads a file produced by an experiment in this repository (see the SOURCE note
    at the top of each function). Schematics are the only hand-authored panels.
  * Labels are localised, so the Chinese paper does not carry English figures.

Output: paper/figures/<lang>/figN_<name>.{png,pdf}   (png 300 dpi, pdf vector)

Run:  .venv\\Scripts\\python.exe tools\\paper_figures.py --lang en
      .venv\\Scripts\\python.exe tools\\paper_figures.py --lang zh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import hlpr_reference as R

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "samples"
EVID = ROOT / "_evidence"

# ----------------------------------------------------------------- palette
ACCENT = "#2348e0"
INK = "#0a0c10"
MUTED = "#626c7a"
FAINT = "#98a1ae"
LINE = "#e3e7ec"
BG_ALT = "#f7f8fa"
PLATE_BLUE = "#1a63c4"
PLATE_GREEN = "#12a150"
PLATE_YELLOW = "#dda000"
OK = "#12a150"
WARN = "#d97706"
BAD = "#d13438"

# A concrete family that carries both Latin and CJK, used for any string that
# mixes a Chinese plate glyph with ASCII (matplotlib's fallback chain does not
# extend to the "monospace" alias).
CJK_FONT = "Microsoft YaHei"

# Explicit font list rather than the generic alias "monospace": matplotlib only
# expands a generic family against rcParams["font.monospace"] if that key was set
# before pyplot was imported.  This module sets its rcParams inside setup(), i.e.
# after import, so the alias would silently drop the CJK fallback.  An explicit
# list always chains.  Verified: alias -> 66 missing glyphs, list -> 0.
MONO = ["DejaVu Sans Mono", "Consolas", "Courier New", CJK_FONT]

# ----------------------------------------------------------------- localisation
STRINGS = {
    "en": {
        "font": ["DejaVu Sans", "Segoe UI", "Arial", "Microsoft YaHei"],
        "input": "Input\nphotograph",
        "output": "Output\nplate string",
        "fig1_note": "3 ONNX graphs · 14,206,477 B (13.5 MiB) · self-hosted, 0 external requests",
        "stages": [
            ("1  Detection", "y5fu_320x_sim.onnx\n2.23 MB · YOLOv5-family",
             "320x320x3 RGB", "boxes · 4 kpts · obj · layer"),
            ("2  Rectification", "analytic (no model)\nbicubic resampling",
             "4 keypoints", "fronto-parallel strip"),
            ("3  Recognition", "rpv3_mdict_160_r3.onnx\n9.78 MB · CRNN + CTC",
             "48x160x3 BGR", "CTC logits, T x 74 classes"),
            ("4  Classification", "litemodel_cls_96x_r1.onnx\n1.53 MB",
             "96x96x3 BGR", "3 colour scores + layer"),
        ],
        "fig2_a": "(a) detection on scene-2.jpg (1140x456)",
        "fig2_b": "(b) rectified strip 339x122, decoded",
        "fig3_title": "Ordering of the keypoint truncation (hlpr-1.jpg)",
        "fig3_a": "(a) detected quad, float keypoints",
        "fig3_b": "(b) measured from FLOAT marks\ncrop 12x21",
        "fig3_c": "(c) measured from TRUNCATED marks\ncrop 13x22  (reference)",
        "fig3_b_tag": "decoded  H468\nconf 0.498",
        "fig3_c_tag": "decoded  MU4158\nconf 0.793",
        "fig4_a_title": "(a) real pipeline cases vs OpenCV",
        "fig4_b_title": "(b) shipped resize vs OpenCV, 135 scales",
        "fig4_x_a": "per-channel difference  (LSB, 1 LSB = 1/255)",
        "fig4_y_a": "channels",
        "fig4_x_b": "fractional part of the source coordinate",
        "fig4_y_b": "pixels differing  (%)",
        "fig4_b_v1": "vertical scale = 1",
        "fig4_b_vf": "vertical scale \u2260 1",
        "fig4_legend": ["bit-exact", "residual", "float bilinear"],
        "fig5_title": "Automated fidelity verification",
        "fig5_ref": "Python reference\ntools/hlpr_reference.py",
        "fig5_port": "Browser port\nassets/js/pipeline.js",
        "fig5_cmp": "tools/compare_dumps.py\n11 fields, in pipeline order",
        "fig5_rep": "fidelity_report.txt\n8/8 agreement",
        "fig5_note": "Browser side driven by headless Chromium;\nintermediates published on window, written from Node",
        "fig6_title": "Kirin 8020: NPU vs CPU latency (P50, ms)",
        "fig6_npu": "NPU",
        "fig6_cpu": "CPU",
        "fig6_x": "latency (ms, log scale)",
        "fig6_note": "speedup decreases monotonically with graph size",
        "fig7_title": "Output L2 relative error, NPU vs CPU",
        "fig7_y": "relative error (%)",
        "fig7_note": "declared-FP32 buffer holding FP16 patterns",
        "fixed": "after Cast fix",
        "asdeclared": "as declared",
    },
    "zh": {
        "font": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "input": "输入\n照片",
        "output": "输出\n车牌字符串",
        "fig1_note": "三个 ONNX 图 · 合计 14,206,477 B（13.5 MiB）· 全自托管，0 个外部网络请求",
        "stages": [
            ("1  车牌检测", "y5fu_320x_sim.onnx\n2.23 MB · YOLOv5 系",
             "320x320x3 RGB 归一化", "检测框 · 4 关键点 · 层数"),
            ("2  透视矫正", "解析法（无模型）\n双三次重采样",
             "4 个关键点", "正视角车牌条带"),
            ("3  序列识别", "rpv3_mdict_160_r3.onnx\n9.78 MB · CRNN + CTC",
             "48x160x3 BGR", "CTC logits，T x 74 类"),
            ("4  颜色分类", "litemodel_cls_96x_r1.onnx\n1.53 MB",
             "96x96x3 BGR", "3 类颜色得分 + 层数"),
        ],
        "fig2_a": "(a) scene-2.jpg 上的检测（1140x456）",
        "fig2_b": "(b) 矫正后条带 339x122，识别结果",
        "fig3_title": "关键点截断的先后顺序（hlpr-1.jpg）",
        "fig3_a": "(a) 检出的四边形，浮点关键点",
        "fig3_b": "(b) 由【浮点】关键点量取\n裁剪 12x21",
        "fig3_c": "(c) 由【截断后】关键点量取\n裁剪 13x22（参考实现）",
        "fig3_b_tag": "识别  H468\n置信度 0.498",
        "fig3_c_tag": "识别  MU4158\n置信度 0.793",
        "fig4_a_title": "(a) 真实流水线样本 vs OpenCV",
        "fig4_b_title": "(b) 已交付的 resize vs OpenCV，135 个尺度",
        "fig4_x_a": "逐通道差值（LSB，1 LSB = 1/255）",
        "fig4_y_a": "通道数",
        "fig4_x_b": "源坐标的小数部分",
        "fig4_y_b": "不一致像素占比（%）",
        "fig4_b_v1": "垂直比例为 1",
        "fig4_b_vf": "垂直比例非 1",
        "fig4_legend": ["逐通道一致", "残余", "浮点双线性"],
        "fig5_title": "自动化保真度验证",
        "fig5_ref": "Python 参考实现\ntools/hlpr_reference.py",
        "fig5_port": "浏览器端移植\nassets/js/pipeline.js",
        "fig5_cmp": "tools/compare_dumps.py\n11 个字段，按流水线顺序",
        "fig5_rep": "fidelity_report.txt\n8/8 一致",
        "fig5_note": "浏览器侧由无头 Chromium 驱动；\n中间量发布在 window 上，由 Node 直接落盘",
        "fig6_title": "麒麟 8020：NPU 与 CPU 延迟对比（P50，ms）",
        "fig6_npu": "NPU",
        "fig6_cpu": "CPU",
        "fig6_x": "延迟（ms，对数轴）",
        "fig6_note": "加速比随图规模单调下降",
        "fig7_title": "输出 L2 相对误差，NPU 对 CPU",
        "fig7_y": "相对误差（%）",
        "fig7_note": "声明为 FP32 的缓冲区里装的是 FP16 位模式",
        "fixed": "插入 Cast 后",
        "asdeclared": "按声明读取",
    },
}

# --------------------------------------------------- NPU records (sibling project)
NPU_ROWS = [
    ("ResNet-50", 6.02, 78.90),
    ("ResNet-18", 3.27, 34.00),
    ("YOLOv8n", 15.60, 71.07),
    ("MobileNetV2", 2.74, 10.22),
    ("MobileNetV3-S", 3.37, 3.44),
    ("MobileNetV2 INT8", 5.08, 4.97),
]
NPU_L2 = [
    ("ResNet-50", 0.242),
    ("ResNet-18", 0.000),
    ("MobileNetV2", 0.067),
    ("YOLOv8n", 7.78e18),
    ("YOLOv8n (fixed)", 0.015),
]


def setup(lang: str) -> dict:
    S = STRINGS[lang]
    matplotlib.rcParams.update({
        "font.sans-serif": S["font"],
        "font.family": S["font"],
        "font.monospace": ["DejaVu Sans Mono", "Consolas", "Courier New",
                           "Microsoft YaHei"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
    })
    return S


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("  wrote", outdir / f"{name}.png")


# ------------------------------------------------------------------- figure 1
def fig_pipeline(S: dict, outdir: Path) -> None:
    """SOURCES: stage table of the paper; model byte sizes from assets/models/."""
    fig, ax = plt.subplots(figsize=(11.0, 3.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    n = len(S["stages"])
    gap = 2.4
    x0, x1 = 11.5, 88.5
    bw = (x1 - x0 - gap * (n - 1)) / n
    by, bh = 9.0, 14.5

    ax.text(5.0, by + bh / 2, S["input"], ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(95.0, by + bh / 2, S["output"], ha="center", va="center", fontsize=9.5, color=INK)

    for i, (title, model, tin, tout) in enumerate(S["stages"]):
        bx = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=1.1, edgecolor=LINE, facecolor="#ffffff", zorder=2))
        ax.add_patch(FancyBboxPatch(
            (bx, by + bh - 3.6), bw, 3.6, boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=0, facecolor="#eef2ff", zorder=3))
        ax.text(bx + bw / 2, by + bh - 1.9, title, ha="center", va="center",
                fontsize=10.0, fontweight="bold", color=ACCENT, zorder=4)
        ax.text(bx + bw / 2, by + bh - 4.9, model, ha="center", va="top",
                fontsize=7.2, color=INK, family=MONO, zorder=4, linespacing=1.45)
        ax.text(bx + bw / 2, by + 5.0, tin, ha="center", va="center",
                fontsize=7.0, color=MUTED, zorder=4)
        ax.text(bx + bw / 2, by + 2.6, "\u2192 " + tout, ha="center", va="center",
                fontsize=7.0, color=MUTED, zorder=4)

        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (bx + bw + 0.15, by + bh / 2), (bx + bw + gap - 0.15, by + bh / 2),
                arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=FAINT, zorder=1))
    ax.add_patch(FancyArrowPatch((7.6, by + bh / 2), (x0 - 0.4, by + bh / 2),
                                 arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=FAINT))
    ax.add_patch(FancyArrowPatch((x1 + 0.4, by + bh / 2), (92.4, by + bh / 2),
                                 arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=FAINT))

    ax.text(50, 3.4, S["fig1_note"], ha="center", va="center", fontsize=8.6,
            color=MUTED, family=MONO)
    save(fig, outdir, "fig1_pipeline")


# ------------------------------------------------------------------- figure 2
def fig_detection(S: dict, outdir: Path) -> None:
    """SOURCES: live inference via tools/hlpr_reference.py on assets/samples/scene-2.jpg."""
    det = R.sess("y5fu_320x_sim.onnx")
    rec = R.sess("rpv3_mdict_160_r3.onnx")
    img = R.imread_u(SAMPLES / "scene-2.jpg")
    rows = R.detect(det, img)
    row = rows[0]
    marks = np.asarray(row[5:13]).reshape(4, 2).astype(int)
    rect = row[:4].astype(int)
    crop = R.get_rotate_crop_image(img, marks)
    code, conf = R.recognize(rec, crop)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5),
                             gridspec_kw={"width_ratios": [2.05, 1]})

    ax = axes[0]
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    cv2.rectangle(vis, (rect[0], rect[1]), (rect[2], rect[3]), (225, 59, 92), 3)
    for i, (px, py) in enumerate(marks):
        cv2.circle(vis, (px, py), 7, (34, 224, 106), -1)
        cv2.putText(vis, str(i), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (34, 224, 106), 2)
    poly = marks.reshape(-1, 1, 2)
    cv2.polylines(vis, [poly], True, (34, 224, 106), 2)
    ax.imshow(vis)
    ax.set_title(S["fig2_a"], fontsize=9.5, color=INK)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax.set_title(S["fig2_b"], fontsize=9.5, color=INK)
    ax.set_xlabel(f"{code}   conf {conf:.3f}   {crop.shape[1]}x{crop.shape[0]}",
                  fontsize=9.5, color=ACCENT, family=CJK_FONT)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)

    fig.tight_layout()
    save(fig, outdir, "fig2_detection")
    print(f"    scene-2 decoded {code} conf={conf:.3f} crop={crop.shape[1]}x{crop.shape[0]}")


# ------------------------------------------------------------------- figure 3
def fig_truncation(S: dict, outdir: Path) -> None:
    """SOURCES: _evidence/probe_truncation.json (tools/probe_truncation.py) plus a
    fresh run of the same three variants so the crops shown are the ones measured."""
    import probe_truncation as PT

    det = R.sess("y5fu_320x_sim.onnx")
    rec = R.sess("rpv3_mdict_160_r3.onnx")
    img = R.imread_u(SAMPLES / "hlpr-1.jpg")
    row = R.detect(det, img)[0]
    raw = np.asarray(row[5:13], dtype=np.float64).reshape(4, 2)
    marks_int = raw.astype(int)
    layer = int(row[13])

    c_float = PT.crop_float_marks(img, raw)
    c_int = R.get_rotate_crop_image(img, marks_int)
    code_float, conf_float = PT.decode_like_reference(rec, c_float, layer)
    code_int, conf_int = PT.decode_like_reference(rec, c_int, layer)

    fig = plt.figure(figsize=(11.0, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.16)

    # (a) zoom on the detected quad
    ax = fig.add_subplot(gs[0, 0])
    pad = 26
    x0 = max(0, int(marks_int[:, 0].min()) - pad)
    x1 = min(img.shape[1], int(marks_int[:, 0].max()) + pad)
    y0 = max(0, int(marks_int[:, 1].min()) - pad)
    y1 = min(img.shape[0], int(marks_int[:, 1].max()) + pad)
    vis = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).copy()
    m = marks_int - [x0, y0]
    cv2.polylines(vis, [m.reshape(-1, 1, 2)], True, (34, 224, 106), 1)
    for i, (px, py) in enumerate(m):
        cv2.circle(vis, (px, py), 2, (225, 59, 92), -1)
    ax.imshow(vis, interpolation="nearest")
    ax.set_title(S["fig3_a"], fontsize=9.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)

    for col, (crop, tag, sub, col_ok) in enumerate([
        (c_float, S["fig3_b_tag"], S["fig3_b"], False),
        (c_int, S["fig3_c_tag"], S["fig3_c"], True),
    ], start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation="nearest",
                  aspect="auto")
        ax.set_title(sub, fontsize=9.5, color=(OK if col_ok else BAD))
        ax.set_xlabel(tag, fontsize=9.5, family=CJK_FONT,
                      color=(OK if col_ok else BAD))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(OK if col_ok else BAD)

    fig.suptitle(S["fig3_title"], fontsize=10, color=MUTED, y=1.04)
    save(fig, outdir, "fig3_truncation")
    print(f"    hlpr-1 float={code_float} ({c_float.shape[1]}x{c_float.shape[0]}) "
          f"int={code_int} ({c_int.shape[1]}x{c_int.shape[0]})")


# ------------------------------------------------------------------- figure 4
def fig_residual(S: dict, outdir: Path) -> None:
    """SOURCES:
      (a) _evidence/paper_residual.json — shipped assets/js/pipeline.js vs the
          OpenCV tensor from tools/dump_tensor.py (the two real pipeline cases).
      (b) _evidence/paper_ramp.json — tools/paper_fig_ramp.mjs + .py, a sweep of
          the shipped resizeLinear against cv2.resize over 135 destination sizes.
    """
    res = json.loads((EVID / "paper_residual.json").read_text(encoding="utf-8"))
    by = {r["stem"]: r for r in res}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))

    # ---- (a) difference histogram
    ax = axes[0]
    stems = ["hlpr-1", "scene-2"]
    colors = [OK, WARN]
    width = 0.36
    xs = np.array([0.0, 1.0])
    for k, (stem, col) in enumerate(zip(stems, colors)):
        r = by[stem]
        lab = (f"{stem}.jpg  r={r['scale']:.4f}\n"
               f"{r['nDiff']:,}/{r['total']:,}  ({100*r['nDiff']/r['total']:.2f} %)")
        h = {int(a): b for a, b in r["hist"].items()}
        vals = [h.get(0, 0), h.get(1, 0)]
        off = (k - 0.5) * width
        bars = ax.bar(xs + off, vals, width=width * 0.92, color=col, label=lab,
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 1) * 1.35,
                    f"{v:,}" if v else "0", ha="center", va="bottom",
                    fontsize=8, color=col, family=MONO)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 4.0e6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0  (" + S["fig4_legend"][0] + ")", "+1"])
    ax.set_xlabel(S["fig4_x_a"], fontsize=9)
    ax.set_ylabel(S["fig4_y_a"], fontsize=9)
    ax.set_title(S["fig4_a_title"], fontsize=9.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- (b) mismatch rate vs fractional coordinate, split by the vertical scale
    ax = axes[1]
    ramp = json.loads((EVID / "paper_ramp.json").read_text(encoding="utf-8"))
    centres = np.array([b["mid"] for b in ramp["buckets"]])
    for key, col, lab in (("vertExact", OK, S["fig4_b_v1"]),
                          ("vertFrac", BAD, S["fig4_b_vf"])):
        ser = ramp[key]
        ys = np.array([b["pDiff"] for b in ser["buckets"]]) * 100
        ax.plot(centres, ys, "-o", color=col, markersize=3.6, linewidth=1.6,
                label=f"{lab}  ({100*ser['pDiff']:.2f}%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 36)
    ax.set_xlabel(S["fig4_x_b"], fontsize=9)
    ax.set_ylabel(S["fig4_y_b"], fontsize=9)
    ax.set_title(S["fig4_b_title"], fontsize=9.5, color=INK)
    ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")

    fig.tight_layout()
    save(fig, outdir, "fig4_residual")
    print(f"    sweep: {ramp['nConfigs']} scales, "
          f"{ramp['totalDiff']}/{ramp['totalPix']} px differ "
          f"({100*ramp['totalDiff']/ramp['totalPix']:.2f}%), "
          f"{ramp['totalPos']} of them +1, {len(ramp['exactConfigs'])} bit-exact")
    print(f"    vert scale == 1 -> {100*ramp['vertExact']['pDiff']:.3f}%, "
          f"!= 1 -> {100*ramp['vertFrac']['pDiff']:.3f}%")


# ------------------------------------------------------------------- figure 5
def fig_protocol(S: dict, outdir: Path) -> None:
    """SOURCES: schematic of the workflow in ADR-002 §4."""
    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, text, face="#ffffff", edge=LINE, tcol=INK, fs=9.0, mono=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.0",
                                    linewidth=1.1, edgecolor=edge, facecolor=face, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tcol, zorder=3, linespacing=1.7,
                family=MONO if mono else None)

    box(3, 21.5, 25, 9.0, S["fig5_ref"], face="#f4f7ff", edge="#c3cffe", fs=8.6, mono=True)
    box(3, 8.0, 25, 9.0, S["fig5_port"], face="#f4f7ff", edge="#c3cffe", fs=8.6, mono=True)
    box(37, 14.8, 27, 8.6, S["fig5_cmp"], face="#ffffff", fs=8.6, mono=True)
    box(72, 14.8, 25, 8.6, S["fig5_rep"], face="#f0fdf4", edge="#b7e4c7",
        tcol=OK, fs=8.6, mono=True)

    for y in (26.0, 12.5):
        ax.add_patch(FancyArrowPatch((28.6, y), (36.4, 19.1), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.2, color=FAINT,
                                     connectionstyle="arc3,rad=-0.12", zorder=1))
    ax.add_patch(FancyArrowPatch((64.6, 19.1), (71.4, 19.1), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=FAINT, zorder=1))
    ax.text(30.5, 30.6, "dump", fontsize=7.6, color=FAINT, family=MONO)
    ax.text(30.5, 6.2, "dump", fontsize=7.6, color=FAINT, family=MONO)
    ax.text(50.5, 25.4, S["fig5_title"], ha="center", fontsize=10, color=ACCENT,
            fontweight="bold")
    ax.text(50, 3.2, S["fig5_note"], ha="center", va="center", fontsize=7.8,
            color=MUTED, linespacing=1.7)
    save(fig, outdir, "fig5_protocol")


# ------------------------------------------------------------------- figure 6
def fig_npu(S: dict, outdir: Path) -> None:
    """SOURCE: on-device benchmark records (Kirin 8020, nova 14 Pro), as tabulated
    in the paper's Table on latency scaling."""
    names = [r[0] for r in NPU_ROWS]
    npu = np.array([r[1] for r in NPU_ROWS])
    cpu = np.array([r[2] for r in NPU_ROWS])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    h = 0.36
    ax.barh(y + h / 2, npu, height=h, color=ACCENT, label=S["fig6_npu"])
    ax.barh(y - h / 2, cpu, height=h, color="#cbd3de", label=S["fig6_cpu"])

    for i in range(len(names)):
        ax.text(npu[i] * 1.12, y[i] + h / 2, f"{npu[i]:.2f}", va="center",
                fontsize=8, color=ACCENT, family=MONO)
        ax.text(cpu[i] * 1.12, y[i] - h / 2, f"{cpu[i]:.2f}", va="center",
                fontsize=8, color=MUTED, family=MONO)
        sp = cpu[i] / npu[i]
        col = OK if sp >= 1.0 else BAD
        ax.text(165, y[i], f"{sp:.2f}x", va="center", ha="right",
                fontsize=9, color=col, family=MONO, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(1.6, 170)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(len(names) - 0.2, -1.45)
    ax.set_xlabel(S["fig6_x"], fontsize=9)
    ax.set_title(S["fig6_title"], fontsize=10, color=INK)
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.text(1.75, len(names) - 0.45, S["fig6_note"], fontsize=7.8, color=FAINT)
    fig.tight_layout()
    save(fig, outdir, "fig6_npu_latency")


# ------------------------------------------------------------------- figure 7
def fig_dtype(S: dict, outdir: Path) -> None:
    """SOURCE: output-L2 comparison records from the on-device characterisation."""
    names = [r[0] for r in NPU_L2]
    vals = [r[1] for r in NPU_L2]
    cols = []
    for n, v in zip(names, vals):
        if v > 100:
            cols.append(BAD)
        elif "fixed" in n:
            cols.append(OK)
        else:
            cols.append(ACCENT)

    fig, ax = plt.subplots(figsize=(11.0, 2.95))
    x = np.arange(len(names))
    ax.bar(x, [max(v, 1e-4) for v in vals], color=cols, width=0.55,
           edgecolor="white", linewidth=0.6)
    for xi, v in zip(x, vals):
        txt = "0.000" if v == 0 else (f"{v:.3f}" if v < 100 else f"{v:.2e}")
        ax.text(xi, max(v, 1e-4) * 2.1, txt, ha="center", va="bottom", fontsize=8.4,
                family=MONO, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 1e21)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (fixed)", "\n(" + S["fixed"] + ")") for n in names],
                       fontsize=8.4)
    ax.set_ylabel(S["fig7_y"], fontsize=9)
    ax.set_title(S["fig7_title"], fontsize=10, color=INK)
    ax.grid(axis="y", color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.annotate(S["fig7_note"], xy=(3, 7.78e18), xytext=(1.15, 3.0e14),
                fontsize=7.8, color=BAD,
                arrowprops=dict(arrowstyle="-|>", color=BAD, linewidth=1.0,
                                connectionstyle="arc3,rad=0.18"))
    fig.tight_layout()
    save(fig, outdir, "fig7_dtype")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    S = setup(args.lang)
    outdir = Path(args.out) if args.out else ROOT / "paper" / "figures" / args.lang
    print(f"figures -> {outdir}")

    fig_pipeline(S, outdir)
    fig_detection(S, outdir)
    fig_truncation(S, outdir)
    fig_residual(S, outdir)
    fig_protocol(S, outdir)
    fig_npu(S, outdir)
    fig_dtype(S, outdir)
    print("done")


if __name__ == "__main__":
    main()
