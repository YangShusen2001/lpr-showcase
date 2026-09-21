# tools/ —— 脚本清单（按各脚本 docstring 首行整理；**新增脚本请顺手补一行**）

> 这些脚本是「每个数字都能当场复现」的凭证。改动手册见根 `README.md`。
> ⚠️ 本清单不是构建产物，不会自动更新 —— A16 那批脚本就漏登记过一轮。

## 资产获取

| 文件 | 说明 |
|---|---|
| `fetch_assets.py` | Fetch the real Chinese-plate test images shipped in the HyperLPR repository. Source: https://github.com/szad670401/HyperLPR (Apache-2.0), resource/ima |
| `fetch_more.py` | Fetch additional real car photos from the HyperLPR repository. |
| `fetch_upstream.py` | Fetch the remaining upstream HyperLPR test photos and score them for demo use. Downloads into assets/samples/upstream/ then runs the full pipeline on  |

## 模型转换

| 文件 | 说明 |
|---|---|
| `convert_bare_head.py` | Convert the bare-head y5fu to .ms, fp16 and fp32, for the NPU acceptance test. The question this answers is narrow and binary: once the 12 rank-5 cons |
| `convert_onnx_ms.py` | Generic ONNX -> MindSpore Lite .ms converter (fp16 + fp32). Same wrapper as tools/convert_bare_head.py, parameterised by a job table so new candidate  |
| `convert_to_ms.py` | -*- coding: utf-8 -*- |

## 保真对照与验证

| 文件 | 说明 |
|---|---|
| `_cmp_crop.py` | One-off: compute the rectified-crop RGB checksum from the Python reference. The native C++ port reports `cropSum` (sum of R,G,B over the rectified cro |
| `_verify_junction.ps1` | — |
| `compare_dumps.py` | Diff the browser port against the Python reference, field by field. Reads _evidence/web__verify_html.json (written by tools/web_check.mjs) and _eviden |
| `dump_det.py` | Dump the same intermediates as tools/_verify.html so the browser port can be diffed against the Python reference field by field. Writes _evidence/det_ |
| `dump_p7b.py` | Dump the printable text inside a HarmonyOS .p7b profile so we can see its structure (the embedded JSON is usually stored verbatim inside the PKCS#7).  |
| `dump_raw.py` | Dump raw end2end outputs (all rows, all confidences) for every sample x every model. |
| `dump_tensor.py` | Dump the detector's letterboxed input tensor as ground truth for the resize fitter. Writes, into _evidence/: tensor_src.rgba — the source image as raw |
| `hlpr_reference.py` | ---------------------------------------------------------------- detection |
| `verify_bare_head_pipeline.py` | Bare-head full-pipeline cross-check on PC (CPU ONNX). Answers ONE question: with the decode verified element-wise (tools/verify_head_decode.py, MATCH) |
| `verify_head_decode.py` | Verify a hand-written YOLOv5 decode reproduces the original model's output. The bare-head export (tools/export_bare_head.py) gets the detector onto th |
| `verify_junction.py` | -*- coding: utf-8 -*- |

## 真实数据集精度验收与真值复核（A16 / ADR-015）

| 文件 | 说明 |
|---|---|
| `lprnet_real_accuracy.py` | **精度门的入口脚本**：LPRNet vs 生产识别器 rpv3 的正面对照。输入是第三方真实人工标注裁剪图（`_dataset/real/crops`，文件名即真值，n=1000）→ rpv3 906/1000=90.6%、LPRNet 888/1000=88.8%、McNemar p=0.1788 |
| `lprnet_vs_rpv3.py` | ⚠️ 同一对照的**早期 n=5 版本**，其"真值"列写的是 rpv3 自己的输出 → 属循环论证，仅作历史留档，**结论一律以 `lprnet_real_accuracy.py` 为准** |
| `stretch_control.py` | 对照实验：水平拉伸会不会**凭空多出一个字符**？（n=200 已知 7 字符牌）→ 拉伸 0/200 新增字符、压缩 17% 吃字符，这是判定黄金图真值为 8 字符的关键一环 |
| `golden_decisive_checks.py` | 黄金图的三项决定性检查（几何 / 字符数 / 牌色），ADR-015 §1 的原始产出 |
| `glyph_count_geometry.py` | 用 GB 7258 / GA 36 的字符宽高（45 mm 字宽 + 12 mm 间隔）反推牌面上该有几个字符 |
| `plate_face_colour.py` | **牌面漆的正确量法**（取代有 bug 的分类器）：取 `S≥90 & 45≤V≤250` 的牌面像素，并用**车牌自己的白字**做白平衡对照 |
| `colour_classifier_audit.py` | 审查旧分类器：被它判成"绿牌"的 86 张 7 字符牌到底是不是真绿牌 |
| `plate_colour_analysis.py` | ⚠️ **DEPRECATED**：用"亮于中位数的像素"判色 → 蓝牌上最亮的是白字，把 86 张蓝牌判成绿牌。**按色细分数字（含"绿 hue 37.2%"）一律不得引用** |
| `lprnet_golden_check.py` | ⚠️ 判据是「must decode to exactly `苏ED5172`」—— 该串已被 A16 判为 rpv3 自身输出而非真值，**此脚本的 PASS/FAIL 不构成精度证据** |
| `lprnet_preproc_sweep.py` | 用 `GOLDEN OK` 选最优预处理（同属旧判据）；其选出的 stretch+BGR 组合后被官方 `_load_data.py` 独立证实，**结论存活、判据口径需按 ADR-015 重述** |
| `lprnet_shape_probe.py` | LPRNet 的 ONNX 输入/输出形状探针 |

## cv2.resize 定点拟合

| 文件 | 说明 |
|---|---|
| `fit_resize.mjs` | Empirical fitter: find the resize variant that reproduces cv2.resize(INTER_LINEAR) bit for bit. Reads the ground-truth tensor dumped by tools/dump_ten |
| `fit_rounding.py` | Brute-force the exact rounding layout of cv2.resize(INTER_LINEAR). Evidence gathered so far: * fixed-point coefficients with COEF_BITS=11 are exact fo |
| `fit_rounding2.py` | Measure cv2.resize(INTER_LINEAR)'s effective rounding threshold and pass order. Round-to-nearest explains 90.5% of pixels and floor 66.6%, so neither  |

## 论文数据与图

| 文件 | 说明 |
|---|---|
| `paper_fig_ramp.mjs` | Scale sweep for the paper's "P(difference = +1) vs fractional coordinate" claim. Imports the **shipped** assets/js/pipeline.js and calls its exported  |
| `paper_fig_ramp.py` | OpenCV side of the scale sweep behind the paper's "+1 LSB" ramp claim. Reads the port's resized buffers dumped by tools/paper_fig_ramp.mjs, reproduces |
| `paper_fig_ramp_diag.py` | Diagnostic: who deviates from exact bilinear -- the port, or cv2? Compares, for a few destination sizes, the port's dumped resize output against (a) c |
| `paper_fig_residual.mjs` | Residual-error data for the paper's fidelity figure. Compares the **shipped** assets/js/pipeline.js letterbox against the OpenCV ground-truth tensor d |
| `paper_figures.py` | Generate every figure used by the paper, in English and Chinese variants. Design rules: * Nothing is drawn from a hard-coded "expected" value. Every q |

## 页面自检与预览

| 文件 | 说明 |
|---|---|
| `_ort_smoke.html` | ORT smoke |
| `serve.mjs` | Minimal static file server for local verification and preview. Usage: node tools/serve.mjs <rootDir> [port] |
| `shot_el.mjs` | Element-level screenshot helper for visual QA. Usage: node tools/shot_el.mjs <pagePath> <selector> <outPng> [timeoutMs] |
| `web_check.mjs` | Headless-browser verification driver. Boots the local static server, opens a page in Chromium, waits for a result object the page publishes on window, |

## 工程辅助

| 文件 | 说明 |
|---|---|
| `a3_app_verify.py` | Build App, install only a fresh successful artifact, capture startup logs. |
| `a3_build_verify.py` | Run the approved ncnn Vulkan build and save UTF-8 evidence. |
| `build_paper.py` | Compile LaTeX sources with tectonic (no TeX distribution is installed). Invoked through Python because the shell's command scanner rejects the `tecton |
| `find_cwd_holder.py` | -*- coding: utf-8 -*- |
| `gen_plate_dataset.py` | Synthetic Chinese license-plate recognition dataset generator. Renders plates with real Chinese fonts, applies photographic augmentation (perspective  |
| `list_upstream.py` | List candidate test images in the upstream HyperLPR repository. Keeps the demo's provenance consistent: the ONNX models already come from szad670401/H |
| `make_ascii_junction.py` | -*- coding: utf-8 -*- |
| `make_ids_breakable.py` | Make long typewriter identifiers breakable so they stop overflowing margins. Why this is needed ------------------ The TeX logs show the real defect b |
| `promote_samples.py` | Evaluate the fetched upstream images, record the verdict as UTF-8 JSON, and promote the ones worth demoing into assets/samples/ under ASCII names. Run |
| `read_profile.py` | Extract the JSON payload embedded in HarmonyOS .p7b provisioning profiles. A HarmonyOS debug/release profile is a PKCS#7 (DER) container whose content |
| `summarize_profiles.py` | Summarise every HarmonyOS .p7b provisioning profile in a directory. Prints, per profile: bundle-name, app-identifier, type, uuid, validity window (hum |
| `sync_rawfile.py` | -*- coding: utf-8 -*- |
| `viz_reference.py` | Render the reference-pipeline result so a human can eyeball it. Produces _evidence/viz_<name>.png : left = source with box + 4 corner points, right =  |

## 其他

| 文件 | 说明 |
|---|---|
| `_verify.html` | Raw detector row carrying the strongest objectness, for numeric diffing vs Python. |
| `det_head_ref_checksum.py` | PC reference for the CPU-diff protocol on the bare-head detector. Mirrors MsBenchRun's protocol exactly (ms_engine.cpp): input fill x[j] = ((j * 26544 |
| `export_bare_head.py` | Export a bare-head y5fu: cut the anchor-grid decode out of the graph. Why: the NPU rejects y5fu because of 12 rank-5 CONSTANTS (ADR-004 §4.1), and tho |
| `export_decode_stages.py` | Locate the exact decode step by exporting intermediate tensors as graph outputs. The first decode attempt (tools/verify_head_decode.py) mismatched, an |
| `fix_lprnet_perm.py` | Rewrite LPRNet's six Transpose[0,3,2,1] away so the model can reach the NPU. Why: the NPU only accepts Transpose perm == [0,1,3,2] (`permute_matmul_fu |
| `inspect_onnx_ops.py` | Dump 一个 ONNX 模型的算子直方图 / 输入输出形状 / 动态维度 —— 用于**判断能否转 ncnn**（例：rec 的 `Shape`/`Slice` 在固定输入形状下被 pnnx 常量折叠，这决定了识别模型能不能上 GPU） |
| `inspect_hlpr.py` | Inspect the HyperLPR3 ONNX models: exact input/output specs. |
| `inspect_model.py` | Inspect the downloaded ONNX license-plate detectors: IO spec + a real inference run. Run with the system Python that already has onnxruntime + numpy + |
| `inspect_onnx_for_ms.py` | -*- coding: utf-8 -*- |
| `lprnet_shape_probe.py` | Reveal the true shapes flowing through LPRNet's Transpose/MaxPool pairs. The rewrite in tools/fix_lprnet_perm.py assumed the [0,3,2,1] transpose puts  |
| `ncnn_ref_check.py` | PC reference for the ncnn port: same image, same NCHW layout, ONNX runtime. The device-side ncnn run (lpr.ncnnRun via hilog `NCNN RUN`) prints the L2/ |
| `npu_op_anatomy.py` | Dissect the NPU-unfriendly ops in the two hard models. Two different failure modes are in play and they must not be conflated: y5fu_320x_sim -> reject |
| `npu_scan.py` | NPU fitness scanner for ONNX models (Kirin 8020). Answers "can this model run on the NPU?" from two hard gates derived from the NPU compiler's own che |
| `pdf_render_check.py` | Verify that a compiled PDF really contains the expected CJK glyphs. Motivation: a LaTeX run can *succeed* while silently dropping CJK characters when  |
| `probe_blocked.mjs` | One-off diagnostic: under the COOP/COEP headers added in ADR-004 §8, some subresources on index.html get blocked with ERR_BLOCKED_BY_RESPONSE.NotSameO |
| `probe_kernel.py` | Recover cv2.resize(INTER_LINEAR)'s effective filter by impulse response. Feeding a single non-zero pixel makes the rounding irrelevant and exposes the |
| `probe_kernel2.py` | 2-D impulse probe: isolate the vertical pass of cv2.resize(INTER_LINEAR). The horizontal pass already matches (see tools/probe_kernel.py), yet the ful |
| `probe_onnx_io.py` | -*- coding: utf-8 -*- |
| `probe_pixel.py` | Zoom into the exact pixels where the port disagrees with cv2.resize. Prints the four source values, their fixed-point weights, and the result produced |
| `probe_resize.py` | Probe cv2.resize(INTER_LINEAR) on a 2-D downscale to recover the exact algorithm. The fixed-point variant fitted on hlpr-test.jpg (scale exactly 6.0)  |
| `probe_truncation.py` | Probe the keypoint-truncation defect described in ADR-002 §2.2. Claim under test: measuring the rectification quad from *float* keypoints (instead of  |

## tools/harmony/ —— PC ↔ 手机协同

| 文件 | 说明 |
|---|---|
| `harmony/pc_drive_phone.mjs` | PC-side driver for the phone's WebView — the transport half of "computer orchestrates, phone computes". Why this works at all: ArkWeb is Chromium, and |
| `harmony/run_mt_sweep.py` | WASM 多线程扫描的采集脚本：装机 → 冷启 → 轮询抓 LprWeb 日志。 与 run_matrix_web.py 的两点区别： 1. 轮询抓取而不是等满再一次性 dump —— 扫描要跑 2~3 分钟，hilog 是环形缓冲， 一次性 dump 时前面的 MT 行可能已经被系统日志挤掉。每  |
| `harmony/webprobe.html` | GPU / WASM 能力探针 |
