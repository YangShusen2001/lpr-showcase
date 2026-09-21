# 归档 · 2026-09-18 项目整理

> 本次整理把「历史备份 + 实验产物 + 断链资产」从项目主体移出，**项目有效内容 144 MB → 38.8 MB**。
> 归档不是删除：所有内容原样保留在本目录，随时可移回。

## 1. 归档了什么

| 子目录 | 内容 | 体积 | 为什么移出 |
|---|---|---|---|
| `dead-assets/ort/` | `ort-wasm-simd-threaded.jsep.{wasm,mjs}` | 26.6 MB | `assets/README.md` 明写「WebGPU/JSEP 变体，本项目**不用**」（走 wasm provider）。JSEP/ORT-Web WebGPU 路线已判死（ADR-008） |
| `dead-assets/models/` | 三个 `yolo-v9-t-*-end2end.onnx` + `y5fu_640x_sim.onnx` | 26.0 MB | ADR-001 的**证伪记录资产**，不参与运行时；判据与结论已写进 ADR-001 + `_evidence/model_probe.json` |
| `intermediate-dumps/` | `tensor_*.f32` / `tensor_*.rgba` / `ramp/port_resized.bin` | 33.1 MB | 中间张量 dump，可由 `tools/dump_*.py`、`tools/fit_resize.mjs` 重生成 |
| `big-logs/` | `a3_device_full.log` | 7.8 MB | 7.8 MB 全量设备日志；结论已落在 `_evidence/A3-verification-result.txt` |
| `screenshots/` | 探针图 / QA 截图 / 渲染图 / 可视化图（27 项） | 6.5 MB | 过程截图，可由 `tools/web_check.mjs`、`tools/shot_el.mjs` 重生成 |
| `latex-probe/` | `paper/latex-*/_probe.*`（12 项） | 0.2 MB | xeCJK 字体排障用的最小复现例，已完成使命 |
| `latex-backups/` · `latex-preview/` | `*.bak-2026*` · `main_p*.png` | 3.8 MB | 编译前备份与逐页渲染校验截图 |
| `backup-files/` · `old-backups/` | `_backup/`（旧 index.html）、`backup-backendprobe-*` | 0.1 MB | 历史版本备份 |
| `upstream-samples/` | `assets/samples/upstream/`（6 张上游原图） | 2.4 MB | 其中 4 张已提升为 `assets/samples/*.jpg` 并在页面使用；上游可重新 clone |

**同时直接删除的（零价值，未进归档）**：`tools/__pycache__/`、LaTeX 编译日志 `*.build.log` / `main.log`、
0 字节文件 `a3_device_summary.txt`、空目录 `harmony/`、与 256 档逐字节重复的 3 张 640 档探针图。

## 2. 怎么恢复

```bash
# 单个文件恢复（示例：把 jsep.wasm 移回）
mv _archive/2026-09-18/dead-assets/ort/ort-wasm-simd-threaded.jsep.wasm assets/ort/

# 整组恢复
cp -r _archive/2026-09-18/latex-probe/* paper/latex-en/     # 按需，注意 en/zh 归属
```

⚠️ 移回 `dead-assets/` 的东西**不影响任何在跑的代码**（它们本来就不被引用）；
但若移回 `upstream-samples/`，注意 `assets/samples/` 与它的 4 个重复文件，别让两份同内容图各自漂移。

## 3. 怎么彻底释放磁盘

确认不再需要后：

```bash
rm -rf _archive/2026-09-18        # 释放约 105 MB
```

需要重新下载的东西（有成本，故归档而非删除）：
- `jsep.wasm` —— onnxruntime-web 的 npm 包内（当前 1.29.0）
- `yolo-v9-t-*.onnx` —— 需 VPN 从上游源取（见 `tools/fetch_upstream.py` 同族脚本）

## 4. 整理时确认过的「看着像垃圾但还活着」的东西

| 文件 | 为什么保留 |
|---|---|
| `demo.html` | 独立的实时演示页，`assets/README.md` §4 自检清单仍在跑它 |
| `mobile.html` | 手机端演示页，`assets/js/mobile.js` + `mobile.css` 专用 |
| `assets/models/*.onnx.json` | **运行时模型**（后缀是静态托管白名单逼出来的），删掉 Demo 就死 |
| `paper/figures/*/*.pdf` | LaTeX 直接引用的图源，不删 |
| `_evidence/*.json` | 论文数字的原始出处 |