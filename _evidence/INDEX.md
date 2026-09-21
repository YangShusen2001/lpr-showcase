# _evidence/ —— 原始证据索引

> 这里的每一个数字都能追到文件。**论文/简历里的数字，只允许引这里的产物。**
> 目录本身按「采集批次」平铺（脚本里硬编码了路径，**不要重排子目录**）。

## 0. 三条全局口径声明（2026-09-18，ADR-015；读任何条目之前先看这里）

1. **凡本索引中出现的 `苏ED5172`，均指 rpv3 在默认裁剪几何下的输出（移植保真参照串），
   不是人工真值。** A16 §4.6 判定该样本正确真值为 8 字符 `苏ED51712`。
   因此 A4/A5/A6/A8/A15 里的 `match=1/0`、`111/000` 一律读作
   **「与 CPU 基线一致 / 不一致」**，不得读作「对 / 错」。原始日志不回改。
2. **精度只引 A16**（`_dataset/real/crops` n=1000，文件名即人工真值）：
   rpv3 90.6% / LPRNet 88.8% / McNemar p=0.1788 / 长度错误 1.0% vs 7.0%。
   任何 n=5 的"GOLDEN OK"结论（`lprnet_vs_rpv3.json` 等）都属旧判据，不构成精度证据。
3. **A14 §4.2.2「相机出帧率 15 fps = 上限、推理侧已彻底不是瓶颈」已撤回待复测**：
   该"零推理基准档"仍支付整帧 NV21→RGBA 转换，且与 A14 §4.1 同一暗场景的
   `arrive=19.92–20.06` 自相矛盾。`34.5 / 23.8 / 12.8 fps` 是**服务时间理论值**，
   不得对外称 fps；真实吞吐只看到达率 / 完成率。

> ⚠️ 另：本目录所有端侧时延，除非日志行带 `build=release`，均为 **debug 构建**（`-O0`）
> 口径 —— 内部可比，**不可与 release 数字并列**。

## 1. 结论级证据（答辩/面试可直接展示）

| 文件 | 支撑的主张 |
|---|---|
| `A16-real-dataset-accuracy-20260918.md` | **真实第三方数据集上的精度验收 + 黄金图真值复核（n=5 → n=1000）**。数据：`sirius-ai/LPRNet_Pytorch` 的 `data/test/` 1000 张真实裁剪图（文件名即真值，非合成）。结果：**rpv3 906/1000 = 90.6% vs LPRNet 888/1000 = 88.8%，McNemar p=0.179 不显著** —— 这是 LPRNet 自己的测试集，rpv3 从未见过该分布，仍不落下风 → **LPRNet 无替换理由**。rpv3 错误画像（逐行重算，非目测）：**94 个错误中 84 个是同长度替换（省份混淆），长度错误 10/1000 = 1.0%**；对照 LPRNet 长度错误 **70/1000 = 7.0%**。**官方字典源码到手（`_load_data.py`）**：与我此前经验恢复的映射逐项一致 → **ADR-007 §5「67 字符需与 77 项对齐」是一条不存在的欠账，撤销**。**黄金图真值被推翻**：`苏ED5172` 不是人工标注，它只是 rpv3 自己的输出（ADR-002 移植保真对照表）；**rpv3 在标准车牌比例下自己读出 `苏ED51712`（8 字符，conf 0.987）**；对照实验（n=200 已知 7 字符牌）证明**水平拉伸从不新增字符（`orig=7→str7=8` 转移 0/200），水平压缩 17% 吃字符**；车牌在画面里被偏航压缩约 2×（真牌四边形 123.5×77.3，比例 1.60 vs 标准 3.14）→ **正确真值是 `苏ED51712`**。**牌色判据（含自纠）**：⚠️ `plate_colour_analysis.py` 的牌色分类器有 bug（用「亮于中位数像素」的色相判色 → 蓝牌上最亮的是白字 → 把 86 张蓝牌判成绿牌），「绿 hue 37.2%」**作废**；修正量法（`plate_face_colour.py`：牌面漆 `S≥90 & 45≤V≤250`，并用**车牌自己的白字**做白平衡对照）→ **黄金牌 G−B=+39 且白字中性（+6）→ 真绿牌**（蓝牌参照组 G−B 中位数 −89.5）。⚠️ **数据集局限：1000 张全为 7 字符，新能源 8 位牌 0 张**（牌色独立确认无真绿牌，互洽） |
| `A16-verify-rerun.log` | **A16 的可复现性复跑（在落盘数据集上重跑）**：888/1000、906/1000、p=0.1788 与 A16 **逐位一致** → 数据集落盘无损、结论可复现。数据集的 60 张曾被同名重下覆盖，此复跑是"覆盖未改结果"的证明 |
| `lprnet_accuracy_20260918.md` | **A16 的原始错误表（逐行数据，非摘要）**：LPRNet 112 个错误 + rpv3 94 个错误的完整清单（`file / gt / LPRNet / rpv3` 四列）。**A16 的错误画像数字（94 / 84 / 10 / 70）就是由本表逐行重算得出的**，可复核、可反驳。⚠️ 引用错误画像时以本表为准，不要引任何"目测/摘要"版本 |
| `A15-expected-string-matrix-release-20260918.log` | **期望串逐字符矩阵（release，决定性）**：黄金图 `hlpr-test.jpg`（期望串 `苏ED5172`）× 各角色 × 各后端。**唯一翻字符的落点是 `det` 落 NPU（match=000，`苏ED5172`→`苏E05172`）**；det cpu/gpu/kirin、rec nnrt(NPU)、cls 全部 111 全对。→ **最优且安全的分配就是生产档 `det=CPU / rec=NPU / cls=CPU`**；「全 NPU」那 1.5× infer 提速是拿正确性换的，不可用。方法论：换档/换模型验收必须比对**期望串**，不能拿「相机碰巧读对一个」当精度证据 |
| `A14-camera-realtime-e2e-20260918.md` | **相机实时端到端三档后端**：新增「相机实时识别」页（双路预览：XComponent 显示 + ImageReceiver 取帧），实时读到真实车牌（`苏A1085K`/`浙J3U1P3`）。三档 p50：**全 NPU 34.5 fps**（infer 12.7 ms）> 生产 det=CPU 23.8 fps > 全 GPU(ncnn-Vulkan) 12.8 fps。**两个真实缺陷被定位并修复**：① NV21→RGBA+旋转原在 ArkTS 层占 21–38 ms（比推理还贵），搬进 C++ 后降到 2–8 ms，fps 15→24+；② `buildMode=debug` 把 native 编成 `-O0`（`cppFlags:"-O3"` 被 debug 的 `-O0` 覆盖），改 release 后 infer ~30→~10 ms。⚠️ **历史基准（bench30/E2EMATRIX）都是 debug 构建测的**，内部可比但不能与 release 数字并列。**§4.2 追加（release）**：修掉「完成率被干等下一帧砍半」缺陷（done 12.5→24.0，丢帧 0，持续 1.77×）；基准档 ≡ 生产档出帧率 → **证明推理侧已非瓶颈**；摘显示路无差别（排除双流互抢）；加速器矩阵证明 **MS Lite 的 gpu/kirin 全静默回落 CPU**、NPU 只在 rec 上有 2.31× 收益、cls 留 CPU；默认档改「全 NPU」→ infer 34.6→23.0 ms（1.50×），识别零翻字符 |
| `A13-cann-friendly-models-20260918.md` | **「按 CANN 偏好挑模型」成立且判据可算**：官方给出 `Cin`/`Cout` 双 16 倍数判据 → 量化得利用率上限（cls **19.2%** ⚠️ / rpv3 55.6% / det 70.6% / **LPRNet 81.2%** ✅），与 §V-C 实测「cls NPU 0.74× 更慢」方向一致。挑 LPRNet 转 `.om` 真机 `build_rc=0/run_rc=0`，**但分区是 `NPU:2, CPU:1`（`ReduceMean` 被拒），不是整图 NPU**；且 `lprnet_npufix`（Transpose 已消）分区**完全相同** → **推翻 ADR-007 §4 的 Transpose 归因**，并得出「NPU 支持性 = 模型 × 工具链 的联合属性」 |
| `A12-paper-grade-bench-and-cann-full-20260918.md` | **论文级 30× 基准 + CANN 全模型收口**（det 多输出探针 bug 修复后 `run_rc=0`） |
| `A11-cann-om-end-to-end-20260918.md` | **CANN 端到端跑通**：用华为官方公开 codelab 的真 `.om`（2.5 MB，`IMOD` 魔数）在同一条 NNRt 入口上全绿 —— `compat=0` / `build_rc=0` / `executor=ok` / `run_rc=0`，IO `[1x3x227x227]`→`[1x1000x1x1]`，`out_sum=1.0005` 证明是合法 softmax（真跑完前向），稳态 **0.92 ms**；同批 `.ms` 3/3 仍被拒 → **门槛是格式而非权限/能力**。缺口收窄为「缺 OMG」 |
| `A10-cann-nnrt-probe-20260918.md` | **CANN/HiAI/NNRt 真机探针**：`/system/lib64/ndk/` 下 NNRt 与 `libhiai_foundation.so` 均可用、syscap 双 true、CANN 版本 `108.631.120.010`；NNRt 枚举出 **2 张设备全是 ACCELERATOR**（`HIAI_F` + `NPU_ohos.boot.hardware.kirin8020_v2_0`）、**无 GPU 设备**；`.ms` ×3 全被拒（`hiai_compat_code=1`，`build_rc=1`）→ **门槛是 `.om` 转换器而非权限**。**关键交叉验证：MS Lite 的 `built on NNRT:NPU_ohos…` 与本枚举设备名逐字相同 → 现有 NPU 通道就是官方 NNRt 底座** |
| `A9-gpu-rec-cls-20260918.md` | **识别与分类也吃上 GPU（三模型 × 三后端闭合）**：pnnx ONNX→ncnn（rec 252 层 / cls 78 层，`Shape`/`Slice` 被常量折叠）；ncnn 槽位化 + 流水线分支；rec/cls 在 ncnn-CPU 与 **ncnn-Vulkan(Maleoon 920C)** 上各 3 次全部 `match=1`，分类分数逐位一致。**结论：GPU 通路成立且保真，但小图无加速（rec 52–72ms vs NPU 7.5ms）** |
| `A8-native-only-teardown-20260918.md` | **拆掉 ArkWeb**：手机端改为纯原生两页（原生演示 / 探针控制台），rawfile 53→26 MB、HAP 87.1→58.3 MB；拆完重跑全链：16/16 矩阵、0 watchdog、`cropSum=2773473` 未变 |
| `A7-native-rewrite-async-20260918.md` | **手机端端到端重写**：推理移入专用线程 + async NAPI + 会话登记表（去重/封顶）+ 结果落盘。验证：**16/16 组合跑完、0 watchdog、进程存活**（改造前同样 16 组合必被 SIGKILL） |
| `A6-three-model-backend-matrix-20260918.md` | **三模型 × 四后端完整矩阵**：rec 在 NPU 上 match=1（唯一真正吃到 NPU 收益的模型）；cls 只有 **fp32** 变体能上 NPU（fp16 构不出 kernel）；MS Lite 的 gpu/kirin 档编译期即不支持；并定位**主线程阻塞 8.887 s → THREAD_BLOCK_6S → SIGKILL** 的架构缺陷 |
| `A5-three-backends-20260918.md` | **2026-09-18 三后端复现验证**：CPU✅ / GPU(ncnn-Vulkan Maleoon 920C)✅ / NPU✅ 四模式各 3 次全跑通；与 A4 逐项一致 |
| `A4-backend-e2e-result.txt` | **A4 端到端四后端对照**：ms-cpu / ncnn-cpu / ncnn-vulkan 全对，ms-nnrt 全翻字符（苏E05172） |
| `A3-verification-result.txt` | A3 构建+装机验证结论 |
| `det_npu_proof_sameprocess_20260917.txt` | **NPU 真在算**：同进程 nnrt vs cpu 的 L2/耗时对照（1043.4696 / 18.0000 / 5.04 ms vs 1042.6634 / 18.0232 / 8.44 ms） |
| `det_npu_vs_cpu_drift_20260917.{json,txt}` | NPU 0.08% 漂移经 anchor 放大约 10 px 的量化记录 |
| `fidelity_report.txt` | 移植保真对照总报告（8/8 一致） |
| `verify_head_decode_match_20260917.txt` | decode 公式与参考的匹配度（MATCH 0.000061） |
| `bare_head_pipeline_check.json` | 裸 head 导出后的流水线自检 |
| `det_head_ref_checksum.json` | 裸 head 参考校验和（回归锚点） |

## 2. 中间张量与数值分析

| 文件 | 说明 |
|---|---|
| `det_dump.json` | 检测器原始输出行 dump |
| `hlpr_reference.json` | Python 参考实现的逐阶段输出（ground truth） |
| `compare_*` 类由 `tools/compare_dumps.py` 生成 | 逐阶段差异 |
| `tensor_meta.json` | 张量 dump 的形状/类型元信息（**数据本体已归档**，见下） |
| `paper_ramp.json` | 135 个缩放比的扫描数据（→ 论文图 4） |
| `paper_residual.json` | 非整数比残差统计 |
| `probe_truncation.json` | 关键点未截断导致的 1 px 偏移探针（MU4158 → H468） |
| `onnx_io.json` | 三个模型的真实输入/输出形状（**NHWC**，勿按 NCHW 处理） |
| `ramp/port_resized.json`、`ramp/diag.txt` | 缩放扫描的过程诊断 |

## 3. 端侧（麒麟 8020）采集

| 文件 | 说明 |
|---|---|
| **A18-stage-budget-thread-sweep-20260920.md** | **2026-09-20 第二轮（重打包版）：① 分段预算结案** —— det 段 encode+infer 18.51 ms 中 pack 仅 0.35 ms，"17.7 ms 之谜"几乎全是推理；流水线内 det 推理 18.15 ms 比孤立 bench 11.37 ms 慢 59%（段间效应，buffer 复用是下一个靶点）；单帧 36.11 ms → 理论上限 27.7 fps；② **CPU 线程扫描**：三模型最优都是 **4 线程**（t6 持平、t8 超订阅更慢），CPU 基线结案为"已调优"；③ **全矩阵 72/72 无截断**（会话上限 20→64 修复），cls-fp32×nnrt_fp32 补跑成功，fp32 结论二次确认；④ **CPU 轮间方差 32–53% vs NPU <8%**（比值必须带轮次/热态标注）；⑤ **源码损坏修复记录**：lpr_pipeline.cpp 三处 yolov8 时代遗留损坏（匿名 namespace 闭合括号被啃 + 两段 LprToken 误粘贴），09-19 起编译不过但设备行为正常。原始：`full-hilog-20260920-r2.txt` + `chain-20260920-r2.txt` |
| **A17-release-30x-fp32-cann-20260920.md** | **2026-09-20 release 构建 30× 全矩阵复测（最新口径）**：① **NNRT-fp32 结案** —— det 上 NPU 在 fp32 档仍然 3/3 翻字符（`苏E05172`），rec 3/3 保持正确 →「det 不能上 NPU」不再需要 fp16 限定；② release 30× p50：det-head NPU 5.3505 vs CPU 7.6638（1.43×）、rec NPU 3.9883 vs CPU 9.1286（**2.29×**）、cls-fp32 NPU 1.0012 vs CPU 0.8395（0.84×）；③ **CANN OMG 缺口闭合** —— 项目自有 dethead/rec/cls 三个 `.om` 全部 `build_rc=0/run_rc=0`（rec 5.19–5.63 ms 仍慢于 MS 的 3.99 ms）；④ GPU= ncnn-Vulkan 三模型全通但全慢于 CPU（rec 251/252 层落 GPU）；⑤ 会话上限截断 4 组合（cap=20，日志可见，非模型问题）。原始：`full-hilog-20260920.txt` + 清洗行 `chain-20260920.txt` |
| `ncnn_device_run_20260917.txt` · `ncnn_ref_check.json` | ncnn CPU 真机跑通 + 与参考的一致性 |
| `vulkan_probe_app_20260917.json` · `vulkan_probe_hilog_20260917.txt` | **GPU 能力探针**：Maleoon 920C / Vulkan 1.3.275 / fp16 |
| `backend_pid11692.log` | A4 四后端端到端对照的完整设备日志（PID 11692） |
| `a3_vk_build.log` · `vk_build2_utf8.txt` | ncnn-Vulkan 交叉编译过程（含 glslang 链接坑） |
| `a3_hap_build.log` · `a3_install.log` · `a3_start.log` · `a3_stop.log` · `a3_app_status.json` · `a3_build_status.json` | 装机与启动链路 |
| `backend_compile_checkpoint.log` | 后端编译检查点 |
| `hilog_help.txt` · `serve_phone.log` | 工具用法与手机侧服务日志 |
| `current_pid.txt` | 当前设备进程号 |
| **已归档** `a3_device_full.log`（7.8 MB 全量日志） | 见 `_archive/2026-09-18/big-logs/` |

## 4. Web 侧自检

| 文件 | 说明 |
|---|---|
| `web_index_html_auto_1.json` · `web_index_html.json` · `web_index_html_390x844.json` | 介绍页自检（桌面 / 手机视口） |
| `web_demo_html_auto_1.json` · `web_demo_html.json` · `web_demo_html_390x844.json` | 演示页自检 |
| `web_demo_html_bench_30.json` · `web_demo_html_bench_30_threads_1.json` | **30 次延迟基准**（6 线程 vs 单线程） |
| `web_mobile_html_auto_1_390x844.json` | 移动版页面自检 |
| `web__verify_html.json` | `tools/_verify.html` 的逐阶段比对结果（最完整的一份，108 KB） |
| `mt_sweep_round1.txt` · `mt_sweep_round2.txt` | WASM 多线程扫描两轮（线程数 → 延迟） |
| `web_diag_phone_20260917.txt` | 手机端诊断输出 |
| `probe_blocked.json` | 扩展名白名单 403 的探针记录（`.onnx` / `.mjs` 被拦） |

## 5. 模型与选型

| 文件 | 说明 |
|---|---|
| `model_probe.json` | **YOLOv9-t 三档输入尺寸的探针结果**（ADR-001 证伪记录的依据） |
| `upstream_eval.json` · `promoted_samples.json` | 上游样本打分与提升记录 |
| `viz_summary.json` | 参考实现可视化摘要 |

## 6. 归档去向（不在本目录，但仍是证据）

| 内容 | 位置 | 体积 |
|---|---|---|
| 中间张量 dump（`.f32` / `.rgba` / `port_resized.bin`） | `_archive/2026-09-18/intermediate-dumps/` | 33 MB |
| A3 设备全量日志 | `_archive/2026-09-18/big-logs/` | 7.8 MB |
| 探针/QA 截图（含 YOLOv9 探针可视化） | `_archive/2026-09-18/screenshots/` | 6.5 MB |

> 中间张量 dump 由 `tools/dump_tensor.py` / `tools/dump_raw.py` / `tools/fit_resize.mjs` 重新生成；
> 截图由 `tools/web_check.mjs` / `tools/shot_el.mjs` 重新生成。归档是为了让仓库清爽，不是丢证据。