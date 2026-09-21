# 项目长期记忆 · 端到端车牌识别（毕业设计，树森）

## 目录与文档（2026-09-18 整理后）
- 入口文档：根 `README.md`（总导航/快速开始/核心数字）· `docs/INDEX.md`（文档地图 + ADR-001~011 索引含状态）· `_evidence/INDEX.md`（证据索引）· `tools/README.md`（62 脚本清单，自动生成）· `docs/research/2026-09-18-idea-feasibility.md`（IDEA 三条的可行性调研）
- **归档**：断链资产（jsep.wasm 26MB / yolo-v9-t×3 + y5fu_640x 26MB）、中间张量 dump（33MB）、`a3_device_full.log`（7.8MB）、探针截图、latex 探针与备份 → 全在 `_archive/2026-09-18/`（105MB，可整目录删）。**不要移回 dead-assets**：那些文件不被任何代码引用
- 已删：`tools/__pycache__`、LaTeX 编译日志、空目录 `harmony/`、0 字节文件、逐字节重复的探针图
- 项目有效内容 144.2MB → 38.8MB（不含 .venv）
- ⚠️ 文档欠账：端侧跑分表**缺「框架」列**（NPU=MindSpore Lite / GPU 只有 ncnn-Vulkan，跨栈比较仅作量级参考）；`index.html` 的 title/h1 与论文口径不一致；采集轮次 <5
- 需求口径确认：**不自训练，只用现成开源权重**

## 命名（已定，改名需四处同步）
- 展示页 h1/demo 标题：基于 YOLOv5 与 CRNN-CTC 的纯前端端到端车牌识别
- 简历/论文：基于 HyperLPR3 的端到端车牌识别与跨语言移植保真验证
- 同步点：index.html(title/meta/.brand/h1/.eyebrow/footer)、demo.html、ieee-lpr-paper.md、resume-bullets.md

## 技术事实
- 模型：det 2.23MB/rec 9.78MB/cls 1.53MB；矫正=单应+bicubic；ORT-web 1.29.0；0外部网络
- 实测：移植8/8一致；Web 真机 6线程 p50 88.3ms；原生 102–116ms；相机 300ms=letterbox（ADR-010）
- ⚠ 旧口径 13.11×/243.8ms 作废勿引；延迟一律 warm-up 后 30 次 p50
  （13.11× 仅「作为本项目整体加速比」这个口径作废；作为 ResNet-50 骨干比值仍有效）

## 2026-09-18 深夜：ADR-015 口径变更 + 本轮新读码事实（**改端侧代码前必读**）

- **`match=` 语义已正式改**：`Index.ets` 的 `EXPECTED_CODE` → **`REFERENCE_CODE_RPV3`**，
  `match=` 只表示「**与同角色 CPU 基线逐字符一致**」，**不是**「读对了」。BEGIN 行 `expected=` → `reference=`。
  精度一律引 `_dataset/real/crops`（n=1000，文件名即人工真值）：rpv3 **90.6%**、LPRNet **88.8%**、
  McNemar **p=0.1788**、长度错误 1.0% vs 7.0%。黄金图真值是 **`苏ED51712`（8 字符）**。
- 🔑 **`addNnrt()` 对每个 NNRT 设备恒传 `fp16=true`**（`ms_engine.cpp:126-130`），`PlanFor` 里
  **从来没有 fp32 分支**（只有 GPU 有 fp16/fp32 两试）→ 论文里「NPU 0.08% 漂移 → det 不能上 NPU」
  是**一次 fp16 推理**的观测。已新增可请求档 **`nnrt_fp32`**（落点标签带 `#fp32` 后缀），**待真机测**。
  → 在测完之前，凡说"检测不能上 NPU"必须加"（fp16）"限定。
- 🔑 **CPU 后端从不设性能模式**：`SetPerformanceMode(HIGH)` 与 `EnableFP16` 一起包在
  `if (type != CPU)` 里（`ms_engine.cpp:179-182`），且 `SetThreadNum(ctx, 4)` 硬编码。
  **原生 CPU 的线程数/性能模式全仓从未扫描**（ADR-004 §5.4 的"6 线程最优"只测过浏览器 WASM）
  → 意味着现在所有"NPU 对 CPU 的比值"，CPU 端是**未调优基线**。
- **回落现在显式可读**：`PlanFor` 末尾无条件追加 CPU，故 `LANDED=CPU` 分不清"选的"还是"回落的"。
  `MsSession` 新增 `requested`/`fallbackFrom`，矩阵行输出 `engine=ms req=gpu fallback=GPU:fp16`。
- **ncnn 落点补强**：`effectiveVulkan` 只是选项回显（我们自己把 opt 设成 true）。
  新增 `VkCoverage()` 输出 `vkLayers=X/Y`（带 Vulkan 实现的层数）+ `vkGpuCount/vkGpuName`；
  **X<Y 时该格是"GPU + 部分层回落 CPU"，不得当纯 GPU 数据点**。
  ncnn **无 `create_pipeline_cache()`**、`use_fp16_packed/storage/arithmetic` 三开关全 `false`。
- **A14 §4.2.2「相机只给 15 fps、推理侧已彻底不是瓶颈」已撤回**（该文件里已就地标注）：
  基准档仍付整帧 NV21→RGBA（`convMs` 非零），且与 §4.1 同场景 `arrive=19.92–20.06` 自相矛盾；
  `frameMs = decodeMs + inferMs` 不含 ArkUI 重渲染（`record()` 每帧写 6–8 个 `@State`）与 napi 往返。
  结案要新增「只数帧不 `readLatestImage`」档 + 固定片元回放台。
- **「全 NPU 快 1.5×」的归因要打折**（但原因不是模型错）：`infer 34.6→23.0` 里 det 确实换成了
  NPU 版裸 head —— `om_dethead` 是**真检测器**，A12:62 实测 `run_rc=0`、5.025|5.224|6.395 ms、
  **3 个 head 全部出数**，A13:91 分区 `NPU:1, CPU:0`（纯 NPU）。
  ⚠️ 曾有分析把它与 codelab 的 `hiai.om`（SqueezeNet，IO `[1x3x227x227]→[1x1000x1x1]`、
  单输出 5 类）**混为同一个模型** —— 那是错的，别再用"det 的 .om 不是检测器"这个论据。
  真正要打折的理由是**命中数混淆**：A14 §4.2.6 两档命中 `×21` vs `×10` 不同，
  少命中就少跑 rec/cls，所以那 1.5× 里有多少属于后端、有多少属于"这帧压根没牌"从未控制。
- 每帧必付的非推理开销（"帧率上不去"的主嫌疑）：`cameraFrameAsync` 整帧 ArrayBuffer 拷贝 +
  native 再 memcpy 一次、`LprNv21ToRgba` **单线程**且 rot=90 按列散射写、`LprDetPrepare`
  **同尺寸也整图重排** + `memcpy(g_blob,...)`、`Mat::clone()` 每帧新建、每帧多跑一遍 RGBA 全缓冲校验和。
- ⚠️ `kMaxSessions=20`（`napi_init.cpp:87`）现在被 **16 组合的 e2eMatrix 用满**；
  `E2E_BACKENDS` 再加一档（如 `nnrt_fp32`）变 20 个新会话 + 生产 3 个 = **23 → 静默 BUILD-FAIL**。
  扩档前必须先做"跑完即卸 + 每候选重启进程"的分批 runner。
- 当前 release signed HAP **76 MB**（A8 记的 58.3 MB 是另一档构建）→ 待查符号是否被打包。
- `npu_scan.py` 只扫了 rank/perm/Reshape，**算子画像不在 tools/**（`_evidence/_opsprof.py` 硬编码绝对路径）。

## 2026-09-20 深夜：gap 系统调查收官（五个假设全部证伪 + 最终版状态）

- **pipeline-vs-bench gap（det 流水线内 14–33ms vs bench 稳定 7.3–8.7ms）做了五个对照实验，全部证伪**：①每帧分配（buffer 复用全上）无效；②线程池唤醒（det@cpu_t1）无效——**与线程数无关**；③marshaling 只值 +0.6ms（新 `p50Io` 忠实 bench 实测）；④张量句柄缓存无效；⑤数据模式（94%零 vs 密集随机）无效。仅"负载节律/调频"获弱支持（bench 插 3ms 空档只 +1.2ms）。thermal service 本机不可读，无法直接观测。**结论：gap 主因是测量环境状态（热态/调度/调频），不是代码路径**；六轮流水线 29–56ms、bench 稳定 → 报最好轮 29.09ms + 轮间带。相机连续负载预期落在快档。
- **最终版（r9）保留改动**：PipeScratch/ResizeScratch buffer 复用、张量句柄缓存、`p50Io` 计时、kMaxSessions=64、cpu_t1..t8 档+threadSweep、pack/infer 分段计时。最好轮单帧 29.09ms（基线 36.11）。正确性全程 cropSum=2773473 逐位不变、72/72 矩阵完整。
- **dopt/INT8 线暂停**（用户叫停）：dopt.so 需要 Python 3.7（3.8 缺 `Py_EnterRecursiveCall`、3.11 段错误、3.14 ImportError）；已编好 3.8/3.11 无用，3.7 编译被 kill。`omg_conv/calib/` 有 70 个校准 bin（magic 510 NCHW，与 App 预处理一致）+ `calib.prototxt`，随时可续。WSL 抽风时用 `wsl --update --web-download` 救活（实测有效）。
- 论文 §V-H 已含：分段预算表（最好轮 29.09ms + 六轮带）、五次证伪表、线程扫描表、CPU 基线"已调优"结案。

## 2026-09-20 下午：A18 分段预算 + 线程扫描 + 源码损坏修复（**帧率归因定论**）

- **17.7ms 之谜结案**：stages 串扩到 9 段（末尾追加 pack|infer）。det 段 encode+infer 18.51 ms = **pack 0.35 + infer 18.15** —— NHWC 打包只占 1.9%，被证伪；**流水线内 det 推理 18.15 ms 比孤立 bench 11.37 ms 慢 59%**（段间效应：每帧新建 vector/页错误/缓存逐出）→ buffer 复用是下一个靶点（约 6.8 ms，不换后端）。单帧 36.11 ms → **理论上限 27.7 fps**，与明亮场景到达率 26.9–27.1 自洽。
- **CPU 线程扫描（cpu_t1..t8 新档）**：三模型最优都是 **4 线程**（t6 持平、t8 超订阅慢 6–165%）→ CPU 基线"已调优"结案；cpu_t4 与普通 cpu 档同轮差 <0.2%（档位实现正确）。
- **CPU 轮间方差 32–53% vs NPU <8%**（两轮间隔 20 分钟同代码同协议）→ 论文比值必须带轮次/热态；CPU 单点绝对值不足为凭。
- 🔴 **lpr_pipeline.cpp 源码树从 2026-09-19 起编译不过**（ yolov8 整合遗留三处损坏：匿名 namespace 闭合括号被啃成注释残迹 → 5 个 ambiguous；两段误粘贴 LprToken 碎片 + 文件截断）。已修复（删碎片 + 恢复 `}  // namespace`），BUILD SUCCESSFUL，72/72 矩阵与 A17 逐项一致。**教训：改端侧前先 assembleHap 干跑编译；`lpr_pipeline_new_v2.cpp` / `.yolov8_backup` 是未完成合并的遗留物。**
- `kMaxSessions` 20→64：A17 被 cap 截断的 cls-fp32×{nnrt_fp32,nnrt_fp16,gpu,kirin} 4 组全跑通；threadSweep 已接入自检链尾（也有"线程扫描"按钮）。
- 帧率问题的完整答案（用户问过）：相机给帧由曝光决定（暗 20/亮 27 fps，dropped=0，强 30/60 无可靠收益）= 无牌时已榨干；有牌时的瓶颈在推理 36 ms/帧；优化靶点排序：det 段间 6.8 ms > letterbox 5.43 ms > rec 已在 NPU 无空间。

## 2026-09-20：A17 release 30× 复测 + NNRT-fp32 结案 + CANN 三模型 .om 全通（**改论文/端侧口径前必读**）

- **NNRT-fp32 已测（结案 ADR-015 未完成动作 2）**：`nnrt_fp32` 档 e2e 实测 det **fp32 下仍 3/3 翻字符**（`苏ED5172`→`苏E05172`），rec fp32 3/3 正确 → **「det 不能上 NPU」不再需要「（fp16）」限定**，成为无条件结论；生产链 `det=CPU/rec=NPU/cls=CPU` 不变且理由更硬。
- **CPU 性能模式已在代码里**（`ms_engine.cpp:179` `SetPerformanceMode(HIGH)` 对 CPU 也生效，2026-09-18 23:19 后加入；机上版本 23:46 装机含此修复）→ 旧记忆「CPU 端从不设性能模式」**作废**；但**线程数仍硬编码 4**（`ms_engine.cpp:176`），CPU 仍非完全调优基线。
- **release 30× p50 新口径**（A17，替掉 debug 旧数字）：det-head NPU 5.3505 / CPU 7.6638（1.43×）；rec NPU 3.9883 / CPU 9.1286（**2.29×**）；cls-fp32 NPU 1.0012 / CPU 0.8395（0.84× 更慢）。旧 bench30_matrix（debug）数字不可与本次并列。
- **CANN：OMG 缺口已闭合**（项目自有 `probe_om_dethead.om`/`probe_om_rec.om`/`probe_om_cls.om` 全部 compat=0/build_rc=0/run_rc=0）：dethead 3.90–5.10 ms（3 输出头全出）、rec 5.19–5.63 ms（**慢于 MS 3.99**）、cls 0.95–0.97 ms；完整 `probe_om_det.om` 仍被拒；三个 `.ms` 同一次运行仍 compat=1/build_rc=1 被拒（格式门槛复验）。**CANN 不替换 MS Lite 作为生产后端**（rec 更慢 + 仅 3 次合成输入采样，未过同输入 L2/30×/thermal 三道门槛）。
- **GPU 纠正口径**：不是"调用不了"——MS Lite 的 GPU 档编译期不支持（该 .so 无 GPU delegate）+ NNRt 无 GPU 设备，但 **ncnn-Vulkan（Maleoon 920C）三模型全通**（vkLayers 218/218、251/252、78/78，clsScores 逐位一致）且**全都慢于 CPU**（小图提交开销）。想快只能等大模型或承认"通路闭合"结论。
- **会话上限会截断矩阵**：`E2E_BACKENDS` 6 档 × 4 角色 = 24 组合 > `kMaxSessions=20`（不卸载）→ cls-fp32 的 {nnrt_fp32, nnrt_fp16, gpu, kirin} 4 组静默 BUILD-FAIL（日志有 `err=session cap 20 reached` 可查）。跑全量矩阵前先做"跑完即卸/分批判"或抬上限。
- **帧率口径（用户问过）**：无牌时到达率=完成率（dropped=0，来多少吃多少，暗 20/亮 27 fps 由相机曝光决定，强 30/60 fps 无可靠收益）；**有牌时 arrive=27→done=13 才是推理瓶颈**（生产链整帧 ~26–35 ms，release 实测）。未拆的分段预算：det 段流水线内 24.86 ms vs 纯推理 7.13 ms（ADR-009 §5）。

## 端侧终审（nova 14 Pro·麒麟8020·HarmonyOS 6.1，ADR-004~011）
- NPU 判据：rank≤4；perm 只许 [0,1,3,2]；Reshape 静态可推。官方算子清单不作数，先静态扫描
- det 裸 head（export_bare_head.py）→ **全图纯 NPU** 4.95–5.04ms（CPU 8.4–9.1）；rank-5 切点回落 CPU
- rec rpv3 NPU **混合执行** 2.97–4.17ms（svtr mixer 回落 CPU）；cls 上 NPU 反而慢→CPU
- **NPU 真在算（ADR-009）**：nnrt L2 1043.4696/18.0000/5.04ms vs cpu 1042.6634/18.0232/8.44ms；日志 HCL+hiaiserver 实锤
- **但 0.08% 漂移经 anchor(≤433) 放大≈10px → 翻字符**（苏ED5172→苏E05172）→ det 落 CPU、rec 留 NPU；**A4 已端到端实测：NPU 检测 3/3 翻字符，ncnn-cpu/vulkan 3/3 正确（backend_pid11692.log）**
- LPRNet 零 Transpose 仍 LANDED=CPU → NPU 只收「窄算子集+无布局变换」模型
- GPU **终审通过（ADR-008）**：dlopen libvulkan.so → **Maleoon 920C**（0x19e5，Vulkan 1.3.275，2 条 G+C 队列，fp16 在位）；MS Lite/ORT-Web 死
- ncnn CPU已跑通：218层/3输出，10.03ms仅1预热5测量；L2范数差非逐元素误差，未证保真。
- **A4 完整链路实测（PID11692）**：det 四后端对照——ms-cpu/ncnn-cpu/ncnn-vulkan 全对（match=1），ms-nnrt 全翻（苏E05172）；ncnn 系框/置信度与 ms-cpu 几乎逐位一致；生产链维持 det=CPU/rec=NPU/cls=CPU
- decode 公式（ADR-006 §5，MATCH 0.000061）：cx=(sigmoid(t0)*2−0.5+gx)*stride；kpt=原始 logits×anchor+grid 像素

## 未完成（优先级序）
1. ~~letterbox 是端到端大头~~ **2026-09-18 已定位并部分解决（A14）**：真正的头号开销是**取帧转换在 JS 层**（21–38 ms）+ **debug 构建 `-O0`**，两者修掉后相机实时生产档 15→24 fps、全 NPU 档 34.5 fps。**同日后续（A14 §4.1）相机出帧率结论（已二次更正）**：新增第 4 档「基准：只取帧不推理」分离「相机给多少」与「我们算多快」。**暗场景**连续 5 窗口 `arrive=done=19.92–20.06 fps、dropped=0`；**明亮场景**同档复测 `26.92–27.12 fps、dropped=0` → **相机出帧率是场景相关的（自动曝光拉长曝光时间），实测 20–27 fps，不是固定 20**。故 **`1000/p50(frameMs)` 是服务时间理论值不是吞吐**（34.5/23.8/12.8 只可用于比较相对开销）；真实吞吐看到达率/完成率。**有车牌时 `arrive=27 / done=13`（每窗口丢 27–30 帧）= 推理真跟不上**，瓶颈回到 `det=CPU 12.7 ms`。剩余：≥5 轮 thermal 采样、相机帧的精度验收
2. Vulkan 30 次无插桩正式计时（rep0→2 降速疑 shader 编译开销）+逐层执行证据；GPU 正式性能矩阵暂不入论文
3. ~~多摄轮询：Camera Kit 顺序切换+ImageReceiver→NAPI~~ **已做（A14）**：`CameraPage.ets` 双路预览 + ImageReceiver→native 取帧链路成立；多摄同开仍未查并发能力集
4. 新数据回灌论文+展示页（**改动等树森确认**，IEEE 格式）
5. 替换模型按 ADR-007 §2 判据筛（看 perm/rank，不看算子名）；LPRNet 补精度验收后方可谈替换

## 工程红线
- hdc=D:\Tools\Huawei\HarmonyOSSDK\hmscore\3.1.0\toolchains\hdc.exe，设备 4CY9K25614046328；shell 无 getprop，用 param get；/vendor 不可列
- **hdc 必须用 DevEco 3.2.0d**（`D:\IDE\DevEco_Studio\sdk\default\openharmony\toolchains\hdc.exe`）：3.1.0 对 install/bm dump 报 `[E000001] version is too low`；换用前先 `hdc kill` 清掉残留 server
- **hdc install 传相对路径**：绝对路径 `C:/Users/...` 会被拼到 cwd 前导致 `no such file`；先 cd 到 hap 目录再传文件名
- **2026-09-18 三后端复现**：装机后 AUTO_SELFTEST 链自动跑完 det 四模式×3（`_evidence/A5-three-backends-20260918.md`）——CPU/GPU/NPU 全部能调用，NPU 仍翻字符，与 A4 逐项一致
- **三模型×四后端矩阵（A6）**：rec 在 NPU 上 **match=1**（唯一真正吃到 NPU 收益的模型）；cls **只有 fp32 变体能上 NPU**（fp16 变体 `Create full model kernel failed`）；MS Lite 的 `gpu`/`kirin` 档编译期就不支持（`GPU is not supported` / `NPU is not supported`）；生产链正确
- ⚠️ **主线程死穴**：`lpr.loadModel`/`lpr.pipeline` 是同步 NAPI，整轮 16 组合连续占 ArkTS uv loop **8.887 s** → `THREAD_BLOCK_6S` → **SIGKILL**（r1 12 组合活、r2/r3 被杀）。**矩阵/长时间推理必须搬到 worker 线程**，否则必被杀
- 剩余缺口：rec/cls **没有 ncnn 模型** → GPU(Vulkan) 侧只有检测器能跑；需 onnx→ncnn 转换
- **重写（A7，已验收）**：`napi_init.cpp` 改为**专用推理线程 + async NAPI**（`loadModelAsync`/`pipelineAsync`/`benchAsync`/`ncnnLoadAsync`），会话登记表按 **(模型文件,后端)** 去重、上限 20，新增 `appendLine` 落盘。**16/16 组合跑完、0 watchdog、进程存活**（改造前同规模必被 SIGKILL）
  - 铁律：ArkTS 侧**一律用 `*Async`**；会话名**必须传模型文件路径**（传角色标签会去重失效撞上限）；新增原生入口涉及 MS/ncnn 时必须走 `RunJob`/`RunJobSync`
  - 未做：拆 ArkWeb（Web 标签页 + rawfile 里 13MB ORT wasm 等，HAP 仍 ~87MB → 目标 ~30MB）
- **纯原生化（A8，已验收）**：`Index.ets` 重写为两页（原生演示 / 探针控制台），**零 WebView、零同步 NAPI**；旧实现存 `lpr-harmony/reference/Index.arkweb-era.ets`（工程外不编译）
  - `sync_rawfile.py` 的 MANIFEST 缩到 3 张样本（**不改它下次同步会把网页资产拷回来**），`EXTRA=[]`，`KEEP=["models"]`
  - 体积：rawfile 53→26 MB、HAP 87.1→58.3 MB；拆完重跑全链 16/16、0 watchdog、`cropSum=2773473` 未变
- **GPU 三模型闭合（A9，已验收）**：pnnx（pip，`pnnx model.onnx ncnnparam=.. ncnnbin=..`）把 **rec（252 层）/ cls（78 层）** 转成 ncnn；`Shape`/`Slice` 被**常量折叠**（固定输入形状下 ncnn 无动态形状层，这是最大风险点，自动消解）
  - ncnn 引擎**槽位化**：0=检测旁路（保留不动）、**1=识别、2=分类**；`pipelineAsync` 第 8/9 参传 `recSlot`/`clsSlot`；`ncnnLoadSlotAsync` 加载
  - 结果：rec/cls × ncnn-CPU / **ncnn-Vulkan(Maleoon 920C)** 各 3 次全 `match=1`
  - **保真口径要分两层**（逐位核验过）：**分类** `0.0001|0.9998|0.0001` 与 MS 路径**逐位一致**；**识别**只到**符号级**——车牌串与 chars 一致，但 charProbs 与 MS-NPU 差 **~0.5–1%**（ncnn `0.7271|…` vs MS `0.7227|…`）。**ncnn 的 CPU 与 Vulkan 输出彼此完全相同**（8 行取值唯一），故这点差是 ncnn↔MindSpore 的框架差，不是 CPU↔GPU 差
  - ⚠️ **GPU 不是加速器**：rec 该段 GPU 52–72ms vs NPU 7.5ms；cls GPU 11–15ms vs CPU 6.6ms —— 小图提交开销主导（同 ADR-003 缩放律）。口径必须是「通路成立 + 数值保真」，**不许写「GPU 加速」**
  - 识别模型的 T/C：ncnn 侧不自省形状，`T=20`（ONNX 声明固定）+ `C = n/20`，**绝不能用字符表的 77**
- **CANN / HiAI / NNRt 路线评估（ADR-012 + A10，已实测）**：设备 `/system/lib64/ndk/` 下 `libneural_network_runtime.so` / `libneural_network_core.so` / **`libhiai_foundation.so`** 都能按 soname `dlopen`；syscap `SystemCapability.AI.HiAIFoundation` 与 `…NeuralNetworkRuntime` 双 true；**CANN 版本 `108.631.120.010`**
  - **门槛不是权限，是 `.om`**：`.ms` ×3 喂 NNRt 离线入口全被拒（`hiai_compat_code=1` + `build_rc=1`）；OMG 转换器要华为账号 + 64 位 Linux + 版本对齐
  - **NNRt 枚举 2 张设备全是 `ACCELERATOR`（`HIAI_F` + `NPU_ohos.boot.hardware.kirin8020_v2_0`），无 GPU** → GPU 仍只有 ncnn-Vulkan
  - 🔑 **关键互证**：NNRt 枚举的设备名与 MS Lite 日志 `model built on NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0` **逐字相同** → **现有 NPU 通道就是官方 NNRt 底座，不是绕路**；所以「直连 NNRt」性能上限极小
  - 探针代码：`cpp/nnrt_probe.{h,cpp}`（dlopen 逐级回退 + `dlerror`），NAPI `nnrtProbeAsync`/`nnrtTryModelAsync`；App 内「CANN/NNRt 探针」按钮
  - 坑：`deviceID` 是 64 位 `size_t`，**落日志必须 `%llu`**（按 `%d` 打印会截断成 `1214632364`，实测踩过）
- **CANN 端到端跑通（ADR-013 + A11，重大）**：**"我们缺的是转换器，不是通路"**
  - 从华为官方公开 codelab 拿到真 `.om`（`harmonyos_codelabs/hiaifoundationkit-codelab-clientdemo-cpp` 的 `rawfile/hiai.om`，2,496,459 B，**`IMOD` 魔数**，Caffe/SqueezeNet ImageNet 分类）→ 放 `rawfile/models/probe_hiai_imagenet.om`
  - **全绿**：`hiai_compat_code=0` → `build_rc=0` → `executor=ok` → IO `in0=data FLOAT32 [1x3x227x227]` / `out0=output_0_prob_0 FLOAT32 [1x1000x1x1]` → `run_rc=0` → **`run_ms_each=2.186|0.944|0.918`（稳态 ~0.92 ms）**
  - **完整性校验**：`out_sum=1.0005`（1000 维和为 1）→ 真是合法 softmax，NPU 跑完了完整前向，不是返回垃圾
  - **同一次运行对照**：三份 `.ms` 仍 `compat=1` + `build_rc=1` → **门槛是格式，不是权限/能力**
  - 顺带确认：`OH_NNCompilation_SetDevice` 可**显式绑定设备**（比解析日志硬一档的落点证据手段）；IO 规格运行时可读
  - ⚠️ 口径：那份 `.om` 是**客人模型**（非车牌识别），**不得**写成"用 CANN 加速了车牌识别"；输入是合成填充，`argmax` 无意义
  - 缺口=**OMG**（华为账号 + 64 位 Linux + 设备 CANN `108.631.120.010` 版本对齐）；替代=NNRt 在线构图（`OH_NNModel_*`，需自研算子映射）
  - 探针实现要点：`OH_NNTensor_Create(deviceID, desc)` → `OH_NNTensor_GetDataBuffer` 填确定性输入 → `OH_NNExecutor_RunSync` → 读输出统计；**逐级销毁**（tensor→desc→executor→compilation）
- **30× 正式基准 + CANN 全模型收口（A12）**：`probeAll()` 走 **`LprMatrix`** 标签（不是 `append()`/`appendNative()`；`MATRIX END` 会与 `E2EMATRIX END` 混淆，勿凭子串判定），日志落 `_evidence/bench30_matrix_20260918.log`（14 条 `p50=`）。论文级数字：**det-head NPU 5.227 ms（CPU 7.829，1.50×）/ rec rpv3 NPU 4.017 ms（CPU 9.136，2.27×）/ cls fp32 NPU 1.054 ms（CPU 0.780，0.74× 更慢）/ cls fp16 静默回落 CPU 0.797 ms**。判 NPU 落点可用 **`L2asFp16` 列**（NPU 行非零、CPU 行 `0.0000`）。探针多输出 bug 已修（`run_tensors=1in/3out`），`om_dethead` 现 `run_rc=0`
- **「按 NPU 友好度选模型」（ADR-014 + A13，本轮）**：用户问「CANN 支持最好的模型，再去搜对应的模型来用」→ **方向成立且判据可算**
  - **官方判据**（`ncnn_conv/模型准备.md` L59，`Convolution` 条）：`Cin`/`Cout` **都是 16 的倍数** → NPU 最大算力；**都 <16** → 算力按 **`(Cin×Cout)/256 × max`** 折算。旁证（L199-201）：官方推荐表是**基于 NPU 硬件利用率**的自我对比，**不是横向对比** → 正好解释「cls 在 NPU 上比 CPU 慢」不是 NPU 不行，是结构不友好
  - **利用率上限**（数 `Cin%16==0 && Cout%16==0` 的 Conv 占比）：cls **19.2%**（10/52，30 个 <16）⚠️ / rpv3 55.6% / det-head 70.6% / **LPRNet 81.2%**（13/16）✅。画像落 `_evidence/_opsprof.json`
  - **LPRNet 真机验证**：`omg_conv/convert_one.sh` 转出 `om_lprnet.om`（916,283 B）+ `om_lprnet_npufix.om`（915,413 B），均 `IMOD`；上机 `compat=0/build_rc=0/run_rc=0`，IO `[1x3x24x94]`→`[1x1x68x18]`，稳态 1.37–1.81 ms，`o0_sum=-62362.3169` 两版逐位相同
  - 🔑 **但「能跑」≠「全落 NPU」**：分区实测 `partitioner_strategy.cpp ToPartitionerList` → LPRNet 是 **`NPU:2, CPU:1`**，拦路的是 **`ReduceMean`**（`fe_sub_stores_manager.cc CheckSupported(314)::"the op name [ReduceMean_58] type [ReduceMean] is not supported in npucl store [elementary_lib]"`），不是 Transpose
  - 🔑 **推翻 ADR-007 §4 假设**：`lprnet_npufix`（Transpose 已消）分区与原始版**完全相同**（`NPU:2, CPU:1`）→ Transpose 改造对 CANN 无用。**「NPU 支持性」= 模型 × 工具链 的联合属性**（同一 LPRNet：MS Lite 判 CPU、CANN 能编译能跑）
  - 分区对照：`imagenet` `NPU:1,CPU:0` / `lprnet(_npufix)` `NPU:2,CPU:1` / `dethead` `NPU:1,CPU:0` / `cls` `NPU:1,CPU:0` / `rec` `NPU:3,CPU:2` / `det`(编译期拒) `NPU:4,CPU:3`
  - ⚠️ 口径：LPRNet 是 CTC 68 项字典（含 blank=67），**项目 77 项字典与之同序同起点 → ADR-007 §5「需对齐」是误判，已撤销**；输入尺寸不同（24×94 vs 48×160），**1.4 vs 3.2 ms 不可横向归因**；仅 3 次采样，非 30× 口径
- **A16 真实第三方数据集精度验收 + 黄金图真值推翻（2026-09-18，决定性）**
  - 数据：`sirius-ai/LPRNet_Pytorch` → `data/test/` **1000 张真实裁剪图**（文件名即真值，非合成），落 `_dataset/real/crops/`；许可=公开仓库随附测试集，仅学术评测；**体积 2.52 MB 实际内容**（`du` 报 4.2 MB 是 4K 块对齐，勿引用）
  - **rpv3 906/1000 = 90.6% vs LPRNet 888/1000 = 88.8%，McNemar p=0.179 不显著** —— 这是 LPRNet **自己的测试集**，rpv3 从未见过该分布，仍不落下风 → **LPRNet 无替换理由**（比原结论更强）
  - rpv3 错误画像（由 `_evidence/lprnet_accuracy_20260918.md` 错误表逐行重算）：**94 个错误中 84 个是同长度替换（省份混淆），长度错误 10/1000 = 1.0%**；对照 LPRNet 长度错误 **70/1000 = 7.0%（rpv3 的 7 倍）**
  - 🔑 **黄金图真值 `苏ED5172` 是错的**：它从不是人工标注，只是 `hlpr_reference.py`（即 rpv3）自己的输出（ADR-002 移植保真对照表）→ 用它当红线是**循环论证**。**rpv3 在标准车牌比例下自己读出 `苏ED51712`（8 字符，conf 0.987）**
  - 对照实验（`tools/stretch_control.py`，n=200 已知 7 字符牌）：**水平拉伸从不新增字符（`orig=7→str7=8` 转移数 0/200）**，**水平压缩 17% 吃字符**（`皖ASS990→皖AS90`）；黄金图车牌被偏航压缩约 2×（真牌四边形 123.5×77.3 = 比例 1.60 vs 标准 3.14）→ **正确真值 = `苏ED51712`**
  - 🔑 **牌色判据（自纠）**：⚠️ `tools/plate_colour_analysis.py` 的牌色分类器**有 bug**（用「亮于中位数像素」的色相判色 → 蓝牌上最亮的是白字 → 量到字不是牌面，把 **86 张蓝牌判成绿牌**）；由此得出的「绿 hue 37.2%」**作废**。修正量法 `tools/plate_face_colour.py`：牌面漆 = `S≥90 & 45≤V≤250`，并用**车牌自己的白字**做白平衡对照 → **黄金牌牌面 G−B=+39 且白字中性（+6）→ 真绿牌**（蓝牌参照组 G−B 中位数 −89.5；数据集里 G−B 为正的仅少数且是**青色偏色 B≈G，非绿**）。`colour_analysis.json` 的**按色细分数字作废**，但总准确率 906/1000 不受影响
  - ⚠️ **数据集局限：1000 张全为 7 字符，新能源 8 位牌 0 张**（牌色独立确认无真绿牌，两条证据互洽）→ 评估 8 位场景需另找数据集
  - **未被推翻**：`det=NPU` 改字符（release 复验 nnrt 000）→ 生产档 `det=CPU/rec=NPU/cls=CPU` **不变**；该结论不依赖真值串位数
  - 待办（待用户确认）：重建黄金验收集（正对拍摄或直接用 1000 张统计）；修正所有文档里的期望串；端侧复验 1000 张；相机帧精度验收
  - 产出：`_evidence/A16-real-dataset-accuracy-20260918.md`、`tools/lprnet_real_accuracy.py`、`tools/stretch_control.py`、`tools/golden_decisive_checks.py`、`tools/plate_face_colour.py`、`tools/colour_classifier_audit.py`（牌色修正量法）、`tools/plate_colour_analysis.py`（⚠️ 分类器有 bug，勿引用）、`tools/glyph_count_geometry.py`
  - **CANN 维持「候选第四后端」**：升后端须过 ADR-002 同输入 L2 对照 + 30× p50 + thermal 标注，本轮未做
- **相机实时端到端（A14，本轮）**：新增 `pages/CameraPage.ets`「相机实时识别」页 —— **双路预览**（XComponent 显示 + 第二路 `ImageReceiver` 取帧分析），三档后端可切（生产 / 全 NPU / 全 GPU）。**功能已验证：相机实时读到真实车牌 `苏A1085K`、`浙J3U1P3`**
  - 取帧：本机 24 个预览档**全是 `YUV_420_SP(1003)`，一个 JPEG 都没有**（`getSupportedFrameRates()` 也返回空）→ 按 profile 选尺寸，实际交付格式由 ImageReceiver 决定，实测拿到 **NV21**
  - **两个真实缺陷（都已修）**：① NV21→RGBA+旋转原先在 ArkTS 层做（`createPixelMap(srcPixelFormat=NV21)`→`rotate()`→`readPixelsToBuffer`），实测 **21–38 ms**，比整个推理还贵 → 新增 `LprNv21ToRgba()`（`lpr_pipeline.cpp`，整数 BT.601 定点 + 写入时按旋转角定位）与合并入口 **`lpr.cameraFrameAsync`**（转换+旋转+推理一次 native 调用），降到 **2–8 ms**；② **`buildMode=debug` 把 native 编成 `-O0`**（`entry/build-profile.json5` 的 `cppFlags:"-O3"` 在命令行更靠后、被 debug 的 `-O0` 覆盖）→ 改 `buildMode=release`（实际 `-O2 -DNDEBUG`）后 infer ~30→~10 ms
  - 帧率 p50（release）：**全 NPU 34.5 fps**（infer 12.7 ms）> 生产 det=CPU 23.8 fps > 全 GPU(ncnn-Vulkan) 12.8 fps。GPU 最慢，与 ADR-008 一致
  - ⭐ **A14 §4.2（2026-09-18 追加，最重要）**：
    - **真缺陷已修：完成率被"干等下一帧"砍半**。丢帧分支原先 `readLatestImage()+release()` 把缓冲抽空 → 算完只能干等下一次 `imageArrival` → `done` 恒为 `arrive` 的一半（24.0 → 12.5）。改为「busy 期间不读不释放、留一帧；新增 `finishFrame()` 算完立刻接手」→ **`done` 24.0，丢帧 0**，持续完成率 12.5 → **22.1 fps（1.77×）**
    - **丢帧计数器语义修正**：只有「已停帧又被新帧顶掉」才算丢 → 恒有 `到达 = 完成 + 丢帧`（旧写法多算 ~1368）
    - **相机上限的干净证明**：基准档（零推理）arrive 14.94–15.09 ≡ 生产档 arrive/done 14.68–15.32 → **推理侧已彻底不是瓶颈**，瓶颈全在相机流
    - **「关掉显示路会更快吗」实测：不会**。显示路 ON 14.7–15.3 / OFF 14.8–15.4（双流互抢排除）。15.00 fps 是暗场景**砍半换曝光**（30→15）
    - **帧率协商**：`PreviewOutput.getSupportedFrameRates()` **必须 session.start() 之后**才非空（建流前 n=0）；设备报 `[1-30, 60-60]`；强制 30/60 未观察到可靠收益（场景漂移，不可作优劣结论）→ 保持不设
    - **加速器矩阵（release，只认 LANDED=）**：MS Lite 的 `gpu`/`kirin` **全部静默回落 CPU（GPU 通路是假的）**；det head `nnrt`→NPU 6.46 ms vs cpu 7.13；**rec `nnrt`→NPU 3.91 ms vs cpu 9.04（2.31×）**；cls fp32 nnrt 0.96 vs cpu 0.87 → **cls 留 CPU**
    - 速度 A/B（同会话 release）：det 24.86→12.15（2.05×）、infer 34.6→23.0（1.50×）、端到端 36.6→29.4（1.24×）、抖动 22–104→19–27 ms；❌「det/rec 争 NPU 会互拖」不成立
    - 🚨 **但「默认档改全 NPU」已撤回 —— `det=NPU` 会翻字符（A15 决定性）**：release 构建 + 黄金图 `hlpr-test.jpg`（期望串 `苏ED5172`）逐字符比对（`AUTO_SELFTEST=true` 跑 backendProbe/e2eMatrix，日志 `_evidence/A15-expected-string-matrix-release-20260918.log`）→ **det `nnrt` LANDED=NPU **match=000 全翻**（`苏ED5172`→`苏E05172`，D→0）；det cpu/gpu/kirin（均落 CPU）111 全对；rec nnrt（NPU）111 全对；cls/cls-fp32 全对**。backendProbe 同证：ms-cpu 111 / **ms-nnrt 000** / ncnn-cpu 111 / ncnn-vulkan 111
    - ✅ **最优且安全的分配 = 生产档 `det=CPU / rec=NPU / cls=CPU` 本身**：det 不能上 NPU（翻字符）；rec 必须上 NPU（2.31×）；cls 留 CPU（更快 + 非 fp32 上不去）。「全 NPU 档」1.5× 提速**不可用**（拿正确性换），只保留为测速档。**默认档已改回 `GEAR_PROD`**，装机复测 arrive=done=23.0–23.2 fps 丢帧 0
    - 📌 **方法论铁律**：换档/换模型后的验收对象必须是**期望串 `苏ED5172`**，不是「相机上碰巧读对一个」（那次 `皖A40675` 21次vs10次看起来"无损"，实为单样本运气，被 45+3 次黄金图翻字符推翻）
  - ⚠️ **口径警告**：`bench30_matrix` 与 `E2EMATRIX` 的历史数字**全是 debug 构建下测的** —— 内部可比，但**不能与 release 数字并列**；同档位跨时段也有热/负载波动（生产档同会话曾到 33–38 fps）
  - 坑：`switchGear` 原先写 `|| this.busy` 就 return → 30 fps 下 `busy` 几乎恒真，**按钮静默点不动**（改置 `switching` 标志、帧循环让路）；**检测器的 ncnn 是全局单例**（`g_param`/`g_bin`，不是槽位），必须 `lpr.ncnnLoadAsync` 装载，否则 mode=2 每帧报 `no cached model (load first)`
  - 未做：**精度未验收**（相机帧与静态图基准不是同一张图，不能据此称精度）；GPU 档 det 的 MS 会话仍以 CPU 占位
  - 原生 ArkUI 按钮可脚本点击：`hdc shell uitest uiInput click <x> <y>`，坐标从 `uitest dumpLayout -p /data/local/tmp/x.json` 取
- **OMG 转换台踩坑（重要）**：`omg` 有**包装脚本**（`ddk/tools/tools_omg/omg`，6.6 KB）和底层二进制两层，**必须走包装脚本**（它自己管库路径）；参数是 **`--hiai_version`**（默认 `master`），**不是 `--omg_version`**；`omg` 的 ELF 解释器写死 `/tmp/ld-linux-x86-64-2.35.so.2`，**WSL 重启会清空 `/tmp` 导致失效**，须重建；DDK 自带 glibc 2.35 **绝不能全局 export**（会把 `mkdir`/`tee` 等基础命令毒死，表现为段错误）；用法：`bash omg_conv/convert_one.sh <x.onnx> om_x <输入名> 1,3,H,W kirin9020`
- venv python 在 `...\envs\default\Scripts\python.exe`（非根目录！）
- build.sh 必须 unset NODE_OPTIONS；rawfile 模型 cp 后必 `attrib -r`（否则 restool 11204003）；绝不手删 intermediates
- **ncnn 内存 API 三种语义**：文本 param 用 `load_param_mem`（0=成功）；`load_param(unsigned char*)` 是**二进制**版；`load_model` 返回**消耗字节数**（0=失败）且只引用内存→数据须存活
- ncnn 打包需带 SONAME `libncnn.so.1` + `libomp.so`；头文件要 ncnn/src + ncnn_build_ohos/src
- onnx.load() 读不了 .onnx.json→ModelProto().ParseFromString
- 预览扩展名白名单 403：ORT worker 须 .js、模型 .onnx.json、wasmPaths 对象形式、create 显式 format:'onnx'
- 手机 Web 线程钉 6（12线程最慢）；DEMO_URL 钉 ?threads=6；演示前 AUTO_SELFTEST=false
- GitHub 直连不通（用户会开 VPN）；可达源 gitee；pip 需 --retries 5 --timeout 60
- rpv3 类别数取 outputShape[1,20,78]（字典77）；三模型输入实际 NHWC；改流水线必跑 hlpr_reference.py 对照
- 预览：node tools/serve.mjs . 8099（进程会回收，演示前探测）
- 用户会实时改 HTML（data-page-node-id）→先 Grep 再 Edit
- 页面：首屏≤900px 装完 nav+hero+pipeline
- **真机 Web 自证入口 = window.__BENCH**（threads/isolated/sab/p50 一次拿全）

## LaTeX 硬约束
IEEEtran 无 CJK→xeCJK+Fandol；标识符用 \path{}；verbatim ≤48字符/行；表格列宽按不可断最长片段；校验 pdf_render_check.py；tectonic 编译（build_paper.py）；图由 paper_figures.py 驱动

## 数据更正（已回灌，勿引旧口径）
截断缺陷输出 H468 非 H4668（LaTeX 附录保留 H4668）；残差由垂直缩放比例决定（=1→0.113%，≠1→29.1%）；OpenCV resize 低于精确双线性
