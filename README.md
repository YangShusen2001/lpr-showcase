# 基于 HyperLPR3 的端到端车牌识别 · 跨语言移植保真验证与麒麟 8020 异构后端实测

> 毕业设计 / 个人项目 · 作者：树森
> 一句话：**照片进，车牌字符串出**；四级流水线在浏览器内离线运行，并把「移植是否算得一样」当成研究对象，而不是只报一个延迟。

**项目名有两种写法，取用场合不同**（同步点见下方 ⚠️）：

| 场合 | 名称 |
|---|---|
| 论文 / 简历 | 基于 HyperLPR3 的端到端车牌识别与跨语言移植保真验证 |
| 展示页（`index.html`） | 基于 YOLOv5 与 CRNN-CTC 的纯前端端到端车牌识别 |
| 代码 / 工程 / 端侧 | 端到端车牌识别（麒麟 8020 / HarmonyOS） |

---

## 1. 这是什么

三阶段流水线，模型全部为**现成开源预训练权重**（不自训练）：

```
输入图 → ① 检测 YOLOv5-face 系（2.23 MB）
       → ② 透视矫正（单应 + bicubic，无网络）
       → ③ CRNN-CTC 序列识别（9.78 MB，字典 77 类）
       → ④ 颜色分类（1.53 MB）
       → 车牌字符串 + 置信度
```

三个主张：

1. **移植保真**：把 C++/Python 参考实现逐行移植为无依赖 ES Module，用 `onnxruntime-web`（WASM+SIMD）在浏览器内推理，**0 个外部网络请求**。逆向 OpenCV `cv2.resize` 的定点实现后，整数缩放比下与 Python 参考**逐通道完全一致**；**8/8 样本识别结果与参考一致**。
2. **端侧表征**：在 nova 14 Pro（麒麟 8020 / HarmonyOS 6.1）上以 MindSpore Lite → NNRT 与 ncnn-Vulkan 双栈对照，判据用**实际落点** `LANDED=` 而非"请求了哪个后端"。
3. **用独立标注集复核自己的判据**：保真度是比"一致"更强的主张，**但前提是被对齐的参照串本身是对的**。本项目在第三方真实集（n=1000，文件名即人工真值）上测出 rpv3 **90.6%**，并据此**推翻了自己一直当验收红线的单图"真值"**（详见 ADR-015）—— 这条自纠与上面两条同等重要。

## 2. 快速开始

```bash
# 静态预览（项目根）
node tools/serve.mjs . 8099          # → http://127.0.0.1:8099/index.html

# 页面自检（无头浏览器，需 playwright-core；验收：errors / consoleErrors 均为空）
node tools/web_check.mjs "/index.html?auto=1" "__PAGE" 240000
node tools/web_check.mjs "/demo.html?auto=1"  "__PAGE" 240000

# Python 参考实现（保真度的 ground truth）
.venv\Scripts\python.exe tools\hlpr_reference.py

# 论文编译（数据与图由 tools/paper_figures.py 驱动）
python tools/build_paper.py
```

页面三个入口：`index.html`（介绍页，含内嵌实时 Demo）· `demo.html`（独立实时演示页）· `mobile.html`（手机端演示）。

## 3. 目录结构

```
车牌识别/
├─ README.md            ← 你在这里（项目总入口）
├─ IDEA.md              原始需求（三条目标，调研结论见 docs/research/）
├─ index.html  demo.html  mobile.html
├─ assets/              页面资产（约定与「不能改回去」的文件名 → assets/README.md）
├─ docs/
│   ├─ INDEX.md         文档地图 + ADR 索引
│   ├─ adr/             ADR-001 ~ ADR-015，全部决策与踩坑记录
│   ─ research/        可行性调研（外部资料检索结论）
├─ paper/
│   ├─ ieee-lpr-paper.md    IEEE 格式论文（Markdown 主稿）
│   ├─ latex-en/  latex-zh/ 两份**手工誊写的独立稿**（main.tex / main.pdf，非生成物）
│   └─ figures/             7 张图的 pdf + png
├─ resume/resume-bullets.md 简历条目（每个数字都可当场复现）
├─ tools/               74 个脚本 + README.md（按 docstring 整理的清单，需手工维护）
├─ _evidence/           原始证据（日志 / 张量比对 / 探针输出）+ INDEX.md
└─ _archive/2026-09-18/ 归档的历史备份与实验产物（可整目录删除）
```

## 4. 核心数字速查

> 延迟一律为 **warm-up 后取 p50**，且必须标 `build=release`（历史 `bench30_matrix` /
> `E2EMATRIX` 数字是 debug 构建测的，内部可比但**不能与 release 并列**）。
> 旧口径「13.11× 作为本项目的整体加速比」与 243.8 ms 已作废，勿引 ——
> 但 13.11× 作为 **ResNet-50 骨干网的 NPU/CPU 比值**仍是有效实测（见下表末行）。

| 项 | 结果 | 出处 |
|---|---|---|
| 移植一致性 | 8/8 样本识别串与 Python 参考一致（**参照串＝rpv3 自身输出，非真值**） | ADR-002 |
| 整数缩放比 | 逐通道完全一致；非整数比残余 ≤1 量化级、输出中性 | ADR-002 §C |
| **识别精度** | rpv3 **906/1000 = 90.6%**，第三方真实裁剪图集（文件名即人工真值） | A16 |
| 替换候选 | LPRNet **888/1000 = 88.8%**（McNemar **p=0.1788** 不显著）→ **不换** | A16 |
| 长度错误率 | rpv3 **1.0%** vs LPRNet **7.0%**（rpv3 优势项） | A16 |
| Web 真机（6 线程 WASM） | 端到端 **88.3 ms**（单线程 126 ms） | ADR-010 |
| 检测器 NPU（裸 head，fp16） | **5.35 ms**（CPU 7.66 ms，1.43×，release 30×p50；但翻字符不可用） | ADR-009 · A17 |
| 识别器 NPU（混合执行，fp16） | **3.99 ms**（CPU 9.13 ms，2.29×，release 30×p50） | ADR-004/005 · A17 |
| NNRT-fp32 对照 | **det fp32 档仍翻字符（3/3）、rec fp32 档 3/3 正确** →「det 不能上 NPU」去掉 fp16 限定 | A17（2026-09-20） |
| GPU | Maleoon 920C（0x19e5，Vulkan 1.3.275，fp16 在位） | ADR-008 |
| 相机实时吞吐 | 暗场景到达率 19.9–20.1 fps、明亮 26.9–27.1 fps，`dropped=0`；**「相机上限=15fps、推理侧已非瓶颈」已撤回待复测** | A14 / ADR-015 |
| 通用骨干加速比 | ResNet-50 **13.11×** → INT8 MobileNetV2 **0.98×** | ADR-003 §5 |

**三条必须一起说出口的结论**：

1. 加速比随图规模单调下降 → 小图在 NPU 上拿不到收益（固定委托开销摊不薄）。
2. NPU 的 0.08% 数值漂移经 anchor 放大 ≈10 px **会翻字符**（rpv3 参照串 `苏ED5172` →
   `苏E05172`）→ 生产链定为 `det=CPU / rec=NPU / cls=CPU`。
   ⚠️ **该漂移是 fp16 条件下的观测**：`ms_engine.cpp` 的 `addNnrt()` 恒传 `fp16=true`，
   NNRT-fp32 **从未测过** → 严格表述是「det 不能在 NPU-**fp16** 上用」（ADR-015）。
3. `1000/p50(frameMs)` 是**服务时间理论值不是吞吐**；对外只准看到达率 / 完成率两列
   （A14 曾把 34.5 / 23.8 / 12.8 当 fps 报，已作废）。

### 三模型 × 后端能力矩阵（2026-09-18，全部真机实测）

| 模型 | MS Lite CPU | MS Lite NPU | MS Lite GPU | ncnn CPU | **ncnn Vulkan（GPU）** |
|---|---|---|---|---|---|
| 检测（裸 head） | ✅ match=1 | ⚠️ 落 NPU 但**翻字符** | ❌ 编译期判否 | ✅ 218 层 | ✅ match=1 |
| 识别 rpv3 | ✅ | ✅ **match=1，最快（该段 7.5 ms）** | ❌ | ✅ 252 层（19–24 ms） | ✅ match=1（52–72 ms） |
| 分类 litemodel | ✅ | ✅ 仅 **fp32 变体**能上（fp16 构不出 kernel） |  | ✅ 78 层（4.6–7.9 ms） | ✅ match=1（11–15 ms） |

⚠️ **GPU 不是加速器**：三个模型在 Vulkan 上都能跑通且与 CPU/MS 结果一致，但**都比 CPU 慢**
（小图提交开销主导，同 ADR-003 缩放律）。GPU 这一格的价值是**通路闭合 + 保真对照的第三个数据点**，
口径必须写「通路成立且数值保真」，**不许写「GPU 加速」**。证据：`_evidence/A9-gpu-rec-cls-20260918.md`。

**但这一格的证据强度此前不够下结论，已在补（ADR-015 §3）**：

- 表里「MS Lite GPU ❌ 编译期判否」证明的是**当前链接的这份 `libmindspore_lite.so`** 没编
  GPU delegate，**不等于麒麟不支持** —— 官方 MindSpore Lite Kit 简介明写「支持通用硬件
  CPU/GPU 与 NNRt AI 加速硬件之间的模型异构推理」，故须复核设备侧 OpenCL 在位性。
- ncnn 行的 `effectiveVulkan` 只是选项回显。现已补 `vkLayers=X/Y`（带 Vulkan 实现的层数）：
  **X<Y 时该格是「GPU + 部分层回落 CPU」，不得当纯 GPU 数据点**。
- MS 行的回落现在显式可读了（`engine=ms req=gpu fallback=GPU:fp16`），不再需要读者
  相信一句结论。
- 「GPU 唯一通路 = ncnn-Vulkan」只证到**框架层**没有别的，未证到**硬件不可能**。
  要钉死需 ncnn 层数 ladder 探针测出每层固定提交开销 `a`，与 ADR-003 的缩放律外推联立，
  给出「GPU 有收益所需的最小单算子计算量」。

**保真精度分两层（别混着说）**：分类是**逐位**一致（`0.0001|0.9998|0.0001`）；
识别只到**符号级**——车牌串与字符序列一致，但逐字符概率与 MS-NPU 差 ~0.5–1%。
且 **ncnn 的 CPU 与 Vulkan 输出彼此完全相同**，所以那点差是 ncnn↔MindSpore 的框架差，
不是 CPU↔GPU 的差。

### 其它加速路线：已实测评估（2026-09-18，见 ADR-012 / ADR-013 / A10 / A11）

问过「CANN 行不行」。设备侧探针的答复：

| 路线 | 状态 |
|---|---|
| NPU：**MindSpore Lite + NNRT delegate**（现状） | ✅ 已跑通，且 **就是官方 NNRt 底座** |
| NPU：NNRt 直连（标准 NDK，无需 HMS） | ⚠️ 库可加载、设备可枚举；但**离线入口不收 `.ms`**（`compat=1` / `build_rc=1`），要 `.om` |
| NPU：CANN Kit / HiAI Foundation（HMS） | ⚠️ 同上，额外提供显式设备序 / 调优 / 单算子；门槛也是 `.om` |
| NPU：**CANN 用厂商 `.om` 实测** | ✅ **端到端跑通**（A11）：`compat=0`→`build_rc=0`→`run_rc=0`，IO `[1x3x227x227]`→`[1x1000x1x1]`，稳态 **0.92 ms** |
| NPU：MNN/TNN 的华为后端 | ❌ 官方定位 Android |
| GPU：Vulkan（ncnn） | ✅ 唯一通路，已做完，不加速 |
| GPU：NNRt / OpenCL | ❌ NNRt 枚举里**没有 GPU 设备**；Maleoon 未暴露 OpenCL |

🔑 最有价值的一条：**NNRt 枚举出的设备名与 MS Lite 日志里的落点设备名逐字相同**
（`NPU_ohos.boot.hardware.kirin8020_v2_0`）—— 现有 NPU 通道不是绕路，**直连没有性能便宜可捡**。
`.om` 转换器（OMG）需华为账号 + 64 位 Linux，是唯一未验证的缺口。

**但「CANN 一点办法都没有」是错的。** 拿一份华为官方公开 codelab 里的真 `.om` 喂进同一条 NNRt 入口：
`compat=0` / `build_rc=0` / `run_rc=0`，读得出 IO 规格，稳态 **0.92 ms**；而**同一批 `.ms` 3/3 仍被拒**。
→ **差别只有文件格式，不是权限也不是能力。** 缺口收窄成一句话：
**我们缺的是把 ONNX 转成 `.om` 的 OMG 工具**（要华为账号 + 64 位 Linux + 与设备 CANN
`108.631.120.010` 版本对齐）。证据：`_evidence/A11-cann-om-end-to-end-20260918.md`（ADR-013）。

## 5. 未完成 / 下一步（优先级序）

> 2026-09-18 重排：原第 1、2、4 条的完成度已由 A12/A14 改写，见下。

0. **NNRT-fp32 对照（最高优先，因为它可能推翻一条头条结论）**：`addNnrt()` 恒传
   `fp16=true`，所以"det 上 NPU 会翻字符"从未在 fp32 下检验过。`nnrt_fp32` 档已加进
   `ms_engine.cpp`，待真机跑：30× p50 + **同一份输入张量**的逐元素 L2 + 参照串差分。
   做不到 30× 零差异就正式放弃「全 NPU」叙事。
1. **帧率归因**（~~正式性能矩阵~~ 已由 A12 拿到 30× 口径、~~letterbox 是大头~~ 已由 A14
   推翻并修掉真凶）：现在缺的是**分段预算**——`det` 段流水线内 24.86 ms 而纯推理只 7.13 ms，
   中间 17.7 ms 到底是 letterbox / NHWC 打包 / decode / NMS，从未拆开过（ADR-009 §5 的
   待办）。需 ≥5 轮 × 每轮 ≥200 帧、同会话 ABBA 交替、每窗口标热档（`@ohos.thermal`）。
2. **相机吞吐口径结案**：「只数帧不取帧」档 + 固定片元回放台，把服务时间与到达率解耦，
   复测 A14 §4.2.2 那条被撤回的"相机上限"。
3. **GPU 钉死**：ncnn `create_pipeline_cache()` + rep0 单独插桩 + 层数 ladder 扫
   → 输出「每层固定提交开销 a / 加速比–尺寸拐点」，据此给正向或负向的决定性结论。
4. **候选模型测试台**：`tools/model_gate.py`（合并 `npu_scan.py` 的 rank/perm 判据 +
   16 倍数通道画像 + 算子黑名单）→ 转换三链（ms / ncnn / om）→ 真机矩阵 → 自动进表。
   含**同权重×16补齐的因果对照实验**（cls 19.2%→100%，把 ADR-014 的相关性升级成因果）。
5. **三模型的逐元素保真对照**：目前 rec 以 `chars`/`code`（符号级）为判据，cls 已到逐位数值一致；
   要做 rec 的逐元素 L2 对照，需给 MS 与 ncnn 两个 runner 喂**同一份输入张量**。
6. 多摄：**并发在手机端做不出来** → 改顺序调度 + 命中锁定（调研见 `docs/research/`）。
7. 论文补数据来源声明（精度口径已换，见 §8）；端侧跑分表补「框架」列（跨栈比较只能当量级参考）。
8. 再压 HAP：`models/` 里有一半是图手术实验产物（探针页面用得上，可按需裁）。
   ⚠️ release 构建的 signed HAP 目前 76 MB（A8 记录的 58.3 MB 是另一档构建），需查符号是否被打包。

## 6. ⚠️ 已知待修（口径一致性）

- ~~`index.html` 的 `<title>` / `<h1>` 与论文口径不一致~~ **已按 §1 的名称表统一**
  （展示页保「纯前端」，论文/简历保「移植保真」）。
- **端侧跑分表必须补一列「框架」**：NPU 走 MindSpore Lite、GPU 只有 ncnn-Vulkan，
  跨栈比较只能当量级参考，可比的是同栈比值。
- **牌色口径冲突未裁决**（ADR-015 只登记不裁决）：`hlpr-test.jpg` 的 cls 模型输出是
  **蓝牌**（ADR-004 §末、A15 日志 `colour=蓝牌`、`index.html` 与 `demo.js` 的展示标签），
  而 A16 的牌面像素测量判**真绿牌**（`tools/plate_face_colour.py`，G−B=+39 且白字中性）。
  这是"模型输出"与"像素测量"两种口径，**在裁决前凡提该样本牌色处必须并列两个口径**。
  另：旧 `tools/plate_colour_analysis.py` 的按色细分（含"绿 hue 37.2%"）**作废**，脚本已标 DEPRECATED。
- **验收基准已换**：凡"精度/读对了"一律引用 `_dataset/real/crops`（n=1000，文件名即人工真值）；
  单张黄金图只作保真参照。n=5 那批判据脚本（`lprnet_golden_check.py`、`lprnet_vs_rpv3.py`）
  已加作废声明 —— 它们的"真值"是被测系统自己的输出。

## 7. 工程约定（改代码前必读）

完整清单见 `assets/README.md`（文件名白名单）、`tools/README.md`（脚本）、
`.workbuddy/memory/MEMORY.md`（长期记忆，含全部陷阱）。最容易踩的七条：

- **`match=` 不是"读对了"**：`Index.ets` 里的 `REFERENCE_CODE_RPV3` 是 rpv3 自身的输出，
  只用于抓"换后端引入的差异"。**任何精度/正确性声明只能引 `_dataset/real/crops`（n=1000）**。
  细则见 ADR-015。
- **NPU 数字默认全是 fp16**：`ms_engine.cpp` 的 `addNnrt()` 恒传 `fp16=true`；要谈"det 能不能上
  NPU"必须先跑 `nnrt_fp32` 档，否则结论只覆盖 fp16。CPU 档则**从不设性能模式、线程数硬编码 4**。
- **`kMaxSessions=20` 已被 16 组合矩阵用满**：往 `E2E_BACKENDS` 里加一档就会静默
  `BUILD-FAIL`（会话键 = `模型文件|后端`，id 永不回收）。扩档前先做"跑完即卸 + 每候选重启进程"。
- **手机端已是纯原生**（2026-09-18）：`lpr-harmony` 的 `Index.ets` 不含 WebView，
  两个标签页 = 原生演示 + 探针控制台。旧 ArkWeb 实现存
  `C:\Users\26671\lpr-harmony\reference\Index.arkweb-era.ets`（工程外，不编译）。
- **ArkTS 侧一律用 `lpr.*Async`**：同步版本会阻塞 UI 线程，连续占用 >3 s/>6 s 会被
  系统 watchdog 判 `THREAD_BLOCK` 并 SIGKILL（实测 `uvLoopTask 8887 ms`）。
  会话名**必须传模型文件路径**（原生侧按 (文件,后端) 去重，传角色标签会去重失效）。
- **`tools/sync_rawfile.py` 的 MANIFEST 决定 HAP 里有什么**：现在是「3 张样本」+
  `KEEP=["models"]`。它是「先 rmtree 再重建」的幂等实现，往 rawfile 里手工加文件
  会在下次同步被静默删掉（登记进 MANIFEST/EXTRA 才留得住）。
- **改文件名会静默弄坏 Demo**：`.onnx.json` / `.js` 的后缀是静态托管白名单逼出来的，别改回 `.onnx` / `.mjs`；
  三处引用必须同步：`assets/js/demo.js`（ORT 入口 + `wasmPaths` 对象形式 + `format:'onnx'`）、
  `tools/hlpr_reference.py`（双后缀兜底）、`assets/samples/` 的相对路径基址。
- **演示前**：`AUTO_SELFTEST=false`，线程钉 6，预览服务先探测（进程会被回收）。

## 8. 数据与合规

- 模型：HyperLPR3 开源预训练权重（Apache-2.0），**刻意不复现训练**，取舍论证见 ADR-001（含 YOLOv9-t 三档尺寸的证伪记录）。
- 样本：上游仓库真实车牌照片 + 合成车牌数据（`tools/gen_plate_dataset.py`，**合成数据只用于链路回归，不用于报准确率**）。
- 精度基准：第三方公开集 `sirius-ai/LPRNet_Pytorch` 的 `data/test/` **1000 张真实裁剪图**
  （文件名即人工真值，非合成），落 `_dataset/real/crops/`，许可＝公开仓库随附测试集、
  仅学术评测。**实际内容 2.52 MB**（`du` 报 4.2 MB 是 4K 块对齐，勿引用）。
- 性能/精度数字**全部为实测**，原始日志在 `_evidence/`。
  ~~本项目刻意不报准确率（无标注基准）~~ → **2026-09-18 口径更新（ADR-015）**：
  标注基准已建立（n=1000），准确率**可以也应该报** —— rpv3 90.6%、LPRNet 88.8%、
  McNemar p=0.1788。**必须同时声明的边界**：该集 1000 张**全为 7 字符，新能源 8 位牌 0 张**，
  且是 LPRNet 自己的主场分布；不做任何"实际路况准确率"的外推。

## 9. 归档说明

`_archive/2026-09-18/` 收纳了历史备份、LaTeX 探针、断链资产（jsep.wasm 26 MB、YOLOv9-t 模型 26 MB）、
中间张量 dump（33 MB）等 —— **项目有效内容从 144 MB 降到 38.8 MB**。
恢复或彻底删除的方法见 `_archive/2026-09-18/README.md`。