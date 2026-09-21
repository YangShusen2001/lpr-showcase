# ADR-004：三后端可用性矩阵、NPU 拒绝边界与跨源隔离实证

- **状态**：已接受（结论均由真机实测 + 官方资料双向确认）
- **日期**：2026-09-17
- **前置**：ADR-003（本 ADR 完成其 §7 的全部后续动作，并替换其 §6 的边界声明）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135(SP8C00E120R7P5)
- **宿主**：自研 ArkTS + NAPI C++ 应用（`com.shusen.lprdemo`），MindSpore Lite 2.6.0 NDK
- **修订**：2026-09-17 晚 —— 新增 §6.1（Web 路径 = CPU 多线程，NPU/GPU 不参与的事实边界）、
  §6.2（PC 编排 + 手机计算 + 手机摄像头 三段架构与 CDP 通路实证）、§8 勾选与补充

---

## 1. 问题

ADR-003 交付的是**方法学**，评测对象是通用骨干网（resnet / mobilenet / yolov8n），
并留下待办：把本项目三个车牌模型送上端侧。

本 ADR 把待办做完，并回答一个更硬的问题：

> **在麒麟 8020 + HarmonyOS 上，「榨干 NPU 和 GPU」的物理边界究竟在哪？**

即：三个车牌模型各能落到哪个后端？落不上的那些，是**配置问题**（可调）还是**边界问题**（不可调）？

---

## 2. 方法

### 2.1 (模型 × 后端) 全矩阵单次遍历

在 ADR-003 §3.4 的 CPU-diff 方法上扩展：

- 后端四档：`nnrt`（NNRT→NPU）/ `gpu`（MS Lite GPU）/ `kirin`（HiAI 直连）/ `cpu`（4 线程基线）
- 每个组合**独立会话**（规避 ADR-003 §3.1 的析构崩溃）
- 每组合打印：实际落点 `LANDED=` + p50/mean + 输出 L2 校验和 + `maxAbs` + `dtype`
- 输入用确定性填充 `x[j] = ((j × 2654435761) mod 1000) / 1000.0`，使跨后端输出可比
- 判据：**`LANDED=` 是实际落地后端**（不是请求的后端）——这是识别静默回退的唯一可靠手段

被测 7 个 `.ms`：3 个正式模型 + 4 个图手术变体（y5fu 的 rank 手术版本、litemodel 的 fp32 版）。

### 2.2 转换前的图结构解剖

先用 `_evidence/graph_anatomy.py` 扫 ONNX 图，统计**常量张量的 rank 分布**。
这个动作后来成为定位 y5fu 失败的钥匙——**在转换之前就知道它会失败**。

### 2.3 Web 层能力探针

ArkWeb 内嵌 `webprobe.html`，用 `runJavaScript` 读回 DOM。
WASM 特性检测的字节码**逐字节取自 `wasm-feature-detect` 官方实现**（见 §5.2 的方法学备注）。

---

## 3. 实测矩阵（麒麟 8020，两轮复现）

| 模型 | 请求后端 | 实际落点 | p50 (ms) | 输出一致性 |
|---|---|---|---|---|
| `rpv3_mdict_160_r3`（识别） | `nnrt` | **NPU** `NPU_ohos.boot.hardware.kirin8020_v2_0` | **2.97 – 4.17** | PASS |
| `rpv3_mdict_160_r3` | `cpu` | CPU | 8.97 – 9.02 | PASS |
| `litemodel_cls_96x_r1_fp32`（分类） | `nnrt` | **NPU** | 0.96 – 1.09 | PASS |
| `litemodel_cls_96x_r1_fp32` | `cpu` | CPU | 0.57 – 0.65 | PASS |
| `litemodel_cls_96x_r1`（fp16） | `nnrt` | **CPU**（回落） | 0.70 | PASS |
| `y5fu_320x_sim`（检测） | `nnrt` / `gpu` / `kirin` | **CPU**（全部回落） | 7.52 – 9.56 | PASS |
| `y5fu` 4 个图手术变体 | `nnrt` / `gpu` | **CPU**（全部回落） | 7.62 – 8.34 | PASS |

**加速比（仅识别模型成立）**：NPU 2.97–4.17 ms vs CPU 8.97–9.02 ms → **2.2× – 3.0×**。

> **分类模型上 NPU 反而更慢**（1.09 vs 0.57 ms）。与 ADR-003 §5 的结论同构：
> 委托开销在小图上摊不薄。**不要默认「上了 NPU 就一定快」。**

---

## 4. 三个根因（逐条附日志证据）

### 4.1 检测模型无法上 NPU：常量张量 rank 越界（**边界问题，非配置问题**）

`NNRt_HiAIAdapter` 日志：

```
Tensorptr format <private> not support.
get weight shape is empty.
CreateConstOp failed, convert const tensor for <private> failed.
```

图解剖给出对应事实（`_evidence/graph-anatomy.json`）：y5fu 共 416 节点，含
**12 个 rank-5 常量**，形如 `[1,3,40,40,15]` / `[1,3,40,40,2]` / `[1,3,20,20,2]` / `[1,3,10,10,15]`，
全部是 YOLOv5 的 **anchor-grid decode 被烘进计算图**的产物。

即：HiAI 的常量转换路径读不出 rank-5 的 shape（`get weight shape is empty`），
`CreateConstOp` 失败，整个子图回退 CPU。

**已证伪的补救路径**（四条，全部实测失败）：

| 手术 | 做法 | 结果 |
|---|---|---|
| `s1` | rank-0 标量 → rank-1 | 仍落 CPU |
| `s1r4` | rank-0 → rank-1，且 rank-5 → rank-4 | 仍落 CPU |
| `flat` | rank-5 展平为 rank-2 | 仍落 CPU |
| `surgery` | rank-5 拆成 rank-4 + 广播 | 仍落 CPU |

关键反证：手术本身数值**逐位等价**（`_evidence/surgery-equivalence.json`，
`max_abs_diff = 0`）。所以失败不是「改坏了」，而是**改了也进不去**。

**结论**：这不是再调参数能解决的。唯一出路是**模型侧改造**——把 anchor-grid decode
移出计算图，导出「裸 head」版本。

> **行业印证（检索所得）**：在华为 NPU 上部署 YOLO，业界通行做法正是**改模型结构**而非调宿主配置——
> 昇腾社区案例把 `MaxPool2d` 换成 `AvgPool2d`（om 不支持 maxpool）、把 Focus 层拆成
> `strided_slice` + `concat`、用近似结构替换早期 CANN 不支持的 SiLU。
> 本项目的「移出 anchor-grid decode」与这些属同一类动作。

### 4.2 原生 GPU 后端在麒麟 8020 上不存在（**边界问题**）

```
[inner_context.cc:207] IsValid# GPU is not supported.
[inner_context.cc:213] IsValid# NPU is not supported.     ← 对 gpu / kirin 档位
```

`gpu` 与 `kirin` 两个档位**100% 回落 CPU**，且不是运行时静默降级——是**编译期直接判否**。

原因：MindSpore Lite 的 GPU 后端基于 **OpenCL**（其 FAQ 单列「OpenCL GPU 推理问题」一节）。
麒麟 8020 的 Maleoon GPU 未向该路径暴露 OpenCL 设备。
`kirin` 档位（HiAI 直连）另需系统预置 `/system/lib64/libhiai_ir_infershape.so`，
本机（MIA-AL00 6.1.0.135）缺失。

### 4.3 ArkWeb 里 ORT 的 GPU 路径被上游删除（**边界问题，双证据**）

**(a) 实测**：GPU 物理存在且 WebGL2 可用（`Maleoon 920C`、`WebGL 2.0 (OpenGL ES 3.0 Chromium)`），
但 `navigator.gpu` 存在而 `requestAdapter()` 返回 **NULL**。

**(b) 官方**：ONNX Runtime **v1.29.0**（**正是本项目锁定并随页面自托管的版本**）
release notes 明确宣布 *deprecation of WebGL and JSEP*：

> "onnxruntime-web has announced the deprecation of WebGL and JSEP.
> The native WebGPU EP is the recommended path going forward."（#29716、#31683）

`onnxruntime-web/webgl` 子路径被移除，官方迁移指南写得很直白：
*「不存在自动重定向，因为 WebGL 构建没有 WASM 回退，且 WebGPU 依赖 navigator.gpu。」*

**两条证据合成的硬结论**：
**在 HarmonyOS ArkWeb 上，ORT 不存在可用的 GPU 加速路径**——WebGL EP 已被上游删除，
WebGPU 又拿不到 adapter。本项目页面选择 WASM 后端**不是妥协，是该平台上唯一可用的 ORT 配置**。

---

## 5. Web 层能力实测（ArkWeb / Chromium M132）

### 5.1 能力表

| 能力 | 结果 |
|---|---|
| UA | `OpenHarmony 6.1 · Chrome/132.0.0.0 · ArkWeb/6.1.0.120` |
| `hardwareConcurrency` / `deviceMemory` | **12** / 8 |
| WebGL2 / WebGL1 | YES / YES |
| GPU 型号 | `HUAWEI Maleoon 920C` |
| `MAX_TEXTURE_SIZE` | 16384 |
| WebGPU | `navigator.gpu` 有，**`requestAdapter()` = NULL** |
| WASM SIMD / bulkMemory / truncSat | **true / true / true** |
| WASM signExtensions / multiValue / referenceTypes / bigInt | **全 true** |
| WASM threads.atomics（模块合法性） | **true** |
| GL 片段吞吐（1024², 400 iters, 60 帧） | 22.6 ms / 0.377 ms·帧⁻¹ / **6675 GFLOPS** |

### 5.2 方法学备注（自我纠错，务必保留）

初版探针的 WASM 特性检测用的是**手写字节码**，结果报出 `bulkMemory=false`、`truncSat=false`。
这是**假结论**：

- `memory.copy` 后面只写了一个 memidx 字节（`252,10,0,11`），官方是两个（`252,10,0,0,11`），
  且段长算成 13 而非 14；
- `trunc_sat` 声明了 `result i32` 却未消费栈顶（缺 `26` = drop）。

→ 模块非法 → `WebAssembly.validate()` 返回 false → **误报「不支持」**。

**更正做法**：字节码逐字节取自 `wasm-feature-detect`（Thomas Steiner 维护，社区事实标准），
并**先在本机 Node 22 / V8 12.4 上校验 `validate=true`** 后才上机
（`wasm_bytes_verified.json`：8 个检测器全部 `validate=true, compile=true`）。

**教训**：字节级构造必须引用权威实现，不能凭记忆手写。**一个字节的偏差 = 一个假结论。**

### 5.3 跨源隔离：从「不可用」到「可开」（本轮最有价值的发现）

修正后的探针精确定位了多线程缺口的性质：

| 探针项 | 走 `$rawfile` 协议 | 走虚拟 https 源 + COOP/COEP |
|---|---|---|
| `origin` | `resource://…` | **`https://lpr.local`** |
| `crossOriginIsolated` | false | **true** |
| `SharedArrayBuffer` | false | **true** |
| `wasm.threads.sabTransferable` | false | **true** |

即：**ArkWeb 的 WASM 特性本身是完整的**，多线程不可用的原因不是 V8 不支持线程
（`threads.atomics` 模块一直合法），而是**页面没有跨源隔离**。

而响应头**在我们自己手里**：宿主用 `onInterceptRequest` 从 rawfile 回灌字节时，
顺带注入三个头即可（官方 API 见华为开发者 FAQ `faqs-arkweb-4`，`setResponseHeader(Header[])`）：

```
Cross-Origin-Opener-Policy:   same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

**实测结果：只加这三个头，`crossOriginIsolated` 由 false 翻为 true，
`SharedArrayBuffer` 与 `sabTransferable` 同步变 true。**

这条路径的意义：网页版（`demo.html` 那句「多线程 WASM 需要 COOP/COEP 响应头」）受制于
服务器配置；但在**鸿蒙原生壳里，响应头由我们自己产生** —— 12 核 CPU 的 WASM 多线程
从「平台限制」变成了「我们自己没配」。

### 5.4 把跨源隔离换成真实收益：WASM 线程数扫描（两轮复现）

§5.3 只证明了「能开」。本节把它变成数字。`demo.js` 新增两个开关：
`?threads=N`（钉住线程数）与 `?bench=N`（warm-up 1 次 + 正式 N 次，报 p50/mean/min/max）。
鸿蒙壳按 1→2→4→6→8→12→1 逐档 `loadUrl` 重载页面，**同一台设备、同一张 1140×456 样本**
（`assets/samples/scene-2.jpg`）、每档 30 次，连测两轮：

| 线程数 | 第 1 轮 p50 (ms) | 第 2 轮 p50 (ms) |
|---|---|---|
| 1 | 126.2 | 126.5 |
| 2 | 107.9 | 107.9 |
| 4 | 108.1 | 110.6 |
| **6** | **86.7** | **84.2** |
| 8 | 121.3 | 123.9 |
| 12 | 162.3 | 157.7 |
| 1（末档复测） | 125.0 | 126.0 |

**结论（与直觉相反，故值得记录）**：

1. **最优档是 6 线程，不是 12**：≈85 ms vs 单线程 ≈126 ms → **1.5×**（两轮分别 1.46× / 1.50×）。
2. **12 线程比单线程还慢**（157–162 ms）。候选解释（**未单独隔离验证**）：
   麒麟 8020 的 12 核是**异构**的，工作均摊到小核会拖长关键路径；
   且 12 线程满载更容易触发**热降频**。二者叠加即可解释 12 档的塌陷。
3. **首末两档同为 1 线程，p50 差 ≤1 ms**（126.2/126.5 vs 125.0/126.0）——
   整轮扫描的热漂移可忽略，说明上表的档间差异是真实差异，不是漂移。
4. **对照实验（本机 PC 无头 Chromium，同为 12 逻辑核、同样 `crossOriginIsolated=true`）**：
   1 线程 p50 **90.9 ms** → 12 线程 p50 **81.2 ms**，**仅 1.12×**。
   → 多线程收益来自**核间异构**，不是「核多就快」；同构/SMT 核上 12 线程几乎白给。

证据：`_evidence/mt_sweep_round1.txt` / `_evidence/mt_sweep_round2.txt`（`hilog` 原文，由
`tools/harmony/run_mt_sweep.py` 按正文去重采集）、
`_evidence/web_demo_html_bench_30.json` 与 `_evidence/web_demo_html_bench_30_threads_1.json`（PC 对照）。

> **顺带作废一个旧数字**：页面与论文原先标的「243.8 ms 端到端延迟」在全仓查无证据文件，
> 且与 warm harness 实测（单线程 124.5–126 ms）不符，判定为**冷单次值**（含首次 JIT 与内核编译）。
> 本 ADR 起，端到端延迟一律按 **warm-up 后 30 次 p50** 口径给出。

### 5.5 开启多线程的副作用：COEP 会拦下跨源子资源（踩坑记录）

`tools/serve.mjs` 补上三个头之后 `demo.html` 正常，但 `index.html` 报出 6 条
`ERR_BLOCKED_BY_RESPONSE.NotSameOriginAfterDefaultedToSameOriginByCoep`。

定位结果：**不是响应头的问题，是页面里的绝对地址**。外部编辑器把 `demo.js` 的运行时输出
（`#samples` 里 6 个 `<img>`）序列化回了 `index.html`，并把相对路径解析成了它自己的预览地址
`http://127.0.0.1:10504/static-html/<hash>/assets/samples/*.jpg`。该地址与页面不同源、
又不带 `Cross-Origin-Resource-Policy`，在 `require-corp` 下必然被拒。

修法：把这 6 个 `src` 改回相对路径 `assets/samples/*.jpg`。
`tools/sync_rawfile.py` 的 `LOCAL_ABS` 本来就在同步到手机时剥这类地址——**同一个坑的第二次现身**。
诊断脚本固化为 `tools/probe_blocked.mjs`：同时从 `requestfailed` 与响应头两个视角列清单，
以后改响应头可一键回归。

---

## 6. 结论：「榨干 NPU 和 GPU」的可落地边界

| 后端 | 物理可用 | 本项目实际达成 | 性质 |
|---|---|---|---|
| **NPU** | 可用 | 识别模型 **2.2–3.0×**；分类模型上 NPU 反而慢；**检测模型 0×**（回落 CPU） | **部分榨干** |
| 原生 GPU（MS Lite） | **不存在** | — | 边界（OpenCL 未暴露） |
| Web GPU 推理（ORT） | **不存在** | — | 边界（WebGL EP 被上游删除 + WebGPU 无 adapter） |
| Web GPU 渲染（WebGL2） | 可用 | Maleoon 920C，6675 GFLOPS | 仅渲染，不能做推理加速 |
| **Web CPU 多线程** | **可开** | 线程数自适应已落地；**6 线程 ≈85 ms，相对单线程 1.5×**（§5.4） | **已榨到（本轮完成）** |

**一句话**：NPU 榨到了一半，GPU 两条路都是死的，Web 侧多线程这条路是通的，且已换成 1.5× 的真实收益。

### 6.1 Web 演示路径与 NPU/GPU 的事实边界（本轮新增，回应「为什么没用上 NPU/GPU」）

演示页（`mobile.html` / `demo.html`）里跑的推理，**100% 是 ORT WASM，只吃 CPU**。
把它接上 NPU 或 GPU **在 Web 层物理上做不到**，理由已在 §4.2 / §4.3 逐条实证：

| 想在 Web 路径上用 | 现实 | 依据 |
|---|---|---|
| GPU 推理（WebGL EP） | 该 EP 已在 ORT **v1.29.0**（本项目自托管版本）被上游删除 | §4.3(b) |
| GPU 推理（WebGPU EP） | `navigator.gpu` 有，但 `requestAdapter()` 返回 **NULL**，无设备可用 | §4.3(a) |
| NPU 推理（Web 侧） | **不存在任何 Web API 能把 ORT 会话绑到 NPU**；NPU 只有原生 MindSpore Lite 路径可达 | §3 / §4.1 |

**因此「Web 端侧推理 = CPU 多线程」不是配置没调，而是本平台的边界。**
想拿到 NPU 的 2.2–3.0×，必须**另起一条原生 MindSpore Lite 路径**（且检测模型还得先做 §4.1 的裸 head 改造）。
演示页本身到不了 NPU——**这一点必须对面试官/评审说清楚，不能含糊**。

**「卡」的归因**：不是没开 NPU/GPU，而是线程数选错。手机 `crossOriginIsolated=true` 且
`hardwareConcurrency=12`，`demo.js` 的自适应会取满 **12 线程**；而 §5.4 已证 12 线程是**最差档**
（157–162 ms，比单线程还慢）——12 线程占满 12 个异构核，且与 ArkWeb 的 UI/渲染线程抢核，
叠加热降频。
**修法**：鸿蒙壳把演示页 URL 钉死为 `?threads=6`（`Index.ets` 的 `DEMO_URL`），
直接落到 §5.4 的最优档；实测 222 ms（含 PC 推图往返）明显比 12 线程流畅。

### 6.2 「PC 编排 + 手机计算 + 手机摄像头」三段架构（本轮新增，已实证）

需求原话：「用电脑调用 8020 的 NPU，然后再电脑上跑，再调用手机的摄像头」。
落地形态是**三段分工**，且**已跑通**：

```
┌─ PC（编排层）───────────────┐   ┌─ 手机 nova 14 Pro（算力 + 采集）──────────┐
│ pc_drive_phone.mjs          │   │ ArkWeb 跑 mobile.html（ORT WASM / 6 线程）│
│  ├ CDP WebSocket 客户端     │◄─►│  ├ onInterceptRequest 注入 COOP/COEP       │
│  ├ Runtime.evaluate         │CDP│  │   → crossOriginIsolated=true（§5.3）  │
│  └ 推图 / 回读结果 JSON      │   │  ├ getUserMedia 环境摄像头（采集层）      │
└─────────────────────────────┘   │  └ __LPR.run(blob) → 同一套推理流水线      │
   hdc fport tcp:9222 →            └──────────────────────────────────────────┘
   localabstract:webview_devtools_remote_<pid>
```

**传输层 = CDP，不是自建服务器**。ArkWeb 原生支持 DevTools 远程调试：
`webview.WebviewController.setWebDebuggingAccess(true)` 让 ArkWeb 起一个 domain socket，
`hdc fport tcp:9222 localabstract:webview_devtools_remote_<pid>` 把它映射到 PC，
PC 侧用标准 Chrome DevTools Protocol（WebSocket）驱动页面。
→ 不需要在手机上跑 HTTP 服务、不需要网络、不碰公网，**与项目「零外部网络请求」的红线一致**。

**程序化入口**：`demo.js` 导出 `window.__LPR = { run, bench, ensureModels, samples, samplesBase }`，
与 UI 路径**共用同一个 `run()`**。PC 与手机壳的相机循环都走这个口，
所以「PC 推图」与「手机拍图」的推理结果**逐字节同源**，不存在两条实现漂移的风险。

**实测闭环（2026-09-17）**：

| 步骤 | 命令 / 动作 | 结果 |
|---|---|---|
| 1 | 装 HAP + 授权相机 | 截图里出现自定义 reason 弹窗 |
| 2 | `cat /proc/net/unix \| grep devtools` | `@webview_devtools_remote_29629` |
| 3 | `hdc fport tcp:9222 localabstract:webview_devtools_remote_29629` | 端口映射成功 |
| 4 | `curl 127.0.0.1:9222/json/list` | 返回演示页 target |
| 5 | `node tools/harmony/pc_drive_phone.mjs` 读 `window.__DEMO` | `ready:true, threads:6, isolated:true` |
| 6 | 推 `assets/samples/hlpr-test.jpg` | **`苏ED5172` 蓝牌 100%**，单层，det 74.1%，123×78 px |
| 7 | 回读分段耗时 | `total 222.4 ms` = 138.1 检测 / 26.1 矫正 / 58.3 识别 / 3.4 分类，`errors []` |

> ⚠️ **第 6 行的两处口径修正（ADR-015 / A16，原始读数保留不改）**：
> ① **牌色两口径未裁决** —— 「蓝牌 100%」是 **cls 模型的输出**，而按牌面漆像素量法
> （`tools/plate_face_colour.py`，G−B=+39、白字中性）该牌是**真绿牌**。引用时必须说明是哪一种口径。
> ② 更关键的是：**本行自己就记录了 A16 用来推翻真值的证据** —— 「123×78 px」即被偏航压缩
> 到比例 1.60 的牌面四边形（标准 3.14）。当时把它当成一次正常读数记下了，没有追问
> 「压缩了一半的牌面，识别结果还可靠吗」。事后看，那一格就是循环论证的破口。

> **注意**：这条通路的用途是**编排与验证**（PC 发指令、手机出结果、PC 收数据），
> 它**不会把推理搬到 PC**，也不会让手机用上 NPU——算力仍在手机 CPU 上，后端边界同 §6.1。

**下一步唯一真实的性能增益点**（按性价比排序）：

1. ~~开启 ORT WASM 多线程~~ —— **本轮完成**，见 §5.4：6 线程 ≈85 ms，1.5×。
   口径已从「固定 `numThreads = 1`」改为**自适应**（`isolated ? min(cores, forced) : 1`）。
   剩余空间：线程数目前是「全核均摊」，若改成**只用大核**（按 `cpuinfo` 亲和性分组）
   或按算子类型分档，理论上还能再拿一点——属可选优化，不再是主要增益点。
2. **y5fu 导出裸 head**（成本中等、收益结构性）——把 anchor-grid decode 移出计算图，
   让检测模型也上 NPU。这属于**模型侧改造**，不是宿主侧配置。

---

## 7. 数据质量与边界（诚实声明）

- **热状态未标注**：两轮实测的 rpv3 NPU p50 分别是 4.17 ms 与 2.97 ms（差 40%），
  CPU 侧则稳定在 8.97–9.02 ms。ADR-003 §5 已证 NPU 延迟对热状态敏感（15–20% 漂移），
  故**加速比只能给区间，不能给单点**。
- **采集轮次**：每个组合 p50 基于单轮 100 次（warm-up 5 次 + 正式 30 次），
  未做 ≥5 轮重复，CI 缺失。
- **可比性**：加速比只在**同一轮内**可比，跨轮不叠加。
- **会话常驻**：为规避析构崩溃，模型会话常驻，组合间可能存在缓存/频率状态残留。
- **`L2asFp16` 字段**：是 ADR-003 §4.1「FP16 位流被声明为 FP32」缺陷的指纹探针
  （把同一缓冲区按 fp16 重读）。本矩阵中 CPU 落点的该值为 0 属预期（未走该缺陷路径）。
- **线程扫描的边界**（§5.4）：只测了 7 个固定档位、每档 30 次、单一样本（1140×456 整车黄牌）、
  未标注 thermal level；两轮结论一致（6 线程最优、12 线程塌陷）故可信，但
  **「为什么是 6」尚未隔离验证**——异构核与热降频两个因素本 ADR 没有分开测。

---

## 8. 后续动作

- [x] ~~在鸿蒙壳里放开 `ort.env.wasm.numThreads`，用 12 核重测 `demo.html` 端到端延迟~~
      → **已完成（§5.4）**：6 线程 ≈85 ms，相对单线程 1.5×；12 线程反而 157–162 ms
- [x] ~~鸿蒙壳把演示页钉到最优线程档~~ → **已完成（§6.1）**：
      `Index.ets` 的 `DEMO_URL` 钉死 `?threads=6`，规避 12 线程自适应导致的卡顿
- [x] ~~PC → 手机远程驱动通路~~ → **已完成（§6.2）**：
      `setWebDebuggingAccess` + `hdc fport` + CDP，`tools/harmony/pc_drive_phone.mjs` 推图回读跑通
- [ ] **真机摄像头采集验证**：nova 14 Pro 对真实车牌 `getUserMedia` 出图并识别
      （目前只验证了「PC 推图」与「相机权限弹窗」，实时相机循环未在真车牌上实测）
- [ ] y5fu 导出「无 anchor-grid decode」裸 head 版本，重上 NPU 矩阵
- [ ] 端侧采集 ≥5 轮并标注 thermal level，补齐 CI
- [x] ~~网页版若要开多线程：`tools/serve.mjs` 需补 COOP/COEP 响应头（与壳内注入同源同法）~~
      → **已完成**：`serve.mjs` 已发三个头；副作用与修法见 §5.5
- [ ] （可选）把线程数从「全核均摊」改为「只用大核」或按算子分档，验证是否还能再拿一点
- [ ] （仅当确需 NPU 收益时）另起**原生 MindSpore Lite 路径**——Web 演示页物理上到不了 NPU（§6.1）
