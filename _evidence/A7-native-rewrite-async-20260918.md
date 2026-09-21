# A7 · 手机端端到端重写（异步化 + 会话登记表）与验证

- **日期**：2026-09-18 11:13–11:25
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **起因**：A6 定位到的架构死穴 —— 推理全在 ArkTS 主线程上同步执行，
  整轮后端矩阵连续占用 uv loop **8.887 s** → `THREAD_BLOCK_3S/6S` → **SIGKILL**（r2/r3 两次被杀）
- **证据**：`_evidence/e2e_matrix_async_r2_20260918.log`（本轮全量日志）、
  `_evidence/e2e_matrix_async_20260918.log`（上一轮，撞会话上限）、
  `_evidence/A6-three-model-backend-matrix-20260918.md`（问题侧）

---

## 1. 改了什么（承重墙）

### 1.1 原生侧：一根专用推理线程 + async NAPI（`napi_init.cpp`）

| 项 | 旧 | 新 |
|---|---|---|
| 执行线程 | 每次 NAPI 调用**同步跑在 JS 线程** | `InferenceRunner` **专用线程**，任务串行执行 |
| ArkTS 接口 | 只有同步版 | 新增 `loadModelAsync` / `pipelineAsync` / `benchAsync` / `ncnnLoadAsync`（Promise） |
| JS 线程占用 | 整轮矩阵 8.9 s → 被 watchdog 杀 | 每次调用立即返回 Promise，UI 线程全程可响应 |

为什么用**专用单线程**而不是 libuv 线程池直接跑：libuv 池有 4 根线程轮流调度，
同一份 NNRT/MindSpore 会话会在不同线程上被使用，而**线程亲和性没有任何文档保证**。
同步接口（`loadModel`/`pipeline`/`bench`/`ncnn*`）保留但内部同样提交给这根线程，
保证"所有 MS/ncnn 调用都在同一根线程上"这条不变量不因调用方而破。

### 1.2 原生侧：会话登记表（去重 + 封顶）

旧的 `static std::vector<MsSession*> g_sessions` 是**只增不减的裸 vector**：
每次 `loadModel` 都 push_back，而 MsSession 因 NNRT 析构 crash 而**永不释放**
（见 `ms_engine.h` 的生命周期说明），跑一轮矩阵就常驻十几份会话。

新设计：
- **键 = `(模型文件, 后端)`**，命中直接复用并回 `cached=1`；id = 下标，永不回收。
- **上限 `kMaxSessions = 20`**：到顶诚实失败（`ok=0;error=session cap 20 reached`），不继续漏。
- ArkTS 侧一律用**模型文件路径**当会话名（不再用 `det`/`rec`/`cls` 标签），
  否则同一份模型会被不同标签重复建会话 —— 这正是第 13 个组合报
  `session cap 12 reached` 的原因（已在同一次修复中改掉）。

### 1.3 结果落盘（被杀也能续）

新增 `appendLine(path, line)`；矩阵每跑完一个组合就写一行到
`<filesDir>/matrix.log`。旧实现全靠 hilog 环形缓冲，r2 被杀后只"碰巧"留下 28 行。

> 边界：该文件在应用沙箱内（`/data/storage/el2/base/haps/entry/files/`），
> `hdc file recv` 会被 SELinux 拒绝（实测 `permission denied`）。
> 用途是**应用自己续跑/汇总**，不是给 PC 侧直读。

## 2. 验证结果（全量 16 组合）

`E2EMATRIX BEGIN ... pairs=16` → `E2EMATRIX END`，
`session cap` 拒绝 **0** 条，watchdog 命中 **0** 条，进程存活（pid 60923）。

| 模型 | 请求后端 | **LANDED** | match | totalMs (3 次) |
|---|---|---|---|---|
| det | cpu | `CPU` | 1/1/1 | 146 / 184 / 145 |
| det | nnrt | `NNRT:NPU_…kirin8020_v2_0` | **0/0/0**（苏E05172） | 96 / 91 / 93 |
| det | gpu | `CPU` | 1/1/1 | 145 / 124 / 116 |
| det | kirin | `CPU` | 1/1/1 | 147 / 140 / 142 |
| **rec** | cpu | `CPU` | 1/1/1 | 183 / 168 / 162 |
| **rec** | **nnrt** | **`NNRT:NPU_…kirin8020_v2_0`** | **1/1/1** | 145 / 147 / 146 |
| **rec** | gpu | `CPU` | 1/1/1 | 193 / 161 / 148 |
| **rec** | kirin | `CPU` | 1/1/1 | 193 / 162 / 153 |
| cls（fp16） | cpu | `CPU` | 1/1/1 | 155 / 140 / 164 |
| cls（fp16） | nnrt | `CPU`（NPU 构不出 kernel） | 1/1/1 | 118 / 145 / 160 |
| cls（fp16） | gpu | `CPU` | 1/1/1 | 114 / 141 / 130 |
| cls（fp16） | kirin | `CPU` | 1/1/1 | 144 / 118 / 151 |
| cls-fp32 | cpu | `CPU` | 1/1/1 | 139 / 141 / 143 |
| **cls-fp32** | **nnrt** | **`NNRT:NPU_…kirin8020_v2_0`** | **1/1/1** | 145 / 156 / 132 |
| cls-fp32 | gpu | `CPU` | 1/1/1 | 164 / 169 / 147 |
| cls-fp32 | kirin | `CPU` | 1/1/1 | 169 / 146 / 123 |

**与 A6 逐项一致**（同一份模型的落点与 match 完全复现），说明改造没有改变语义，
只是把执行位置从 UI 线程搬到了专用线程。

### 对照：三次运行的存活情况

| 轮次 | 执行方式 | 组合数 | 结果 |
|---|---|---|---|
| r1（A6） | 同步 NAPI | 12 | 跑完，**侥幸**未被杀 |
| r2/r3（A6） | 同步 NAPI | 16 / 8 | 跑完即被 **SIGKILL** |
| **本轮** | **async + 专用线程** | **16** | **跑完，存活，0 watchdog** |

## 3. 仍未做（下一阶段：完全原生路线）

本轮改的是**引擎与驱动逻辑**（端到端能跑、不被杀、可续跑）。
用户选定的"拆掉 ArkWeb，手机端只留原生界面"**尚未动**：
`Index.ets` 里仍有 Web 标签页、`webprobe.html`、`mobile.html?embed=1` 的 ArkWeb 组件，
rawfile 里仍打包着 `assets/`（ORT wasm 13 MB 等）与 `index/demo/mobile.html`（HAP ≈ 87 MB）。

拆除清单（下一步执行）：
1. `Index.ets`：删 `webview` 导入、Web 组件、Web 标签页、`DEMO_URL`/`PROBE_URL`、
   `appendWeb`/`webDiag`/`waitBench`/`webProbe`/`probeAll`（webprobe 部分）。
2. rawfile：只保留 `models/` 与 `assets/samples/`（本地推理需要），
   其余（`assets/ort`、页面 html、paper/docs/resume）移出，预计 HAP 从 ~87 MB 降到 ~30 MB。
3. `tools/sync_rawfile.py`：同步更新白名单，否则下次同步会把删掉的资产又拷回来。
4. 验收：`AUTO_SELFTEST` 全链仍跑完、三后端仍能调用、无 watchdog。

## 4. 工程约定（本轮新增，务必遵守）

- **ArkTS 侧一律用 `*Async`**。同步版只是为了兼容旧调用点，它仍会阻塞 UI 线程。
- **会话名必须传模型文件路径**，不要传角色标签 —— 否则去重失效、撞上限。
- **NNRT/MindSpore 调用只允许出现在原生推理线程上**；新增原生入口时，
  若涉及 MS/ncnn，必须走 `RunJob`/`RunJobSync`，不要直接调用引擎函数。
- 会话上限 20 是**硬边界**：新增模型文件 × 后端组合时要同步上调，否则会得到
  `session cap` 而不是静默错误。