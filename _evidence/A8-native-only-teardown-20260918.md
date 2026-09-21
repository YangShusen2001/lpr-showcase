# A8 · 拆掉 ArkWeb —— 手机端纯原生化

- **日期**：2026-09-18 11:28–11:35
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **前置**：A7（引擎异步化，已验收）；本轮是用户选定的"完全原生路线"外壳拆除
- **证据**：`_evidence/native_only_20260918.log`（全量 hilog）

---

## 1. 改了什么

### 1.1 页面重写（`Index.ets`）

旧文件 1270 行、三标签页、内嵌一个常驻 ArkWeb 组件（`Web({src: DEMO_URL})` + 自定义
`onInterceptRequest` 走 rawfile、COOP/COEP 头、相机权限转发、`runJavaScript` 抓 `__BENCH`）。

新文件 37946 字节、**两标签页、零 Web**：

| 标签页 | 内容 |
|---|---|
| 原生演示 | 加载模型 / 枚举镜头 / 跑三个样本 / 车牌 + 颜色 + 分段计时 + 镜头清单 |
| 探针控制台 | 加速器矩阵（warmup 5 + 30 次 p50 逐模型×逐后端）、GPU 专项、NPU 漂移对照、ncnn/Vulkan 自证、Vulkan 探针 + 日志 |

删掉的东西（连同其存在理由）：`webview` 导入、`Web` 组件、`webCtrl`/`webReady`/`loadedTab`、
`DEMO_URL`/`PROBE_URL`、`MT_COUNTS`/`MT_ITERS`/`MT_TIMEOUT_MS`（WASM 多线程扫描）、
`appendWeb`/`webDiag`/`waitWaitBench`/`collectWeb`/`mtSweep`/`switchTab`/`onPageEnd`/
`makeResponse`/`mimeOf`、`setWebDebuggingAccess(true)`。

旧文件移到 `C:\Users\26671\lpr-harmony\reference\Index.arkweb-era.ets`（工程外，不参与编译），
作为 Web 时代实现的参考留存。

**新页面里的线程纪律**：全部走 `loadModelAsync` / `pipelineAsync` / `benchAsync` / `ncnnLoadAsync`，
一个同步 NAPI 都没有（这是 A6 那条 SIGKILL 的直接对策）。

### 1.2 rawfile 瘦身（`tools/sync_rawfile.py`）

清单从「网页交付物 22 项」改成「原生真正读的字节 3 项」：

```diff
-MANIFEST = [index.html, demo.html, assets/css/**, assets/js/**,
-            assets/ort/ort.min.js, ...threaded.{js,wasm},
-            assets/models/*.onnx.json ×3, assets/samples/*.jpg ×6,
-            docs/adr/ADR-001..003, paper/**, resume/**]
+MANIFEST = [assets/samples/hlpr-test.jpg,
+            assets/samples/scene-2.jpg,
+            assets/samples/crop-0-津B6H920.jpg]
-EXTRA = [("tools/harmony/webprobe.html", "webprobe.html")]
+EXTRA = []
 KEEP   = ["models"]   # 不变：16 份 .ms / ncnn 资产由 lpr-harmony 侧投放
```

**必须一起改**：该脚本是「先 rmtree 再重建」的幂等实现，不改清单就等于下次同步把删掉的
网页资产又拷回来（用户在选择时点明了这一条）。

### 1.3 体积

| 项 | 前 | 后 |
|---|---|---|
| rawfile | 53 MB | **26 MB**（只剩 `models/` + 3 张样本） |
| signed HAP | 87.1 MB | **58.3 MB** |

剩余 58 MB 主要是原生库（libncnn.so 带 Vulkan、libomp、MindSpore Lite NDK）与
`models/` 里 16 份 `.ms` 变体 —— 后者是探针控制台要用的，暂不裁剪。

## 2. 验证（拆完必须重跑，不能只看编译过）

`AUTO_SELFTEST` 全链在新页面下逐项通过：

| 检查项 | 结果 |
|---|---|
| `VULKAN PROBE len=` | ✅ |
| `NATIVE LENS count=2` | ✅ |
| `NATIVE LOAD ok`（det=CPU / rec=NPU / cls=CPU） | ✅ |
| `NATIVE PIPE code=苏ED5172`（cropSum=**2773473**） | ✅ 与参考/浏览器路径一致 |
| `BACKEND E2E END`（4 模式 × 3 次） | ✅ ms-cpu / ncnn-cpu / ncnn-vulkan 全 match=1；ms-nnrt 全 match=0（苏E05172，已知漂移） |
| `E2EMATRIX BEGIN … END` | ✅ **16/16 组合** |
| `session cap` 拒绝 / watchdog 命中 | **0 / 0** |
| 进程存活 | ✅ pid 5122 |

矩阵落点与 A6/A7 完全一致（rec→NPU match=1；cls-fp32→NPU match=1；cls-fp16→CPU；
MS 的 gpu/kirin 档回落 CPU）。

**保真度未受影响**：`cropSum=2773473` 与 A5/A6/A7 三次运行逐值相同 —— 拆外壳没有碰到算法层。

## 3. 现在的形态

```
手机端（纯原生，无 WebView）
├─ ArkUI 两页：原生演示 / 探针控制台
├─ NAPI（libentry.so）
│   └─ 专用推理线程 → MindSpore Lite(cpu/nnrt) + ncnn(cpu/vulkan)
├─ rawfile：models/(16 份 .ms+ncnn) + assets/samples/(3 张)
└─ 会话登记表（按 文件|后端 去重，上限 20）

PC 端（仍在项目仓库，供浏览器演示）
├─ index.html / demo.html / mobile.html + assets/**（ORT wasm 等）
└─ paper/ docs/ resume/ tools/
```

网页版交付物**没有删除**，只是不再进 HAP —— 简历页面/毕设演示仍然靠它。

## 4. 下一步（未做）

1. **rec / cls 的 ncnn 模型**：GPU(Vulkan) 侧目前只有检测器能跑，需 onnx→ncnn 转换。
2. **正式性能矩阵**：`benchAsync` 已就绪（warmup 5 + 30 次 p50），矩阵里目前仍是
   "3 次重复 + 1 次预热"的功能验证口径；要进论文需换 30 次基准并标注 thermal。
3. **裁剪 `models/`**：16 份 `.ms` 变体里有一半是图手术实验产物，探针页面用得上，
   但若要再压 HAP 体积可以从这里下手。