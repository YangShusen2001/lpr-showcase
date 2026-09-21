# A5 · 三后端（CPU / GPU / NPU）跑通复现验证

- **日期**：2026-09-18 10:32（本地）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1 · 序列号 4CY9K25614046328
- **产物**：`entry-default-signed.hap` 86,995,330 B（2026-09-17 22:00 构建，本次源码未改动，hvigor 增量 `UP-TO-DATE`）
- **工具链**：hdc **3.2.0d**（DevEco `sdk/default/openharmony/toolchains/hdc.exe`）
- **采集脚本**：`_evidence/e2e_backends_20260918.log`（全量 4.08 M 字符）+ `.summary.txt`（关键行 107 条）
- **上游**：A4-backend-e2e-result.txt（2026-09-17 22:01，PID 11692）

---

## 1. 验证方式

装机后启动 App，`AUTO_SELFTEST=true` 的无人值守链自动执行：

```
vulkanProbe（应用级 Vulkan 能力）
  → listLenses（镜头枚举）
  → loadNative（三模型独立后端加载：det / rec / cls）
  → runNative（整图样本端到端一次）
  → backendProbe（**被测对象**：同一张解码图 × 4 种检测后端 × 3 次重复）
```

`backendProbe` 固定「识别 = NNRT NPU、分类 = CPU」，**只切换检测器后端**，
判定字段 `match=` 由原生代码逐字符对比期望值 `苏ED5172`，不依赖日志编码。

## 2. 结果矩阵（3 次重复全一致）

| mode | 检测后端 | 结果 | match | totalMs (rep0/1/2) | 加载自证 |
|---|---|---|---|---|---|
| `ms-cpu` | **MindSpore Lite CPU** | 苏ED5172 ×3 | **1 / 1 / 1** | 93.6 / 106.7 / 114.0 | `model built on CPU` |
| `ms-nnrt` | **NNRT NPU** | **苏E05172 ×3** | **0 / 0 / 0** | 102.0 / 94.6 / 94.0 | `model built on NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0` |
| `ncnn-cpu` | **ncnn CPU** | 苏ED5172 ×3 | **1 / 1 / 1** | 110.7 / 93.7 / 96.5 | `layers=218;requestedVulkan=0;effectiveVulkan=0` |
| `ncnn-vulkan` | **ncnn Vulkan → Maleoon 920C** | 苏ED5172 ×3 | **1 / 1 / 1** | 164.9 / 131.2 / 123.6 | `requestedVulkan=1;effectiveVulkan=1;gpuProbe=ok=1;count=1;name=Maleoon 920C;fp16=111` |

### 结论

> **CPU ✅ 能调用且结果正确 · GPU ✅ 能调用且结果正确 · NPU ✅ 能调用（结果有已知数值漂移）**

1. **三个后端都能被第三方 App 正确调用并跑通完整流水线**，无一崩溃、无一 `SKIP`、无一 `EXCEPTION`。
2. **NPU 的"跑通"需要限定词**：会话在 NPU 上成功构建并有 `dllite_service/DLSA [AddModel] add hiaiID:… nnrtID:…` 实锤，
   但输出 `苏E05172`（D→0）—— 与 ADR-009 的 0.08% 漂移推断、A4 的端到端翻字符**完全同源**。
   → **"能调用" ≠ "结果可用"**：生产链仍为 `det=CPU / rec=NPU / cls=CPU`。
3. **与 A4 逐项一致**（4 模式 × match 值 × 字符级输出全同），说明该现象**可复现**，不是偶发。
4. 识别器（rec）本次仍落 **NPU 成功**（`id=1;backend=NNRT:NPU_…;inputShape=[1,48,160,3]`），分类（cls）落 CPU 成功 ——
   与生产链配置一致，**但本次矩阵未对 rec / cls 做后端遍历**（见 §4）。
5. GPU 侧附加证据：`VULKAN PROBE` 本次重新输出 `libvulkan.so` dlopen OK、deviceExtensions **97** 个、
   `graphicsComputeQueues=2` —— 与 ADR-008 相同。

## 3. 本次新增的两个环境坑

| 坑 | 现象 | 解法 |
|---|---|---|
| **老 hdc 被拒** | `D:\Tools\Huawei\…\hmscore\3.1.0\toolchains\hdc.exe` 对 `install`/`bm dump` 报 `[E000001] The sdk hdc.exe version is too low` | 改用 DevEco 的 **3.2.0d**，且**先 `hdc kill` 停掉残留 server** 再用它连设备 |
| **install 路径被拼接** | 传 `C:/Users/.../xxx.hap` 时 hdc 把 cwd 拼在前面 → `C:\…\lpr-harmony\C:/Users/…\xxx.hap` → `no such file` | **先 `cd` 到 hap 所在目录，传相对文件名** |

## 4. 边界（不得过度解读）

- 本次只做**检测器**的后端遍历；**rec / cls 的三后端矩阵未做**（rec 固定 NPU、cls 固定 CPU）。
  若要"整条流水线每级都可选三后端"，需要扩展 `Index.ets` 的 `backendProbe`。
- 每模式仅 **3 次重复**，`totalMs` 为功能验证采样，**不是 30 次 p50 性能基准**；
  且四模式共享同一矫正/识别/分类段，差异全部来自检测输出。
- `ncnn-vulkan` 的 `effectiveVulkan=1` + GPU 枚举证明「**以 Vulkan 配置执行**」，
  仍**不是逐层 GPU 时间戳级执行证据**；Vulkan 首次 164.9 ms → 第三次 123.6 ms 的下降趋势
  与 A4 的 rep0→rep2 现象一致（疑 shader 编译/内存分配开销），无插桩 30 次基准未做。
- 未标注 thermal level；跨栈比较（MS Lite ↔ ncnn）只作量级参考。

## 5. 证据链

- 全量日志：`_evidence/e2e_backends_20260918.log`（4.08 M 字符，含 hilog 原始前缀）
- 关键行摘录：`_evidence/e2e_backends_20260918.summary.txt`（`BACKEND E2E` / `NCNN` / `VULKAN PROBE` / `LENS` 等 107 条）
- 上游 ADR：ADR-004（后端矩阵与 NPU 拒绝边界）、ADR-008（Vulkan 可达）、ADR-009（NPU 漂移）、ADR-011（ncnn 移植）
