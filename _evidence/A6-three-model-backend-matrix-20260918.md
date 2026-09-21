# A6 · 三模型 × 四后端完整矩阵 + 主线程阻塞缺陷

- **日期**：2026-09-18 10:45（r1，det/rec/cls）/ 10:59（r3，cls 变体）
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **被测**：`e2eMatrix()`（本次新增，`Index.ets`）—— 逐模型替换后端，其余保持生产配置，每次跑完整流水线
- **证据**：`_evidence/e2e_matrix_20260918.{log,summary.txt}`（r1，完整 36 行）、`_evidence/e2e_matrix_r3_cls_20260918.log`（r3）、`_evidence/e2e_matrix_r2_20260918.log`（失败轮）
- **上游**：A4（四后端端到端对照）、A5（三后端跑通复现）

---

## 1. 为什么补这一轮

`backendProbe`（A4/A5 用的）**只遍历检测器**，识别器与分类器一直钉死在生产配置，
所以「三个后端都能被正确调用」这句话在 rec / cls 上**没有证据**。
本矩阵对**每个模型 × 每个后端**都真跑一遍完整流水线。判据沿用 A4：`LANDED=`（实际落点）+ `match=`（逐字符比对 苏ED5172）。

## 2. 结果矩阵（r1 完整，3 次重复全一致）

| 模型 | 请求后端 | **实际落点 LANDED** | match | 回落轨迹 trail |
|---|---|---|---|---|
| det | cpu | `CPU` | **1/1/1** | `CPU(ok)` |
| det | nnrt | `NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0` | **0/0/0**（苏E05172） | `NNRT(ok)` |
| det | gpu | `CPU` | 1/1/1 | `GPU:fp16(fail) GPU:fp32(fail) CPU(ok)` |
| det | kirin | `CPU` | 1/1/1 | `KIRIN_NPU(fail) CPU(ok)` |
| **rec** | cpu | `CPU` | 1/1/1 | `CPU(ok)` |
| **rec** | **nnrt** | **`NNRT:NPU_…kirin8020_v2_0`** | **1/1/1** ✅ | `NNRT(ok)` |
| **rec** | gpu | `CPU` | 1/1/1 | `GPU:fp16(fail) GPU:fp32(fail) CPU(ok)` |
| **rec** | kirin | `CPU` | 1/1/1 | `KIRIN_NPU(fail) CPU(ok)` |
| cls（fp16 变体） | cpu | `CPU` | 1/1/1 | `CPU(ok)` |
| cls（fp16 变体） | nnrt | `CPU` | 1/1/1 | `NNRT:NPU(…)kirin8020(fail) NNRT:HIAI_F(fail) CPU(ok)` |
| cls（fp16 变体） | gpu | `CPU` | 1/1/1 | `GPU:fp16(fail) GPU:fp32(fail) CPU(ok)` |
| cls（fp16 变体） | kirin | `CPU` | 1/1/1 | `KIRIN_NPU(fail) CPU(ok)` |

### r3 补充：cls 的两份模型文件，NPU 命运完全不同

| 角色 | 请求 nnrt | LANDED | match | 备注 |
|---|---|---|---|---|
| `litemodel_cls_96x_r1.ms`（**fp16**） | nnrt | `CPU` | 1/1/1 | **NPU 构不出图**：`nnrt_delegate.cc:151 BuildKirinNPUModel# Create full model kernel failed` |
| `litemodel_cls_96x_r1_fp32.ms`（**fp32**） | nnrt | **`NNRT:NPU_…kirin8020_v2_0`** | **1/1/1** ✅ | 上 NPU 成功且结果正确 |

→ **分类器能上 NPU，但只有 fp32 那一份能**。生产链里配置的是 fp16 变体，所以它实际一直在 CPU 上跑。
→ 这与 ADR-004 记的「cls 上 NPU 反而更慢（1.09 vs 0.57 ms）」并不矛盾：那次用的是能上 NPU 的变体。
**是"哪一份模型文件"决定了能不能上 NPU，不是"这个模型能不能"。**

## 3. MS Lite 两条加速档为什么必死（本轮拿到编译期原文）

| 档 | 失败原文 | 含义 |
|---|---|---|
| `gpu`（fp16/fp32） | `inner_context.cc:207 IsValid# GPU is not supported.` + `ModelBuild status=-2` | MS Lite 的 GPU 是 OpenCL 路线，本机不支持 |
| `kirin` | `inner_context.cc:213 IsValid# NPU is not supported.` + `ModelBuild status=-2` | `OH_AI_DEVICETYPE_KIRIN_NPU` 机型不支持 |
| NNRT（FP16 开关） | `context_c.cc:283 OH_AI_DeviceInfoSetEnableFP16# Unsupported Feature.` | 该属性对非 NNRT 设备无意义 |

→ **MS Lite 侧只有 CPU 与 NNRT 两档可用**；GPU 唯一通路仍是 ncnn-Vulkan（ADR-008/A4）。

## 4. ⚠️ 架构缺陷（本轮最重要的发现）：整张矩阵跑在主线程上

r2 与 r3 两次运行在矩阵跑完之后被系统杀进程，原因一致且可复现：

```
10:59:17.788  49046  LprNativeUI: E2EMATRIX END
10:59:17.788  49046  XCollie: BlockMonitor event name: uvLoopTask, Duration Time: 8887 ms
10:59:13.473  49046  AppDfr: THREAD_BLOCK_3S
10:59:16.526  49046  AppDfr: THREAD_BLOCK_6S
10:59:20.097  1599   AppMS: com.shusen.lprdemo pid 49046 will exit because THREAD_BLOCK_6S
10:59:20.524  576    APPSPAWN: com.shusen.lprdemo with pid 49046 exit with signal:9
```

**结论**：每次 `lpr.loadModel` / `lpr.pipeline` 都是**同步 NAPI 调用**，整轮 16 组合连续占用 ArkTS
uv loop **8.887 秒**，触发系统 watchdog 的 3 s/6 s 阈值 → **SIGKILL**。

- r1（12 组合、无 warmup）**活下来并跑完**；r2（16 组合 + warmup）**跑到第 10 个组合被杀**；
  r3（2 角色 8 组合、有 warmup）**跑完 END 后立刻被杀**。
- 也就是说：**矩阵的存活与否只取决于主线程被占用多久，与具体后端无关。**
- 这是"能不能做"的硬边界：**不把推理搬离主线程，任何较大的组合扫描都必然被杀。**

## 5. 修复前的最终能力矩阵（诚实版）

| 模型 | MS Lite CPU | MS Lite NPU | MS Lite GPU | ncnn CPU | ncnn Vulkan |
|---|---|---|---|---|---|
| det（裸 head） | ✅ | ⚠️ 落 NPU 但**翻字符** | ❌ 编译期不支持 | ✅（218 层） | ✅ match=1 |
| rec（rpv3） | ✅ | ✅ **match=1 且更快** | ❌ | **无模型** | **无模型** |
| cls（fp16） | ✅ | ❌ 构不出 kernel | ❌ | 无模型 | 无模型 |
| cls（fp32） | ✅ | ✅ match=1 | ❌ | 无模型 | 无模型 |

**剩余缺口（补后端的下一步）**：
1. **rec / cls 没有 ncnn 模型** → GPU（Vulkan）侧目前只有检测器能跑。要让三模型都吃 GPU，
   需要 onnx → ncnn 转换（`pnnx` / `onnx2ncnn`）并逐层验证。
2. 矩阵必须搬到 worker 线程（见下），否则无法安全跑全量。

## 6. 重写方向（已由本轮实测钉死）

| # | 问题 | 证据 | 重写要求 |
|---|---|---|---|
| 1 | **推理全在主线程** | `uvLoopTask 8887 ms` → SIGKILL | 推理必须跑在 worker / taskpool 线程；UI 线程只收结果 |
| 2 | 会话**只建不毁** | `MsSession` 注释里的 NNRT 析构 crash 规避 | 需要受控的会话池 + 上限；NNRT 会话不允许无限累积 |
| 3 | 矩阵**不可中断、不可续跑** | r2 死掉后前 28 行只是"碰巧"留在 hilog 里 | 每组合落一次盘（写文件），被杀也能续 |
| 4 | **fp16 变体的 NPU 可用性靠猜** | cls fp16 构不出 kernel，fp32 可以 | 模型选型按「能否在目标后端建图」自动判定，不写死 |

**验收标准（重写不得放松）**：同一张样本图，`code` 与 Python 参考逐字符一致；
`cropSum` 与浏览器路径一致（当前 2773473）；任一后端行都必须带 `LANDED=`。
