# ADR-012 · CANN / HiAI / NNRt 直连路线的评估与「不换」决策

- **状态**：已接受（2026-09-18）
- **前置**：ADR-003（端侧部署路径）、ADR-004（三后端矩阵与 NPU 拒绝边界）、ADR-008（Vulkan 终审）、ADR-011（ncnn 接入）
- **后续**：**ADR-013 已把本条的前提改写** —— 厂商产出的 `.om` 已端到端跑通（编译/IO/推理/计时全绿），
  缺口从"通路未知"收窄为"缺 OMG 转换器"。§5「未完成动作」第 1 条因此更具体了，但仍未完成。
- **证据**：`_evidence/A10-cann-nnrt-probe-20260918.md`、`_evidence/cann_probe_20260918.log`、`docs/research/2026-09-18-backend-alternatives-and-cann.md`

## 问题

「NPU 与 GPU 还有没有别的办法？CANN 框架呢？」——即：现有 `MindSpore Lite + NNRT delegate`
是不是**绕远路**，能不能改走更直接的厂商通路（CANN Kit / HiAI Foundation / NNRt 直连），
从而拿到更好的性能或更硬的落点证据？

## 决策

**维持现状（`MindSpore Lite + NNRT delegate`）为生产路径；不为性能改走 CANN/NNRt 直连。**

理由有三，按重要性排序：

1. **现有路径就是官方底座，不是绕路。** A10 探针从 NNRt 枚举出的设备名
   `NPU_ohos.boot.hardware.kirin8020_v2_0`，与 MS Lite 落到 NPU 时打印的
   `model built on NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0` **逐字相同**。
   两个独立来源互证：MS Lite 的 NNRT delegate 走的就是 NNRt，NNRt 在麒麟上落到 HiAI。
   → 直连 NNRt 能省掉的最多是一层胶水，而 rec 段已经 7.5 ms，**性能上限很小**。
2. **CANN/NNRt 的离线入口不吃我们手上的模型格式。** 三份 `.ms` 全部被拒：
   `hiai_compat_code=1`（HiAI 判为不兼容）+ `build_rc=1`（`OH_NN_FAILED`），
   与官方文档「offline model 由设备厂商提供的转换器生成」三方一致。
   门槛是 **OMG 转换器**（需华为开发者账号 + 64 位 Linux + 与设备 CANN 版本
   `108.631.120.010` 对齐），**不是权限**（syscap 双 true、库在 `/system/lib64/ndk/` 可 `dlopen`）。
3. **GPU 侧没有新路可走。** NNRt 枚举出的 2 张设备**全是 `ACCELERATOR`，没有 GPU**；
   MS Lite 的 GPU 档（OpenCL）编译期判否（ADR-004 §4.2）；三方框架（MNN/TNN/Paddle-Lite）
   的华为 NPU 后端官方定位是 Android。→ GPU 唯一通路仍是 ncnn-Vulkan，已做完（A9），
   且结论是「通路成立、数值保真、**小图不加速**」。

## 证据摘要

| 项 | 实测值 |
|---|---|
| NNRt 库是否可用 | ✅ `libneural_network_runtime.so`(1.52 MB) / `libneural_network_core.so`(96 KB) 按 **soname** 可 `dlopen` |
| CANN Kit 库是否可用 | ✅ `libhiai_foundation.so`(195 KB) 可 `dlopen` |
| syscap | ✅ `SystemCapability.AI.HiAIFoundation=true`、`SystemCapability.AI.NeuralNetworkRuntime=true` |
| CANN 版本 | `108.631.120.010` |
| NNRt 设备 | 2 张，**均为 ACCELERATOR**：`HIAI_F`、`NPU_ohos.boot.hardware.kirin8020_v2_0` |
| `.ms` 喂离线入口 | 3/3 被拒（`compat=1`、`construct=ok`、`set_device_rc=0`、`build_rc=1`） |
| 对生产链影响 | 零：`cropSum=2773473`（第五次连续同值）、16/16 矩阵、ncnn 槽位全绿、0 watchdog |

## 边界声明（不要把结论读过头）

- `build_rc=1` 只证明**这份字节**被拒，**没有**证明 `.om` 可行——手上没有 `.om` 可测。
- 只试了 `deviceIndex=0`（`HIAI_F`）；`d1`（麒麟 NPU 实名）未单独试。
- 未评估 CANN Kit 的**单算子通路**（`hiai_single_op.h`）——那是另一条可能绕开 `.om` 的路线。
- 与 ADR-004 §4.2 的「`kirin` 档缺 `libhiai_ir_infershape.so`」**不矛盾**：那条说的是
  MindSpore Lite **内部**私有档位，本 ADR 说的是 NDK 对外库。文档里必须分开写。

## 未完成动作

1. 若要拿"显式设备绑定"这一档证据：取 DDK/OMG（WSL2 + 开发者账号）→ 转一个 `.om` →
   用**已写好**的 `HMS_HiAICompatibility_CheckFromBuffer` 验兼容性 → 再用
   `HMS_HiAIOptions_SetModelDeviceOrder` / `OH_NNCompilation_SetDevice` 做硬绑定。
   探针已就位，改一行即可测。
2. 若要彻底摆脱转换器：写 ONNX→NNRt 的**在线构图**映射器（`OH_NNModel_AddTensor/AddOperation`）。
   CANN Kit 官方能力页明确支持 ONNX 在线构图；代价是自研算子映射。
   规模（实测）：ONNX 节点 **det 416 / rec 444 / cls 103**；转成 ncnn 后对应 **218 / 252 / 78** 层。
   工作量「几天」量级，且性能上限已知不高 —— **不建议在毕设期内做**。
3. 论文口径：把这轮当作「厂商通路可用性的实测边界」，与 ADR-004/007 同属"边界类"证据，
   标题中**不得**升级为「用了 CANN 加速」。
