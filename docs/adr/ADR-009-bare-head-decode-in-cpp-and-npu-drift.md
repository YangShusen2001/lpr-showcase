# ADR-009：检测裸 head 的 C++ decode 落地，与 NPU 数值漂移的量化取舍

- **状态**：已接受（真机实测，两轮对照）
- **日期**：2026-09-17
- **前置**：ADR-006（decode 公式 + 裸 head 上 NPU）、ADR-008（GPU 终审）
- **后续**：ADR-015 —— 本 ADR 的「NPU 0.08% 漂移 → 翻字符」是 **fp16 NPU** 下的观测
  （`ms_engine.cpp` 的 `addNnrt()` 恒传 `fp16=true`，从未试过 fp32）。
  `det=CPU` 的生产分配**暂时维持**，但表述须限定为「det 不能在 NPU-fp16 上用」；
  文中 `苏ED5172`/`苏E05172` 均为 rpv3 参照串及其差分结果，非人工真值。
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135
- **证据**：`_evidence/det_npu_vs_cpu_drift_20260917.{json,txt}`、
  `_evidence/verify_head_decode_match_20260917.txt`、
  `_evidence/bare_head_pipeline_check.json`、`_evidence/det_head_ref_checksum.json`
- **工具**：`tools/verify_head_decode.py`、`tools/verify_bare_head_pipeline.py`、
  `tools/det_head_ref_checksum.py`

---

## 1. 做了什么

ADR-006 只证明「裸 head 能上 NPU」，流水线里跑的仍是原模型（含 decode，落 CPU）。
本 ADR 把 decode 真正搬进 C++，让流水线可以消费裸 head：

- `ms_engine.{h,cpp}`：新增 `MsRunMulti()` —— 原 `MsRun` 只取 `outs.handle_list[0]`，
  裸 head 有 **3 个输出**，必须全取；每个输出按声明 dtype 转 fp32。
- `lpr_pipeline.{h,cpp}`：新增 `LprDecodeBareHead()` —— 按 ADR-006 §5 的公式把
  三个 `[1,45,H,H]` 变成与原模型同一份 `[6300,15]`；行序 (scale, anchor, y, x)
  与图内 Reshape 一致，因此结果可直接喂给既有 `LprDecodeDetections`（NMS 不动）。
- `LprRunPipeline`：3 输出走裸 head decode，单输出走原路径 —— 向后兼容。
- App：det 换成 `models/y5fu_320x_head_fp32.ms`。

## 2. 正确性：先在 PC 上把「移植」和「NPU」分开

排查真机异常时最容易犯的错，是把「C++ 写错」和「NPU 有噪声」混为一谈。
分层验证（这是本 ADR 的方法论要点）：

| 层 | 工具 | 结果 |
|---|---|---|
| decode 公式 | `verify_head_decode.py`（裸 head vs 原模型输出） | maxAbsDiff **0.000061** → **MATCH** |
| 全链路 | `verify_bare_head_pipeline.py`（裸 head+decode 走完整后处理） | `苏ED5172`、marks 逐值、det_score 0.7413、crop 78×123 —— **与基线全同** |

→ **C++ 移植无罪**：真机上任何异常都不该归因于 decode。

## 3. 真机：NPU 能跑，但数值不够喂给矫正

`det` 请求 `nnrt` → **确实落上 NPU**（`NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0`），
5–6.6 ms，但车牌被认成 `苏E05172`（参照 `苏ED5172`），`cropSum` 也变了。

### 3.1 「是不是其实没调用 NPU？」——同进程双会话对照

静默回落 CPU 会走**同一条代码路径**，输出必然逐位相同。于是在**同一进程**里
加载两份相同的 `y5fu_320x_head_fp32.ms`（backend `nnrt` / `cpu`），跑同一确定性输入：

| 后端 | landed | checksum(L2) | maxAbs | p50 |
|---|---|---|---|---|
| nnrt | `NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0` | **1043.4696** | **18.0000** | **5.04 ms** |
| cpu | CPU | **1042.6634** | **18.0232** | **8.44 ms** |

三条独立证据说明 NPU 真的在算：
1. **输出不同**（L2 差 0.08%、maxAbs 差 0.0232）——回落则必逐位相同；
2. **延迟差 1.67×**——换硬件才会有；
3. **NPU 编译器/运行时日志实锤**（`hiaiserver` + 本进程）：
   - `HIAI_DDK_MSG: model_manager_impl.cpp(525) "DDK pipe is HCL."`
   - `hiaiserver/AI_FMK: process_manager.cpp(29) "add the record of pid : 16633"`
   - `hiaiserver/AI_NPUCL: npu_graph_kernel_info_manager.cc(35) "call rtProcessCreate for pid:16633"`
   - `NPUCL: DevmmManagerInit`（NPU 设备内存初始化）、`AI_NPUCL: NpuclEventListener register`
   - DDK 版本 `108.631.120.010`
   （`nnrt_delegate.cc:638 hiai_foundation is nullptr` 只是一条警告——HCL 管道照常起来。）

顺带确认：**det 裸 head 在 NPU 上是全图执行**（无 Reshape/Transpose，符合 ADR-007
判据）；而 **rec（rpv3）是混合执行**——日志里 `/neck/encoder/svtr_block.{0,1}/mixer/
Reshape`、`Transpose` 判 `not supported in npucl store`，这些子图回落 CPU
（与 ADR-005 §6 一致）。所以 rec 的 2.2–3.0× 是「部分 NPU」的收益。

用 CPU-diff 协议量化（确定性输入，输出 0 = `[1,45,40,40]`）：

| 落点 | checksum(L2) | maxAbs | 车牌结果（与 rpv3 基线比对） |
|---|---|---|---|
| NPU | 1043.4696 | 18.0000 | `苏E05172` 与基线不一致 |
| **设备 CPU** | **1042.6634** | **18.0232** | `苏ED5172` ＝基线 |
| PC ONNX（NCHW 填） | 1144.4541 | 21.77408 | — |

**两个必须说清的读法**：
1. **PC 与设备的 8.8% 差不是精度问题**——PC 按 NCHW 填、设备按 NHWC 填，
   同一串数字的**空间排列不同**，输出 L2 天然不同。跨端 L2 对比必须同布局。
2. **真正的 NPU 漂移是 NPU vs 设备 CPU**：L2 差 **0.08%**、maxAbs 差 **0.0232**。
   看着小，但关键点 `kpt = logit × anchor + grid_px`，anchor 最大 433
   → 0.023 的 logit 差放大后约 **10 px** 偏移。而下游单应矫正对 1 px 都敏感
   （ADR-002 已证明 1 px 能翻一个字符）。

## 4. 决策

| 模型 | 后端 | 依据 |
|---|---|---|
| det（裸 head） | **CPU** | 输出被下游逐像素消费，0.08% 漂移即翻字符 |
| rec（rpv3） | **NPU** | argmax + CTC 对数值噪声鲁棒，2.2–3.0× 收益照拿 |
| cls | CPU | NPU 反而更慢（ADR-004） |

「裸 head 能上 NPU」仍是成立的结论（ADR-006/008 的能力证明保留），
只是**这条流水线的检测环节不适合吃 NPU 的输出**——能力边界与应用边界是两件事。

**最终状态**：`det=CPU`（裸 head + C++ decode）、`rec=NPU`、`cls=CPU`，
端到端 102.6 ms，`苏ED5172` **与 CPU 基线逐字符一致**（注意：该串是 rpv3 自己的输出、
非人工真值，见 ADR-015），`cropSum=2773473` 与基线一致。

## 5. 待办（下一轮）

- [ ] det 阶段 68.9 ms 仍是最大单项（推理只占 ~10 ms）→ **分段计时**拆 letterbox/打包/推理/decode，
      验证 ADR-006 §7 的「预处理是主要成本」假设。
- [ ] 端侧 ≥5 轮稳定测量 + thermal 标注（本轮 CPU p50 14.3 ms、min 9.9 ms、p95 40 ms，抖动大，
      单次测量不足以下结论）。
- [ ] 若想让 det 也吃 NPU：可试 NNRT 的精度模式（fp32 不降精度），或改用「NPU 出粗框 + CPU 精修关键点」。

## 6. 工程坑

- **Vec 广播维度**：kpt 通道是 5 维张量，anchor 是 4 维，写作 `kpt * aw` 会
  `ValueError: operands could not be broadcast`；C++ 侧用 `base[ch * chStride]` 取通道规避。
- **rawfile 模型只读**：cp 进去带只读位会让 restool 删不掉旧文件
  （`11204003 Failed to delete`）——复制后 `attrib -r`（icacls 的 `:W` 语法在本机会报 87）。
