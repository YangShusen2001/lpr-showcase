# A13 · 「按 CANN 偏好挑模型」：判据可算，且新模型真机跑通

- **日期**：2026-09-18 15:00–15:25
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **动机**：用户提出 ——「CANN 支持最好的模型，然后再去搜相对应的模型来用」。
  即：不硬啃当前模型，而是**按 CANN 的偏好挑一个模型**。
- **结论**：
  1. **方向成立，判据可算**（官方给了公式，不是玄学）；
  2. 按判据挑出的 LPRNet 已**真机跑通 CANN**（`build_rc=0 / run_rc=0`）；
  3. **但「能跑」≠「全落 NPU」** —— 实测分区是 `NPU:2, CPU:1`，拦路的是 `ReduceMean`；
  4. **并推翻 ADR-007 §4 的一个假设**：LPRNet 的 Transpose 改造（`lprnet_npufix`）
     对 CANN 分区**毫无影响**（改前改后同为 `NPU:2, CPU:1`）。
- **证据**：`_evidence/cann_lprnet_npufix_20260918.log`（本轮干净单次抓取，16,284 行）、
  `_evidence/_opsprof.json`（算子画像）、`_evidence/cann_lprnet_clean_20260918.log`
- **官方依据**：`ncnn_conv/模型准备.md` §「NPU IR算子性能指导」+ §「NPU性能友好计算结构」

---

## 1. 官方判据（可直接算，不用猜）

`模型准备.md` 第 59 行，`Convolution` 条：

> 当 **Cin 和 Cout 都是 16 的倍数**时性能最优，达到 NPU 最大规格算力。
> 当 **Cin 和 Cout 都小于 16** 时，硬件算力按照：**`(Cin×Cout)/256 × max computer power`** 折算。

推论：**一个模型的 NPU 算力利用率上限，可以从权重形状直接算出来** ——
数一数有多少个卷积同时满足 `Cin%16==0 && Cout%16==0`，占比即上限。

官方「友好结构」表的旁证（同文件第 199–201 行）：

> 这里的评价推荐标准是**基于 NPU 内部硬件利用率**得出的，但相对于其他计算架构（CPU、GPU）
> 可能在同样不推荐的网络在 GPU 和 CPU 下也会有类似问题，所以这里推荐指标**仅仅是自我对比，不一定是横向对比**。

→ 这句话正好解释了本项目的**分类器悖论**：它在 NPU 上比 CPU 慢（0.74×），
不是「NPU 不行」，而是**这个网络结构对 NPU 不友好**。

## 2. 把判据量化到我们的模型（`_opsprof.json`）

| 模型 | Conv | `Cin&Cout` 都 16 倍数 | **利用率上限** | 任一 `<16` | opset |
|---|---|---|---|---|---|
| 分类器 `litemodel_cls_96x` | 52 | **10** | **19.2%** ⚠️ | 30 | 11 |
| 识别器 `rpv3_mdict_160_r3` | 36 | 20 | 55.6% | 14 | 11 |
| 检测器 `y5fu_320x_head` | 85 | 60 | 70.6% | 22 | 12 |
| **识别器 `lprnet`（候选）** | 16 | **13** | **81.2%** ✅ | 1 | 9 |
| `lprnet_npufix`（改造版） | 16 | 13 | 81.2% | 1 | 9 |

**分类器只有 10/52 个卷积达标**，且 3 个卷积双 `<16` → 按 `(Cin×Cout)/256` 折算，
能吃到的算力只有最大值的零头。这与 §V-C 实测的「分类器 NPU 1.054 ms vs CPU 0.780 ms（0.74×）」
**方向完全一致**：不是实现问题，是结构问题。

## 3. 上机验证（本轮新做）

### 3.1 转换

```
bash omg_conv/convert_one.sh lprnet.onnx        om_lprnet        input 1,3,24,94 kirin9020
bash omg_conv/convert_one.sh lprnet_npufix.onnx om_lprnet_npufix input 1,3,24,94 kirin9020
→ OMG generate offline model success.
  om_lprnet.om        916,283 B   魔数 IMOD
  om_lprnet_npufix.om 915,413 B   魔数 IMOD
```

### 3.2 真机结果（干净单次抓取）

```
CANN TRY model=models/probe_om_lprnet.om
  compat=0  construct=ok  set_device_rc=0  build_rc=0  executor=ok
  in0  = input   F32 [1x3x24x94]
  out0 = output  F32 [1x1x68x18]
  run_tensors=1in/1out   run_rc=0   run_ms_each=2.686|2.059|1.805
  o0_sum=-62362.3169

CANN TRY model=models/probe_om_lprnet_npufix.om
  compat=0  build_rc=0  executor=ok   run_rc=0
  run_ms_each=2.906|1.466|1.366
  o0_sum=-62362.3169      ← 与原始版逐位相同
```

**两者都全绿跑通**，且 `o0_sum` 逐位一致 → 说明 ADR-007 §4 的 Transpose 改造
**数值上等价**（这是它第一次被真机验证）。

### 3.3 但分区证据推翻「换模型 = 全落 NPU」

从同一份日志提取每个模型的 CANN 分区决策（`partitioner_strategy.cpp ToPartitionerList`）：

| 模型 | 分区决策 | NPU 图加载 | **NPU 明确拒绝的算子** |
|---|---|---|---|
| `probe_hiai_imagenet.om`（对照） | `NPU:1, CPU:0` | 7 | Permute, Reshape |
| **`probe_om_lprnet.om`** | **`NPU:2, CPU:1`** | 2 | **ReduceMean** |
| **`probe_om_lprnet_npufix.om`** | **`NPU:2, CPU:1`** | 2 | **ReduceMean** |
| `probe_om_dethead.om` | `NPU:1, CPU:0` | 2 | — |
| `probe_om_cls.om` | `NPU:1, CPU:0` | 2 | — |
| `probe_om_rec.om` | `NPU:3, CPU:2` | 3 | Permute, Reshape |
| `probe_om_det.om`（编译期拒） | `NPU:4, CPU:3` | 0 | Activation, Add, Mul, Permute, Pow, Reshape, StridedSliceV2, Sub |
| `.ms ×3`（编译期拒） | 未走到分区 | 0 | — |

**三条硬结论**：

1. **LPRNet 不是整图 NPU** —— `NPU:2, CPU:1`，`ReduceMean` 被 NPU 库明确拒绝
   （`fe_sub_stores_manager.cc CheckSupported(314)::"the op name [ReduceMean_58] type [ReduceMean]
   is not supported in npucl store [elementary_lib]"`）。
   这**修正了本证据初稿的说法**：LPRNet 并非「零差评算子」，它有 `ReduceMean×5` +
   `Pow×4` + `Div×4`（L2 归一化），而 `ReduceMean` 恰是 NPU 不支持的算子。
2. **`npufix` 改造对 CANN 无用** —— 改前改后分区完全相同（都是 `NPU:2, CPU:1`、都因
   `ReduceMean` 被拒）。ADR-007 §4 假设「Transpose 是拦路虎」，在 **MS Lite** 路径上成立
   （当时 `LANDED=CPU`），在 **CANN** 路径上不成立 —— 拦路的是 `ReduceMean`。
3. **`cls` 与 `dethead` 是唯二 `CPU:0`（纯 NPU）的模型**，但 `cls` 仍比 CPU 慢 →
   进一步印证 §2 的**结构利用率**解释（19.2%），而非「有没有落 NPU」。

## 4. 跨工具链判决分歧（同一模型，两条路径结论不同）

| 模型 | MS Lite 路径（ADR-007） | **CANN 路径（本轮）** |
|---|---|---|
| `lprnet` | `LANDED=CPU`（6 个 `perm=[0,3,2,1]` Transpose） | **`build_rc=0 / run_rc=0`**，分区 `NPU:2, CPU:1` |
| `lprnet_npufix` | 未测 | **`build_rc=0 / run_rc=0`**，分区同上 |

→ **同一个 ONNX，两条工具链给出不同判决**。这是能写进论文的观察：
「NPU 支持性」不是模型的内在属性，而是**模型 × 工具链**的联合属性。

## 5. 口径声明（不许夸大）

- **输入尺寸不同，不能横向比快**：`om_lprnet` 输入 24×94，`om_rec` 输入 48×160，
  FLOPs 不同。LPRNet 稳态 1.366–1.805 ms vs rpv3 3.196 ms，**差里混着输入变小和结构变友好
  两个同向因素，本轮无法分离**。能确证的只有：**LPRNet 在 CANN 上跑得通、且结构按官方判据更友好。**
- **只测了「能跑」，没测「准」**：LPRNet 是另一套识别网络（CTC 字典 67 字符，与项目现有
  77 项字符表不同）。要替换必须重做精度验收（期望串 `苏ED5172`）。**本轮不下「可以替换」的结论。**
- **只跑 3 次**：`run_ms_each` 是 3 次采样（首次含建图），不是 30× 正式口径；
  且两轮独立运行稳态有波动（1.402 / 1.805 ms）→ **该数字不具论文级稳定性**，
  要引用须先补 30× 协议。

## 6. 边界与未做

- 换模型**不解决检测器的精度放大问题**（ADR-009：L2 差 0.077% → keypoint 放大 ≈10 px）。
- **异构执行未实测**：官方「异构（可选）」允许指定部分 OP 在 CPU、部分在 NPU ——
  这正是 LPRNet / rpv3 的 `ReduceMean`/`Permute` 该走的路，本轮只观察到分区器**已经自动**
  做了（`NPU:2, CPU:1`），但没有做「手工指定 + 性能对比」。
- 分类器没必要换：它 CPU 只要 0.78 ms，换模型收益上限极低。

## 7. 复现

```
# 算子画像（判据计算）
.venv/Scripts/python.exe <脚本>  → _evidence/_opsprof.json

# 换模型转换
bash omg_conv/convert_one.sh <x.onnx> om_x <inname> 1,3,H,W kirin9020

# 上机：App「探针控制台」→ 自检链末尾 cannProbe 自动跑
# 抓日志：hdc shell "hilog -x" | grep -E "CANN TRY|ToPartitionerList|npu_graph_executor_om"
```
