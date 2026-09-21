# ADR-007：NPU 算子约束的真实边界，与按支持清单选模型的路径

- **状态**：已接受（官方文档 + 真机实测双向对照）
- **日期**：2026-09-17
- **前置**：ADR-006（裸 head 上 NPU）；本 ADR 回应「不支持的话就去官方看看支持什么模型」
- **后续**：ADR-014（按 16 倍数判据选模型 → LPRNet 真机能跑但分区仍含 CPU）、
  ADR-015（§5 的「67 字符需与 77 项对齐」经查是**一条不存在的欠账**，已撤销；
  且本 ADR 涉及的 NPU 数字全部是 fp16 条件下的观测）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135
- **证据**：`_evidence/lprnet_probe_raw.txt`、`bare_head_npu.txt`
- **工具**：`tools/npu_op_anatomy.py`、`tools/convert_onnx_ms.py`

---

## 1. 官方支持清单：读起来「都支持」，但不够

华为官方「支持的算子」列表（`developer.huawei.com/consumer/cn/doc/hiai-Guides/supported-operators`）
按类别列出，与本次失败直接相关的几条**全部在列**：

| 算子 | 官方类别 | 在列表里？ |
|---|---|---|
| `Reshape` | array_defs | ✅ |
| `Permute` | detection_defs | ✅ |
| `Swish` / `Mish` / `HardSwish` | nn_defs | ✅ |
| `GatherV2D` | array_defs | ✅ |
| **`LSTM`** | nn_defs | ✅ |
| `SSDDetectionOutput`、`NonMaxSuppressionV3D/V6` | detection/image_defs | ✅ |

**只看这张表会得出错误结论**——「算子都支持，那随便什么模型都能上」。
实际不是：**支持的算子是有条件的**，条件写在编译器内部检查里，不在文档上。

---

## 2. 真正的判据：从 NPU 编译器日志反推

把 `nnrt` 档的失败日志按编译器模块归类，约束就露出来了：

```
permute_check_support.cc      IsSupport(21):: realdimCnt > 4
reshape_check_support.cc      ReshapeCheckFunc(41):: check reshape dimInfo fail
permute_matmul_fusion_pass.cc IsMatch(91):: permute node:... only support order 0132
squeeze_graph_builder.cc      RecoverShape(33):: wrong input dims
```

**译成可执行的选型判据**：

| # | 约束 | 来源 |
|---|---|---|
| 1 | **张量维度 ≤ 4**（rank-5 直接判否） | `realdimCnt > 4` |
| 2 | **Permute/Transpose 只支持 order `[0,1,3,2]`** | `only support order 0132` |
| 3 | **Reshape 的输出形状必须静态可推导** | `check reshape dimInfo fail` |
| 4 | Squeeze 的输入维度必须匹配 | `wrong input dims` |

**约束 1 和 2 是本次两个案例的直接死因**，且都**不在官方文档里**。

> **选型判据（本 ADR 的核心产出）**：
> **一个模型要上麒麟 8020 NPU，必须满足：所有张量 rank ≤ 4，
> 且所有 Transpose 的 perm 只能是 `[0,1,3,2]`（或可被融合消除）。**

---

## 3. 两个案例的对照

### 3.1 y5fu 检测：裁掉 decode → 上 NPU ✅

见 ADR-006。裁到 head conv 输出（rank-4、无 Transpose），**4.95 ms**。
裁到 sigmoid 之后（rank-5、含 `Reshape_501`）→ 回落 CPU。
**两个约束同时踩中**。

### 3.2 LPRNet 识别：结构几乎完美，仍被一个 perm 挡住 ❌

LPRNet（`rknn_model_zoo/examples/LPRNet`，纯卷积 + CTC，无 RNN）的静态指标**全部合格**：

| 指标 | LPRNet | 判定 |
|---|---|---|
| 节点数 | 63 | 轻 |
| rank ≥ 5 的常量 | **0** | ✅ 过约束 1 |
| initializer 最大 rank | **4** | ✅ |
| 张量 rank 分布 | 只有 3 和 4 | ✅ |
| Reshape 节点 | **0** | ✅ 过约束 3 |
| 算子集合 | Conv/Relu/MaxPool/Pad/AvgPool/Pow/ReduceMean/Div/Concat | 全在官方列表 |
| 输入 / 输出 | `[1,3,24,94]` / `[1,68,18]` | rank-4 / rank-3 |

**但实测 `LANDED=CPU`**，原因是 **6 个 Transpose 全是 `perm=[0,3,2,1]`**，
而 NPU 只认 `[0,1,3,2]`。

**结论：静态指标全绿 ≠ 能上 NPU。`perm` 是独立的一道门。**

---

## 4. LPRNet 的 Transpose 可以全部消除（已找到方案）

6 个 Transpose 成三对，模式固定：

| 位置 | 结构 | 等价改写 |
|---|---|---|
| `_2 → _3 → _4 → _5` | `MaxPool k=3x3 s=1` → `T` → **`MaxPool k=1x1 s=1`** → `T` | **整体恒等**（`[0,3,2,1]` 自逆，1×1 池化不改数据）→ **直接删** |
| `_14 → _15 → _16 → _17` | `MaxPool k=3x3 s=[1,2]` → `T` → **`MaxPool k=1x1 s=[1,2]`** → `T` | 1×1 窗口 + stride 2 = **纯抽样** → **沿 W 维 `Slice(step=2)`** |
| `_34 → _35 → _36 → _37` | `MaxPool k=3x3 s=[1,2]` → `T` → **`MaxPool k=1x1 s=[1,4]`** → `T` | 同上 → **沿 W 维 `Slice(step=4)`** |

**为什么成立**：`Transpose[0,3,2,1]` 是**自逆**的（作用两次回到原状），
而中间那个 `MaxPool` 的窗口是 **1×1**——1×1 窗口意味着它不聚合邻域，
只按 stride 抽样，所以整对的效果就是「在转置后的布局上沿某一维按步长抽样」，
等价于在原布局上对对应维做 `Slice`。

**`Slice` / `StridedSlice` 都在官方支持列表里**，且都是 rank-4 操作。

---

## 5. 下一步

- [ ] **改造 LPRNet**：删掉第 1 对 Transpose，把第 2、3 对替换成 `Slice(step=N)`，
      用 onnxruntime 验证输出与原始 LPRNet **逐元素一致**（这是改造正确性的唯一判据）。
- [ ] 转换 → 上机测 `nnrt` → 期望 `LANDED=NPU`。
- [ ] 若通过：LPRNet 替换 rpv3（SVTR）作为识别模型，配合 y5fu 裸 head 组成
      **检测 + 识别全 NPU** 的流水线。
- [x] ~~LPRNet 的 CTC 字典（67 字符）需与项目现有 77 项字符表对齐或替换。~~
      **2026-09-18 撤销：这是一条不存在的欠账。** 上游官方 `_load_data.py` 的字符表
      与我此前经验恢复的 77 项映射**逐项一致**（含 blank=67 起点），无需对齐。
      证据：`_evidence/A16-real-dataset-accuracy-20260918.md` §3。
- [ ] 备选：若 LPRNet 改造不通过，按 §2 的选型判据重新筛选
      （重点看 `perm` 与 rank，而不是算子名）。

---

## 6. 工程提示

- **GitHub 直连在本机不可用**（`curl: (28) Failed to connect to github.com port 443`），
  `raw.githubusercontent.com` 返回 301。可用镜像：
  `https://ftrg.zbox.filez.com/v2/delivery/data/<hash>/examples/LPRNet/lprnet.onnx`
  （1.7 MB，实测下载正常）。
- **`onnx.load()` 不能读 `.onnx.json`**（按扩展名当 JSON 解析）→ 用
  `ModelProto().ParseFromString(open(p,'rb').read())`。
- 判断一个模型能否上 NPU，**先跑静态扫描（rank + perm）**，再花时间做转换——
  转换一次几分钟，扫描一次几秒。
