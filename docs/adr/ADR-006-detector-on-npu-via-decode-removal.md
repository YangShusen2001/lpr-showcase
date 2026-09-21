# ADR-006：检测模型上 NPU —— 裁掉 decode，并定位到真正的算子缺口

- **状态**：已接受（真机实测，含对照实验）
- **日期**：2026-09-17
- **前置**：ADR-004 §4.1（判定「唯一出路是导出裸 head」）、ADR-005 §8（列为待办）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135
- **证据**：`_evidence/bare_head_npu.txt`、`head_probe_raw.txt`、`dec_probe_raw.txt`
- **工具**：`tools/npu_op_anatomy.py`、`export_bare_head.py`、`convert_bare_head.py`、
  `export_decode_stages.py`、`verify_head_decode.py`

---

## 1. 问题

ADR-004 §4.1 已证明：y5fu 检测模型被 NPU 拒收，原因是 **12 个 rank-5 常量**
（anchor-grid decode 被烘进计算图），并给出唯一出路——**把 decode 移出计算图**。
当时试过的 4 种图手术（s1 / s1r4 / flat / surgery）全部失败，因为那些手术都在
**保留 decode 的前提下**改 rank。

本 ADR 换了思路：**不修改 decode，而是把 decode 整段裁掉**。

---

## 2. 方法：按「产生 head 张量所需」反向裁剪

`_evidence/head-anatomy.json` 已经定位了三个检测头（1×1 conv，45 = 3 anchors × 15）：

| 节点 | 输出张量 | 形状 |
|---|---|---|
| `Conv_500` | `948` | `[1, 45, 40, 40]` |
| `Conv_596` | `1061` | `[1, 45, 20, 20]` |
| `Conv_692` | `1174` | `[1, 45, 10, 10]` |

裁切算法（`tools/export_bare_head.py`）：

1. 从三个 head 张量**反向可达**（DFS 到 graph input / initializer 叶子），
   得到「产生它们所必需」的节点集合；
2. 只保留这个集合，删掉其余 **103 个 decode 节点**；
3. 把三个 head 张量提升为新的 graph output；
4. 清理不再被引用的 initializer（**rank-5 常量就是在这里离开的**）。

结果：**416 → 313 节点，rank-5 常量 12 → 0**，onnxruntime 验证三个输出可正常推理。

---

## 3. 对照实验：两个切点，结果相反

为区分「是 rank-5 常量的问题」还是「rank-5 本身的问题」，导出了两个版本：

| 版本 | 切点 | 输出形状 | 实际落点 | p50 |
|---|---|---|---|---|
| **A** | head conv 输出（`948/1061/1174`） | `[1,45,H,W]` **rank-4** | **NPU** `NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0` | **4.95 ms** |
| **B** | sigmoid 之后（`980/1093/1206`） | `[1,3,H,W,15]` **rank-5** | CPU（回落） | 7.52 ms |

> B 版等价于 `982/1095/1208`——因为 rank-5 常量 `981/1094/1207` **全为 0**，那个 Add 是空操作。
> 这一点是先用 onnxruntime 把中间张量导出为图输出、再与 `sigmoid(raw)` 逐通道对比确认的
> （`tools/export_decode_stages.py`）：ch0/1/2/4/13 diff = **0.000000**。

**A 版成功，意味着 ADR-004 §4.1 的判定是对的**：
把 decode 移出计算图，检测模型就能上 NPU。**4.95 ms vs CPU 9.07 ms（1.83×）。**

---

## 4. 关键发现：真正的算子缺口是 Reshape / Permute，不是 rank

B 版为什么回落？原始日志直接给出答案：

```
W AI_NPUCL: CheckSupported: op [Reshape_501] type [Reshape] is not supported in npucl store [fe_lib]
W AI_NPUCL: CheckSupported: op [Reshape_597] type [Reshape] is not supported
```

`Reshape_501` / `Reshape_597` 正是 **decode 的第一个 Reshape**（`[1,45,H,W] → [1,3,15,H,W]`）。
它不在 NPU 算子库里 → **从它开始的整段子图回落 CPU**。

> **⚠ 本节结论已被 ADR-007 修正**：`Reshape` / `Permute` **本身在官方支持列表里**，
> 不是「算子缺口」。真正判否的是 **perm 的 order**（NPU 只支持 `[0,1,3,2]`）
> 与 **张量 rank**（≤4）。`Reshape_501` 失败是因为它产出 **rank-5** 张量，
> 触发 `permute_check_support: realdimCnt > 4`，而不是 Reshape 不可用。
> 精确判据见 ADR-007 §2；据此可推出「NPU 友好模型」的选型标准。

**这个发现的价值超出本模型**：

> **Reshape / Transpose(Permute) 是麒麟 8020 NPU 的算子缺口。**

它同时解释了 **rpv3 识别模型的混合执行**（ADR-005 §6）——
`svtr_block.{0,1}/mixer` 里报的正是 `Reshape` 和 `Transpose`。

于是「能改算子吗」这个问题有了明确答案：

| 模型 | 要改什么 | 收益 |
|---|---|---|
| **y5fu 检测** | 裁掉 decode（已做）→ **已上 NPU** | CPU 9.07 → NPU 4.95 ms |
| **rpv3 识别** | 消除 `svtr_block/mixer` 的 **Reshape / Permute** | 上限约 9.0 → 3–4 ms（小） |

**不是 rank 问题，是 Reshape/Permute 支持问题。** 这修正了此前「rank-5 常量越界」
的表述——rank-5 常量确实是 y5fu 在**转换阶段**的拦路虎，
但一旦进到**算子检查阶段**，缺口是 Reshape/Permute。

---

## 5. decode 数学（已解开，供 C++ 实现）

`982` 之后的 26 个节点/尺度已完整 dump，逻辑是纯逐通道线性变换。以 40×40 尺度为例：

```
输入：raw head 张量 [1,45,40,40]  →  reshape  → [1,3,15,H,W]  →  transpose → [1,3,H,W,15]

# 通道布局（由 Slice 索引常量 0,2,4,5,7,9,11,13,15 确认）
#   ch0:1 = cx,cy    ch2:4 = w,h    ch4 = obj
#   ch5:13 = kpt0x,kpt0y,...,kpt3y    ch13:15 = cls0,cls1

cx = (sigmoid(t0) * 2 − 0.5 + grid_idx_x) * stride
cy = (sigmoid(t1) * 2 − 0.5 + grid_idx_y) * stride
w  = (sigmoid(t2) * 2)² * anchor_w
h  = (sigmoid(t3) * 2)² * anchor_h
obj = sigmoid(t4)
kpt_i_x = t[5+2i] * anchor_w + grid_px_x        ← 注意：kpt 通道【不做 sigmoid】
kpt_i_y = t[6+2i] * anchor_h + grid_px_y
cls = sigmoid(t13:15)
```

**两个反直觉点**（都是从实测值反推出来的，别凭直觉写）：

1. **kpt 通道在 `982` 里是原始 logits，不是 sigmoid 输出**。
   实测 `982` 的 ch5 范围是 `[-2.465, 1.396]`（超出 sigmoid 的 [0,1]），
   而 ch0/1/2/4/13 与 `sigmoid(raw)` **逐位相等**。
   图里只对 `[0:5]` 和 `[13:15]` 调了 Sigmoid，`[5:13]` 直接透传。
2. **kpt 乘的是 anchor，不是 stride**；加的是 grid **像素**值（`1014`，0..312 = 39×8），
   而 cx/cy 加的是 grid **索引**（`992`，0..39）再整体乘 stride。两条路径形式不同、结果等价。

anchors（9 对，可硬编码，无需任何大常量）：

| 尺度 | stride | anchors (w,h) |
|---|---|---|
| 40×40 | 8 | `(4,5) (8,10) (13,16)` |
| 20×20 | 16 | `(23,29) (43,55) (73,105)` |
| 10×10 | 32 | `(146,217) (231,300) (335,433)` |

grid 可用 `x*stride` / `y*stride` **程序生成**，不必存常量。

---

## 6. 交付物

| 文件 | 内容 |
|---|---|
| `tools/npu_op_anatomy.py` | 按算子类型解剖图，标出常量输入与 rank |
| `tools/export_bare_head.py` | 反向裁剪导出裸 head（支持任意切点） |
| `tools/convert_bare_head.py` | ONNX → `.ms`（fp16 / fp32） |
| `tools/export_decode_stages.py` | 把中间张量提升为图输出，逐通道对比定位切分点 |
| `tools/verify_head_decode.py` | decode 公式验证（**当前 MISMATCH，见 §7**） |
| `y5fu_320x_head_{fp16,fp32}.ms` | **rank-4 版本，已上 NPU**（967 KB / 1.85 MB） |
| `y5fu_320x_dec_{fp16,fp32}.ms` | rank-5 版本，落 CPU（仅作对照留档） |

---

## 7. 未完成

- [ ] **C++ 实现 decode**（§5 的公式），把 rank-4 裸 head 接进 `lpr_pipeline.cpp`。
      当前流水线的检测阶段仍是原 `y5fu_320x_sim.ms`（含 decode，落 CPU）。
- [x] `tools/verify_head_decode.py` 曾报 **MISMATCH**（maxAbsDiff=173）——
      已修复（2026-09-17 晚）：kpt 通道按 §5 改为「原始 logits × anchor + grid 像素、
      不做 sigmoid」，并修两处实现（广播升维、缩进）。复测
      **maxAbsDiff=0.000061 / meanAbsDiff=0.000000 → MATCH**
      （证据 `_evidence/verify_head_decode_match_20260917.txt`）。decode 允许移植 C++。
- [ ] 检测阶段耗时的大头尚未拆解：流水线里 `tDetectMs` 包含
      letterbox（1920×1080 → 320×320 双线性）+ NCHW 打包 + 推理 + decode，
      实测 63.7 ms，而单独 bench 原模型只要 9.07 ms。
      **差 7 倍，预处理很可能是主要成本**，需分段计时后单独优化。
- [ ] 端侧 ≥5 轮 + 标注 thermal level（与 ADR-004 §8 同项）。

---

## 8. 工程坑（本轮）

- **`onnx.load()` 会把 `.onnx.json` 当 JSON 解析**（按扩展名判断格式）→
  `UnicodeDecodeError`。用 `ModelProto().ParseFromString(open(p,'rb').read())` 绕开。
- **protobuf repeated field 不支持切片赋值**（`g.node[:] = [...]` 报
  `TypeError: does not support assignment`）→ 用 `del g.node[:]` + `g.node.extend(...)`。
- **反向可达必须把 initializer 当叶子**，否则常量张量会被误报为「无 producer」。
- **`cp` 会把只读权限带进 rawfile**，restool 删不掉旧文件 →
  `Error Code: 11204003 Failed to delete the directory or file`。
  复制后必须 `chmod u+w`。
- **绝不手删 `intermediates` 目录**：会让 hvigor 增量状态错乱，
  报 `validateModulePage: Cannot read properties of undefined (reading 'module')`
  （00308018），且清 build / 清缓存都无效。
  **正确做法是 `bash build.sh clean` 再重建。**
