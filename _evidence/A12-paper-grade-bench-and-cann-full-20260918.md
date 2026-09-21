# A12 · 论文级数据落定：三模型 30× 正式基准 + 三模型全部跑通 CANN + 论文 §V 重写

- **日期**：2026-09-18 14:15–15:00
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **动机**：论文 §V-E 曾写「三个 LPR 模型没在 NPU 上跑过，属 future work」——
  **该陈述已被本日实测证伪**，论文在低报自己的成果。本轮把缺口补齐并改写论文。
- **证据**：`_evidence/bench30_matrix_20260918.log`（30× 基准）、
  `_evidence/cann_multout_20260918.log`（CANN 全模型）、`_evidence/cann_selfmade_om_20260918.log`

---

## 1. 论文级正式基准：三个生产模型 × 30 次 p50

把 `probeAll()`（本就实现为 warmup 5 + 正式 30 次 p50，但**从未被调用**）接进自检链，
并把模型集从「图手术变体」改成**三个生产模型**。结果（同一次运行）：

| 模型 | 请求后端 | **实际落点** | **p50 (ms)** | mean | L2 | CPU p50 | 加速比 |
|---|---|---|---|---|---|---|---|
| 检测器（裸 head，1.85 MB） | `nnrt` | **NPU** `NPU_ohos…kirin8020_v2_0` | **5.2273** | 5.3523 | 1043.4696 | 7.8289 | **1.50×** |
| 识别器 rpv3（4.48 MB） | `nnrt` | **NPU** `NPU_ohos…kirin8020_v2_0` | **4.0172** | 4.1123 | 2.9997 | 9.1359 | **2.27×** |
| 分类器 fp32（1.61 MB） | `nnrt` | **NPU** `NPU_ohos…kirin8020_v2_0` | **1.0541** | 1.0882 | 0.9980 | 0.7795 | **0.74×** |
| 分类器 fp16（0.82 MB） | `nnrt` | **CPU（静默回落）** | 0.7969 | 0.7919 | 0.9999 | — | 无 NPU kernel |
| 检测 / 识别 / 分类 | `gpu` / `kirin` | 全部 CPU（编译期判否） | 7.29 / 9.22 / 0.87 | — | — | — | — |

**三条结论，每条都反直觉，且都进论文：**

1. **识别器是唯一明确受益的（2.27×）** —— 也正好是对数值扰动最鲁棒的（CTC argmax 不因概率微扰翻字符）。
2. **检测器快 1.50× 但不可用** —— NPU 输出 L2 与 CPU 差 **0.077%**，而 keypoint = `logit × anchor`（anchor 最大 433），
   把相对扰动放大成 **≈10 px** 位移 → 翻字符（`苏ED5172` → `苏E05172`）。生产链因此保持 `det=CPU`。
3. **分类器在 NPU 上更慢（0.74×，慢 35%）** —— §V-B 的「小图规则」在本项目自己的模型里复现。
   fp16 变体更糟：**根本构不出 NPU kernel 且静默回落 CPU、不返回错误** —— 只看请求后端会把 CPU 数字
   当成 NPU 数字。只有 `LANDED=` 字段能区分。

## 2. 三个模型全部跑通 CANN/HiAI（含生产配置的检测器）

### 2.1 用 DDK 自带 OMG 自转 `.om`

| 步骤 | 结果 |
|---|---|
| 工具链 | `DDK-tools-next-6.0.1.0.zip`（239 MB）+ `kirin9020-plugin-next-6.0.1.0.zip`（40 MB） |
| OMG 入口 | `tools/tools_omg/omg` 是**包装脚本**（6.6 KB），真正二进制在 `master/omg`（178 KB） |
| 关键约定 | **`--platform kirin9020` 必需**（插件要放在 `tools/platform/`，脚本会报 `Please install platform plugin first`） |
| 自带工具链 | `master/x86_64-pc-linux-gnu-6.3.0/` 带 glibc 2.35 + **`libcrypto.so.1.1`** → 我担心的 libssl1.1 缺库问题自动消失 |
| 平台事实 | `kirin9020.ini`：`AIC_version=AIC-L-310`、`dav-l310`；`kirin9020.json`：`model_version=V310`、`kernel_magic=Ascend310B` |

转换命令（`omg_conv/convert.sh` 已封装，含自动定位 omg + 组装 `LD_LIBRARY_PATH`）：

```bash
bash ddk/tools/tools_omg/omg --model X.onnx --framework 5 --output out/om_X \
     --platform kirin9020 --input_shape <name>:1,3,H,W
```

**4/4 转换成功**（`om_det` / `om_dethead` / `om_rec` / `om_cls`，魔数均为 `IMOD`）。

### 2.2 设备侧结果（同一次运行、同一条 NNRt 入口）

| 模型 | `build_rc` | `run_rc` | 耗时（3 次） | 输出 |
|---|---|---|---|---|
| 厂商 `.om`（对照） | 0 | 0 | 2.748 \| 0.949 \| 0.926 | `o0_sum=1.0005`（1000 维 softmax） |
| **`om_rec`（识别器）** | 0 | **0** | 4.550 \| 3.873 \| 3.842 | `o0_sum=19.9978`（20 步 × ≈1.0） |
| **`om_cls`（分类器）** | 0 | **0** | 1.584 \| 0.999 \| 0.959 | `o0_sum=0.9981`（3 类概率） |
| **`om_dethead`（裸 head = 生产配置）** | 0 | **0** | 5.025 \| 5.224 \| 6.395 | 3 个 head 全部出数 |
| `om_det`（带 decode） | **1** | — | — | 编译期拒 |
| `.ms` ×3 | **1** | — | — | 格式不对（I/F 对照） |

**结论：生产配置的三个模型全部能在 NPU 上经 CANN 跑通。** 唯一被拒的是「把 decode 烘进图里」的那份，
而生产链本来就不用它（ADR-006 已把 decode 移到 C++）。

**跨工具链互证**：CANN 侧（rec 3.87 / cls 0.96 / det-head 5.0–6.4 ms）与 MS Lite 侧
（4.02 / 1.05 / 5.23 ms）量级一致、结论一致 → **瓶颈是加速器，不是框架**；
且**两套独立编译器都拒绝同一个 rank-5 常量构造**，比单边证据强得多。

## 3. ⚠️ 本轮最重要的方法论教训：探针的 bug 伪装成模型的缺陷

裸 head 检测器**第一次测出 `run_rc=1`**，看起来就是"执行失败"。真相：

```
out_count=3   out0=[1x45x40x40]  out1=[1x45x20x20]  out2=[1x45x10x10]
而探针只建了 1 个输出张量、只传 1 个 → RunSync 返回失败
```

运行时把**张量数量不匹配**报成泛化的失败码，调用点**与"算子不支持"无法区分**。
→ **一个假设"单入单出"的测量工具，会把能跑的模型判成坏的**，而且产出一个看起来完全合理的负结论。

**修复**（`nnrt_probe.cpp`）：改为按 `inN`/`outN` 建齐全部张量、全部传给 `RunSync`、逐输出统计
（`o0_sum/o0_argmax/o0_max` …）。修复后 `run_rc=0`。

> 这条已写进论文 §VI（"a lesson about the instruments, not the hardware"），因为 §IV 主张的
> "比中间量、不比端点"同样适用于**测量装置本身**。

## 4. 论文改写清单（`paper/ieee-lpr-paper.md`）

| 位置 | 改动 |
|---|---|
| Abstract | 追加三模型 30× 结果（2.27× / 1.50× / 0.74×）+ 跨工具链格式门槛结论 |
| §I 贡献 | 第 5 条扩写（含两个负结论）；**新增第 6 条**（跨工具链检验） |
| §V-A 表 | 新增「Second path: CANN/HiAI via `OH_NN*`」一行 |
| **§V-C（新增）** | 三模型 30× 正式基准表 + 三条反直觉结论 + **「框架」列必需**的方法学声明 |
| §V-D/E/F/G | 原 C→D→E 顺延；**§V-F 重写**（覆盖范围 + 跨工具链 + 多输出教训） |
| §VI | 新增两段：跨工具链互证的价值；测量装置也会骗人 |
| §VII | 第 2 条改为「CANN 侧为 3 次重复」；**第 3 条「backbone-only」删除**（已被证伪），改为"无加速的端到端流水线数字" |
| Conclusion | 新增一段：三个阶段三个相反结论，单一端到端数字会掩盖全部 |
| Keywords | 增补 CANN / Neural Network Runtime / heterogeneous inference |
| Appendix A | 新增四行，把新数字追到 `bench30_matrix_*.log` 等文件 |

## 5. 边界（论文已按此标注）

- §V-C 的 30× 是 **MS Lite → NNRT** 路径；§V-F 的 CANN 侧是 **3 次重复**，标注为 indicative。
- **没有**"NPU 上的端到端流水线"数字：letterbox / 透视变换 / 解码都跑在 CPU，且是端到端大头。
- §V-B 的六个骨干网络仍是单轮数据（历史记录），本轮未重测。
- CANN 侧未做 warmup，稳态取自第 2/3 次；首轮含首调用开销。

## 6. 复现

```
App 内「探针控制台」→「正式基准 30×」按钮  → hilog tag `LprMatrix`，每项 p50/mean/L2/maxAbs/dtype/verdict
自检链末尾也会自动跑（e2e 矩阵 → ncnn 槽位 → CANN 探针 → 30× 基准）
```