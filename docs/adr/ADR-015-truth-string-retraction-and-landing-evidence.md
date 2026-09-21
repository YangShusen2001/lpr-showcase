# ADR-015 · 真值串推翻与 `match=` 语义变更；撤回「相机出帧率即上限」；NPU 全部数字均属 fp16 条件

- **状态**：已接受（2026-09-18）
- **前置**：ADR-002（跨语言移植保真）、ADR-009（NPU 数值漂移 → det 落 CPU）、
  ADR-010（300 ms 归因 letterbox）、ADR-014（按 NPU 友好度选模型）
- **后续**：本 ADR 是 A14/A15/A16 三批证据进入上层文档的**唯一入口**；凡引用
  `match=`、`111/000`、相机 fps、NPU 延迟者，均须同时读本 ADR
- **证据**：`_evidence/A16-real-dataset-accuracy-20260918.md`、
  `_evidence/A15-expected-string-matrix-release-20260918.log`、
  `_evidence/A14-camera-realtime-e2e-20260918.md`、`_evidence/lprnet_accuracy_20260918.md`、
  `_dataset/real/README.md`
- **代码**：`lpr-harmony/.../ets/pages/Index.ets`（`REFERENCE_CODE_RPV3`）、
  `.../cpp/ms_engine.cpp`（`PlanFor`/`MsLoad`）、`.../cpp/ncnn_engine.cpp`（`VkCoverage`）

## 问题

三批新证据（A14 相机实时、A15 release 期望串矩阵、A16 第三方真实集精度）暴露出三个
**方法论级**缺陷，它们不是"数字旧了"，而是"数字的含义被错标"：

1. 一直被当作"样本真值"的 `苏ED5172`，其实**从不是人工标注** —— 它只是
   `tools/lpr_reference.py`（即 rpv3 本身）的输出。端侧矩阵拿它当验收红线
   （`Index.ets` 旧注释原文：「矩阵期望串：样本里的**真值**车牌」），
   等于**用被测系统的输出去验收被测系统**。
2. A14 §4.2.2 宣布「相机自己只给 15 fps，推理侧已彻底不是瓶颈」。但同一份证据 §4.1
   在同一暗场景报到达率 `19.92–20.06 fps` —— 同场景同档位不可能既 20 又 15。
3. 全部 NPU 数字（含"漂移 0.08% → det 不能上 NPU"这条生产链决策）都是在
   `EnableFP16=true` 下测的，而**这个 fp16 是代码硬编码的，不是实验选出来的**。

4. **det 在 NPU-fp32 下也翻字符**：经过多轮验证（fresh install + cache clear），
   det 在 fp16/fp32 下都会翻字符（苏 ED5172 → 苏 E05172），因此"NPU-fp32 不翻字符”的结论被推翻。
   cls 在 NPU 上无法构建 kernel，全部 fallback 到 CPU。rec 是唯一能在 NPU 上正确运行且有显著收益的模型。

## 决策

### 1. `苏ED5172` 降级为"参照串"，`match=` 降级为"与基线一致"

- 常量改名 `EXPECTED_CODE` → **`REFERENCE_CODE_RPV3`**，值不变（它仍然是有效的**移植保真
  锚点**：JS/原生 与 Python 参考是否逐字符相同，这个命题与真值对错无关，ADR-002 的
  8/8 结论**存活**）。
- **`match=` 的语义正式定义为：与同角色的 CPU 基线逐字符一致**。它用于抓"换后端引入的
  输出变化"，**不构成任何正确性或精度证据**。字段名保留 `match=` 是为了让 A4~A8/A15 的
  历史日志仍可比；BEGIN 行的 `expected=` 改为 `reference=`。
- 要声称"读对了"，只能引用带人工真值的第三方集：`_dataset/real/crops` n=1000，
  rpv3 **906/1000 = 90.6%**、LPRNet **888/1000 = 88.8%**、McNemar **p=0.1788** 不显著。
  → 本 ADR 起，"本项目刻意不报准确率"的旧口径**作废**（原声明见 README §8 /
  `docs/INDEX.md` §3 / 论文 §VII.5–.6 与附录 B，均已改写）。
- 黄金图的正确真值是 **`苏ED51712`**（8 字符）。支撑链四条，见 A16 §4.6：车牌在画面里被
  偏航压缩约 2×（真牌四边形比例 1.60 vs 标准 3.14）、标准比例下 rpv3 自己读出 8 字符
  （conf 0.987）、n=200 对照实验证明**水平拉伸从不新增字符而压缩 17% 吃字符**、
  牌面像素测量（`tools/plate_face_colour.py`）判真绿牌 → 新能源 8 位牌。

### 2. 撤回 A14 §4.2.2 的"相机上限"结论

撤回理由（可反驳）：所谓"零推理基准档"**仍然支付整帧 NV21→RGBA 转换**（其自身日志
`convMs` 非零），所以它测的是「取帧+转换上限」而非「相机上限」；且与 §4.1 的 20 fps
自相矛盾。同时 `frameMs = decodeMs + inferMs` 不含 ArkUI 重渲染与 napi 往返，
`1000/p50(frameMs)` 因此**系统性乐观**。

- `34.5 / 23.8 / 12.8 fps` 一律标注为**服务时间理论值**，**不得对外称 fps**；
  真实吞吐只看到达率 / 完成率两列。
- 复现条件未锁死前，"推理侧已不是瓶颈"与"det=CPU 12.7 ms 是大头"这两条**同时挂起**。
- 结案路径：新增「只数帧不取帧」档 + 固定片元回放台（同一段帧文件喂同一条流水线，
  把服务时间与相机到达率彻底解耦）。

### 3. 落点自证补强：回落必须在日志里读得出来

- MindSpore Lite 的每个后端计划都以 CPU 兜底（`PlanFor` 末尾无条件追加），因此
  `LANDED=CPU` 本身**分不清"我们选的"还是"回落的"**。现在 `MsSession` 记录
  `requested` 与 `fallbackFrom`，矩阵行输出 `engine=ms req=gpu fallback=GPU:fp16`。
- ncnn 行的 `effectiveVulkan` 只是选项回显（我们自己把 `opt.use_vulkan_compute`
  设成 true，它当然是 true）。补 `vkLayers=X/Y`（带 Vulkan 实现的层数 / 总层数）+
  `vkGpuCount` + `vkGpuName`：**X<Y 时该格是"GPU + 部分层 CPU 回落"，不能当纯 GPU 数据点**。
- 口径收益：ADR-004/005 的「MS Lite 的 gpu/kirin 全部静默回落 CPU」今后由 `fallback=`
  字段**直接自证**，不再需要读者信任一句结论。

### 4. 「det 不能上 NPU」降级为「det 不能在 NPU-fp16 上用」

`addNnrt()` 对每个 NNRT 设备恒传 `fp16=true`，计划里**从来没有 fp32 分支**（GPU 反而有
fp16/fp32 两试）。因此 ADR-009 那条"NPU 0.08% 漂移经 anchor 放大 ≈10 px → 翻字符"
是一次 **fp16 NPU 推理**的观测，而非 NPU 的固有属性。

- 新增可请求档 **`nnrt_fp32`**（`addNnrt(false)`），落点标签带 `#fp32` 后缀，保证
  fp16/fp32 两次落在日志里不会长得像同一个落点。
- **在完成 fp32 对照之前**，生产链 `det=CPU / rec=NPU / cls=CPU` 维持不变（保守），
  但论文与页面的表述必须限定为"fp16 条件下"。
- 附带发现：`TryBuild` 把 `SetPerformanceMode(HIGH)` 与 `EnableFP16` 一起包在
  `if (type != CPU)` 里，且 CPU 线程数硬编码 4 → **原生 CPU 后端的性能模式与线程数从未扫描**
  （ADR-004 §5.4 的"6 线程最优"只测过浏览器 WASM）。

## 后果

- A4/A5/A6/A8/A15 里所有 `111/000` 读作"与基线一致 / 与基线不一致"，**不再读作
  "对 / 错"**。原始日志不回改，只加本 ADR 指针。
- 论文 §IV.D 保真表数字不动，但"零翻字符"的表述统一降级；`README.md` §4 的两条
  "必须一起说出口的结论"之一（NPU 会翻字符）须补 fp16 限定。
- 「全 NPU 比生产档快 1.5×」这条收益归因**要打折，但折扣的理由不是"模型搞错了"**：
  `om_dethead` 经核对是**真检测器**（A12 §表：`run_rc=0`、5.025｜5.224｜6.395 ms、
  **3 个 head 全部出数**；A13 §分区表：`NPU:1, CPU:0`，即纯 NPU）。
  本轮曾有一份分析把它与厂商 codelab 的 `hiai.om`（SqueezeNet，IO `[1x3x227x227]→[1x1000x1x1]`、
  单输出）混作同一个模型并据此否定该收益 —— **那个论据是错的，已撤销**；它恰好说明
  「引用二手归因前必须回原始日志核一遍」，与本 ADR §1 的教训同构。
  真正的折扣来自**未控制的混淆因子**：A14 §4.2.6 两档的命中数不同（`×21` vs `×10`），
  少命中即少跑 rec/cls，因此"1.5×"里有多少来自后端、有多少来自"这些帧根本没有牌"从未分离。
  复测协议见 §未完成动作 1。
- 最有论文价值的产出：**保真度是一个比" agreement "更强的主张，但前提是被对齐的参照串
  本身是对的** —— 本项目用一次独立第三方集测量，抓到了自己参照串的错误。

## 边界声明

- 本 ADR **不推翻**：移植保真 8/8 一致、`det=NPU` 会改变输出（差分命题与真值位数无关）、
  rpv3 仍是生产识别器、LPRNet 不构成替换理由。
- 本 ADR **不给出**任何新的端侧性能数字 —— fp32 对照、到达率复测、CPU 线程扫描全部
  未完成，相关表须等 `_evidence/A17+`。
- 牌色口径冲突（cls 模型输出"蓝牌" vs 牌面像素测量"真绿牌"）**本 ADR 只登记不裁决**；
  在裁决前，页面/论文凡提该样本牌色处必须同时给两个口径并标明来源。
- 旧数据里 `plate_colour_analysis.py` 的按色细分（含"绿 hue 37.2%"）**作废**：该分类器
  用"亮于中位数的像素"判色，蓝牌上最亮的是白字 → 把 86 张蓝牌判成绿牌。

## 未完成动作

已完成：
1. **nnrt_fp32 对照已跑完**：实测 det 在 fp32 下也翻字符（match=000），因此"NPU-fp32 不翻字符”的结论被推翻。
   cls 在 NPU 上无法构建 kernel，全部 fallback 到 CPU。rec 是唯一能在 NPU 上正确运行且有显著收益的模型。
   → **生产档配置确认为 `det=CPU / rec=NPU-fp16 / cls=CPU`**。
2. **CPU PERFORMANCE_HIGH 已启用**：ms_engine.cpp 中 CPU 后端补上了 `OH_AI_DeviceInfoSetPerformanceMode(dev, OH_AI_PERFORMANCE_HIGH)`，榨干 CPU 性能。
3. **GEAR_ARRIVE_ONLY 档已实现**：CameraPage.ets 中实现了只数 callback 不 readLatestImage 的档位，用于测相机真实上限。
4. **细粒度计时已实现**：lpr_pipeline.cpp 中将 letterbox/encode/infer/decode/NMS 拆分成独立计时字段（tLetterboxMs/tEncodeInferMs/tDecodeNmsMs），并通过 napi_init.cpp 输出到 p0 字段。

待办：
5. 「只数帧不取帧」档 + 固定片元回放台 → 复测相机到达率上限，结案 §4.2.2。（需要手机解锁后验证）
6. 原生 CPU 后端 `SetPerformanceMode` × 线程数 {2,4,6,8,12} 扫描（同 ADR-004 §5.4 协议）。
7. 端侧验收表换成 n=1000 真实集基线；`tools/lprnet_golden_check.py` 等 n=5 判据脚本作废或加声明。
8. 牌色口径裁决（模型输出 vs 像素测量），以及 `index.html` / `demo.js` 展示标签是否跟改。
9. GPU pipeline cache 预热 + 层数 ladder 扫描 → 给出"GPU 有收益所需的最小单算子计算量”。（ncnn 版本不支持该 API，需考虑其他优化策略）
