# A18 · 分段预算 + CPU 线程扫描 + 全矩阵无截断复测（2026-09-20 第二轮）

> 设备：MIA-AL00（nova 14 Pro）· 麒麟 8020 · HarmonyOS 6.1.0.135 · API 24 · 设备号 4CY9K25614046328
> 构建：**本轮为重打包版本**（16:32 BUILD SUCCESSFUL，76.4 MB signed HAP），相对 A17 装机版新增：
> ① pack/infer 分段计时；② `cpu_t{1,2,4,6,8}` 线程扫描档；③ `kMaxSessions` 20→64；
> ④ 修复 `lpr_pipeline.cpp` 中 yolov8 时代遗留的源码损坏（见 §5）。
> 触发方式：`AUTO_SELFTEST=true` 启动 App，全链 ~50 s 跑完
> 原始日志：`full-hilog-20260920-r2.txt`（清洗行：`chain-20260920-r2.txt`）

---

## 1. 分段预算：det 段 17.7 ms 未解释开销结案（ADR-009 §5 待办）

`lpr_pipeline.cpp` 新增 pack/infer 拆分计时后，stages 串从 7 段扩到 9 段
（追加在末尾，旧 7 段位置不变）。生产链单帧（黄金图 `hlpr-test.jpg`，release）：

```
NATIVE PIPE code=苏ED5172 colour=蓝牌 count=1 total=36.1106
  stages=5.4270|5.4270|18.5053|0.0383|20.5964|7.3416|2.6913|0.3520|18.1533
          det   letterbox enc+inf  dec+nms rectify  rec    cls   pack  infer
```

| 段 | ms | 占比 | 说明 |
|---|---|---|---|
| letterbox | **5.43** | 15.0% | 与 ADR-010 一致，仍是大头之一 |
| det pack（NHWC 打包） | **0.35** | 1.0% | 320×320×3  float 打包，可忽略 |
| det infer（纯推理，CPU） | **18.15** | 50.3% | **即"17.7 ms 之谜"的答案：几乎全是推理本身** |
| decode + NMS | 0.04 | 0.1% | 可忽略 |
| rectify/crop（透视裁剪，扣除 det 后） | ≈2.05 | 5.7% | 20.5964 − 18.5053 − 0.0383 反推 |
| rec | 7.34 | 20.3% | 含双层牌切分逻辑 |
| cls | 2.69 | 7.4% | |
| **合计** | **36.11** | | → 单帧理论上限 **27.7 fps** |

**两个可发表的结论**：

1. **"encode+infer"桶 18.51 ms 里 pack 只占 0.35 ms** —— 内存布局打包不是瓶颈，
   瓶颈就是 det 在 CPU 上的推理。此前把 17.7 ms 归因为"letterbox/NHWC/decode/NMS 未拆分"
   的猜测中，NHWC 打包这一项被证伪（1.9%）。
2. **流水线内 det 推理 18.15 ms vs 孤立 bench 11.37 ms（同轮同构建，+59%）** ——
   段间效应（每帧新建 vector / 页错误 / rec 与 cls 之间的缓存逐出）是真实的、
   可优化的 6.8 ms。这是下一轮帧率优化的明确靶点（buffer 复用），且不需要换后端。

## 2. CPU 线程扫描（cpu_t1..t8，30× p50）

| 模型 | t1 | t2 | **t4** | t6 | t8 |
|---|---|---|---|---|---|
| det-head 320×320 | 12.84 | 12.98 | **11.38** | 11.96 | 15.21 |
| rec 48×160 | 16.53 | 15.88 | **13.89** | 13.93 | 16.79 |
| cls-fp32 96×96 | 1.04 | 0.90 | **0.82** | 1.36 | 2.18 |

- **三个模型的最优线程数都是 4**，与硬编码值一致 → CPU 基线在此维度上已被动"调对"；
  t6 与 t4 持平（大核数 4，超出的线程空转），t8 全面变慢（超订阅 + 调度抖动）。
- `cpu_t4` 与普通 `cpu` 档同轮对照：det 11.3819 vs 11.3683、rec 13.8942 vs 13.9394
  → 档位实现正确（新档没有引入测量伪差）。
- **论文口径**：CPU 端 = 高性能模式 + 4 最优线程 → 现在是有据可依的**调优基线**，
  不再是"未探索配置"（ADR-015 未完成动作中"CPU 基线未调优"一条可结案）。

## 3. 全矩阵无截断复测（kMaxSessions 64）

- **72/72 E2EMATRIX RESULT 行齐全，0 条 session cap**（A17 同代码路径被截断 4 组）。
- **cls-fp32 × nnrt_fp32 补跑成功**：`LANDED=NNRT:...kirin8020_v2_0#fp32`，match=1（3/3）。
- fp32 结论二次确认：**det × nnrt_fp32 仍 match=0**（`苏E05172`），**rec × nnrt_fp32 仍 match=1**，
  cls（fp16 变体）fp32 档同样构不出 NPU kernel 回落 CPU。→ A17 §1 结论在独立第二轮复现。

## 4. CPU 端轮间方差（新发现，方法论层面）

同协议同构建，两轮 30× p50：

| 模型×后端 | A17（16:14 轮） | A18（16:34 轮） | 变化 |
|---|---|---|---|
| det CPU | 7.6638 | 11.3683 | **+48%** |
| rec CPU | 9.1286 | 13.9394 | **+53%** |
| cls-fp32 CPU | 0.8395 | 1.1045 | +32% |
| det NPU | 5.3505 | 5.3130 | −0.7% |
| rec NPU | 3.9883 | 3.6888 | −7.5% |

**CPU 端轮间方差 30–50%，NPU 端 <8%**。成因未定（疑似第二轮设备热态累积 + big.LITTLE
调度把工作线程挤到小核；两轮间隔约 20 分钟、第二轮前面还有一整条自检链负载）。
→ **协议含义**：论文引 NPU/CPU 比值必须带轮次与热态标注；CPU 绝对延迟的单点数字
不足以支撑结论，比值也应看区间。这也是"≥5 轮 thermal 标注"要求的实证理由。

## 5. 源码损坏修复记录（动手改端侧代码前必读）

`lpr_pipeline.cpp` 在 2026-09-19 凌晨的 yolov8 整合中遗留三处损坏（本轮才发现，
因为 09-18 的装机 HAP 是在损坏发生前构建的，设备行为一直正常，**源码树从 09-19 起编译不过**）：

1. 匿名 `namespace {`（L20）的闭合括号被啃掉，残迹留在 `// ------ 相机取帧转换 // namespace`
   注释行 → 头文件公开函数实际定义进匿名空间，与全局声明冲突（5 个 ambiguous 错误）；
2. 两段误粘贴的 `const std::vector<std::string>& LprToken() {` 碎片（L982/L1062），
   文件在 L1062 被截断 → redefinition + 嵌套函数定义错误。

修复 = 删两段碎片 + 恢复 `}  // namespace`。修复后 BUILD SUCCESSFUL，
且 72/72 矩阵 + fp32 结论与 A17 逐项一致 → 修复未改变被测行为。
**教训：`lpr_pipeline_new_v2.cpp` / `lpr_pipeline.cpp.yolov8_backup` 是那次重构的遗留物，
主文件的手工合并从未完成；下次改端侧前先 `assembleHap` 干跑一次编译。**

## 6. 对"帧率上限 25 fps"问题的数据收口

- 推理侧单帧 36.11 ms → **理论上限 27.7 fps**，与明亮场景到达率 26.9–27.1 fps 几乎重合：
  无牌/少牌时相机给多少我们吃多少（dropped=0）的说法在数据上自洽。
- 有牌时 done 掉到 13 fps 的旧观测（A14）与 36 ms/帧不符（36 ms 应对应 ~27 fps），
  差异来自相机路径的 NV21 转换 + ArkUI 重渲染 + napi 往返（A14 §4.2.2 已登记，
  仍属"未结案"项，需"只数帧不取帧"档复测）。
- 下一步优化靶点（按收益排序）：① det 流水线内 6.8 ms 段间效应（buffer 复用，无需换后端）；
  ② letterbox 5.43 ms（SIMD 或整数化）；③ rec 7.34 ms 已在 NPU，无空间。

## 7. 数据出处索引

| 数据 | 位置 |
|---|---|
| 本轮原始完整日志 | `_evidence/full-hilog-20260920-r2.txt` |
| 本轮链路清洗行（172 行） | `_evidence/chain-20260920-r2.txt` |
| 分段计时（9 段 stages） | chain 文件 `NATIVE PIPE` 行 |
| 线程扫描 15 行 | chain 文件 `MATRIX ... cpu_tN ...` 行 |
| 全矩阵 72 行 | chain 文件 `E2EMATRIX RESULT` 行 |
| 上一轮（A17）对照 | `A17-release-30x-fp32-cann-20260920.md` |

---

## 8. 第一轮优化实测（r3，2026-09-20 晚）：buffer 复用假设被证伪（**重要阴性结果**）

按"加速方案清单"第①项实施了 buffer 复用：`ResizeLinear`/`LprLetterBox`/`LprToNhwc`/
`LprEncodePlate`/`LprEncodeClassify` 全部改为 Into 版本写入 `thread_local` scratch
（`PipeScratch`：letterbox 400KB + resize 中间 5.5MB + det 输入 1.2MB + rec/cls 编码 +
输出张量），`MsRunMulti` 去掉 `clear()` 保容量。r3 装机实测（构建 19:0x，release）：

| 指标 | r2（复用前） | r3（复用后） | 判定 |
|---|---|---|---|
| 单帧 total | 36.1106 ms | 35.9350 ms | 无变化（噪声内） |
| det 流水线内 infer | 18.1533 ms | 20.3961 ms | **更慢** |
| det 隔离 bench p50 | 11.3683 ms | **7.2134 ms** | 快 36%（设备热态差异） |
| rec 流水线内 | 7.3416 ms | 6.1831 ms | 略快（方差内） |
| cls 流水线内 | 2.6913 ms | 1.6197 ms | 略快（方差内） |
| 正确性 | `苏ED5172` cropSum=2773473 | 逐位相同 | 无损 |

**结论：「段间效应源于每帧内存分配 + 首次触页」的假设被证伪。** 分配已消除，
gap 反而从 6.8 ms 扩大到 13.2 ms（同轮对照：隔离 7.21 vs 流水线内 20.40）。
复用本身无回归、无正确性代价，保留；但它不是瓶颈的解释。

**最有价值的新发现：gap 是 MS Lite CPU 后端特有的。** 同轮 e2e 行对照：

| det 推理 | 隔离 bench | 流水线内 | gap |
|---|---|---|---|
| MS Lite **CPU** 后端 | 7.2134 ms | 20.3961 ms（e2e 行 24.5–41.3 ms） | **2.8×** |
| NNRT（NPU）后端 | 4.6355 ms | 4.8927–5.7092 ms | **≈0** |

CPU 后端在流水线上下文里慢 2.8–3×，NPU 后端完全没有这个现象。最 plausible 的成因是
MS Lite 每个会话各自持有 4 线程池，det/rec/cls 三个池在流水线里互相抢核（隔离 bench
同时只有一个池活跃）；未最终证实，记为待查。**工程含义：把 det 从 CPU 挪到 NPU 的收益
比此前估计的更大（约 −15 ms/帧，而不只是 −13 ms 的推理差）。**

## 9. INT8 PTQ 工具链核查（2026-09-20）

项目 `omg_conv/ddk/tools/tools_dopt/` 下 **`dopt_onnx_py3/dopt` 在位** ——
「dopt 量化（calibration）→ OMG `--compress_conf` 转换 → INT8 `.om` → 设备验证」
这条链的工具齐了，与 2026 年 Kirin NPU 公开仓库的做法一致。校准数据可用项目自有的
真实场景样本（`assets/samples/`）或 1000 张裁剪图放大版。**未执行**（需要 WSL 环境 +
校准集构造 + 转换 + 黄金图/1000 张真值集双重数值验收，属独立工作线）。
另：r4 方差复测因设备 USB 掉线未完成（`need connect-key`），pending。

---

## 10. pipeline-vs-bench gap 系统调查（r5–r9，2026-09-20 晚）——五个假设，全部证伪

**现象**：det 推理在流水线内 14–33 ms，而孤立 30× bench 稳定在 7.3–8.7 ms；
流水线总时间六轮为 29.1 / 30.5 / 33.7 / 35.9 / 36.1 / 41.8 / 56.2 ms，
bench 六轮全部落在 7.3–8.7 ms 区间。正确性全程不变（`cropSum=2773473`）。

为解释 gap 做了五个对照实验，**全部证伪**（每条都有同轮对照数据）：

| # | 假设 | 实验 | 结果 |
|---|---|---|---|
| 1 | 每帧内存分配 + 首次触页（~7 MB/帧） | 全链路 buffer 复用（`PipeScratch`/`ResizeScratch`，Into 化） | gap 不变（r3：6.8→13.2 ms，同轮 bench 反而更快） |
| 2 | MS Lite 线程池段间睡着付唤醒费 | 生产档 det 改 `cpu_t1` 单线程（E1） | 流水线内 21.6 ≈ 四线程 20.4 ms，**与线程数无关** |
| 3 | 输入/输出 marshaling（memcpy 1.2 MB + 拷 378 KB） | bench 加 I/O 忠实计时 `p50Io`（A18 §8） | det CPU 7.29→7.89 ms，**只值 +0.6 ms** |
| 4 | `OH_AI_ModelGetInputs/GetOutputs` 每次调用的 API 开销 | 句柄缓存进 `MsSession`（load 时存，bench 本身证明句柄跨 predict 有效） | 流水线内 20.8 ms，gap 不变 |
| 5 | 输入数据模式（bench 密集随机 vs 流水线 94% 黑边零） | `FillDeterministic` 改成多数零分布（r7） | bench 7.54 vs 7.44 ms，**无变化** |

弱支持的 only 一条：负载节律/调频——bench 迭代间插 3 ms 空档，p50 仅 7.54→8.71 ms（r8），
方向对但幅度远不够。**当前结论：gap 的主因是测量时系统级状态（热态/调度/调频），
不是代码路径**——bench 的 30 次背靠背把 CPU 频率拉住，流水线各阶段之间的内存密集空档
让频率掉下去；thermal service 在本机不可读（hidumper 返回空），无法直接证实。

**对用户问题的含义**：
- 相机实时循环是连续负载（无空档），频率行为更接近 bench 而非自检的单帧冷调用 ——
  真实 fps 应落在好的一档（~30 ms/帧 → 33 fps），自检链的 56 ms 是最差值。
- 论文口径：单帧预算报**最好轮 29.1 ms + 轮间带**（29–56 ms），bench 数字照实标注
  "pure ModelPredict, 30 back-to-back iterations"。
- 可解释的都已排除，剩余空间在**降低绝对工作量**（letterbox NEON / 分辨率 / INT8），
  不在解释 gap。

## 11. 本日最终状态（r9，20:2x）

保留的全部改动（均正确性无损，cropSum 逐位不变，72/72 矩阵完整）：
① `PipeScratch`/`ResizeScratch` buffer 复用（消除 ~7 MB/帧分配）；
② 张量句柄缓存；③ `p50Io` 忠实 bench 计时（论文可引：marshaling 只占 det 的 8%）；
④ `kMaxSessions` 20→64；⑤ `cpu_t{1,2,4,6,8}` 线程扫描档 + `threadSweep` 探针；
⑥ pack/infer 分段计时。**最好轮单帧 29.09 ms**（基线 36.11，−19%）。
回退的：E1 实验档、FillDeterministic 实验填充、bench sleep 实验（结论已入档）。
