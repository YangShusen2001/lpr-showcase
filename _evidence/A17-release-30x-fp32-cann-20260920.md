# A17 · release 构建 30× 全矩阵 + NNRT-fp32 结案 + CANN 项目自有模型三连跑通（2026-09-20）

> 设备：MIA-AL00（nova 14 Pro）· 麒麟 8020 · HarmonyOS 6.1.0.135(SP8C00E120R7P5) · API 24
> 设备号 4CY9K25614046328 · 电池 83%（未充电）
> 构建：`com.shusen.lprdemo`，装机 updateTime = 2026-09-18 23:46（晚于 cpp 源码最后修改 23:32，
> 因此包含 CPU 高性能模式修复与 `nnrt_fp32` 档）
> 触发方式：`AUTO_SELFTEST=true` 启动 App → vulkanProbe → listLenses → loadNative → backendProbe
> → e2eMatrix(24 组合) → ncnnSlotProbe → cannProbe → probeAll(30× p50)
> 原始日志：`full-hilog-20260920.txt`（清洗行：`chain-20260920.txt`）

---

## 1. NNRT-fp32 结案（ADR-009 / ADR-015 未完成动作，此前从未测过）

`ms_engine.cpp` 的 `addNnrt()` 曾恒传 `fp16=true`，「det 上 NPU 会翻字符」因此只在 fp16 条件下成立。
本轮 e2eMatrix 跑通 `nnrt_fp32` 档（落点标签 `#fp32`，LANDED 明确区分），结论：

| 模型 × 档 | LANDED | 结果 | 判定 |
|---|---|---|---|
| det × nnrt_fp32 | `NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0#fp32` | `code=苏E05172`，match=0（3/3 全翻） | **fp32 下依然翻字符** |
| det × nnrt（fp16） | 同上（无后缀） | `苏E05172`，match=0 | 与 fp32 一致 |
| det × cpu / gpu / kirin | 均 LANDED=CPU | `苏ED5172`，match=1 | 基线 |
| rec × nnrt_fp32 | `..._v2_0#fp32` | `苏ED5172`，match=1（3/3） | 正确，与 fp16 同 |
| cls × nnrt_fp32 | LANDED=CPU（fp16 变体构不出 kernel，回落） | match=1 | 同前 |

**结论变化**：「det 不能上 NPU」此前必须加「（仅 fp16 条件下测过）」限定；现在 **fp16/fp32 双条件下
均翻字符**，限定可去除 → 「det 上 NPU 会翻字符」成为无条件结论（生产链 `det=CPU/rec=NPU/cls=CPU` 不变，
且理由更充分）。ADR-015 未完成动作 2 可结案。

附带观测：det 的 nnrt_fp32 端到端 19.3–21.2 ms，比 nnrt_fp16 的 24.4–40.7 ms 还快（pipeline 口径，
非纯推理，仅作量级参考）。

⚠️ 口径提示：e2eMatrix 的 24 组合被会话表上限 `kMaxSessions=20` 截断 —— `cls-fp32 × {nnrt_fp32,
nnrt_fp16, gpu, kirin}` 共 4 组 `BUILD-FAIL err=session cap 20 reached`（日志 L95–98 可见）。
这是**上限问题不是模型问题**；det/rec/cls 三角色的 nnrt_fp32 行全部正常。

---

## 2. 30× p50 release 矩阵（probeAll / `LprMatrix` tag，论文级口径）

判 NPU 落点看 `L2asFp16` 列：NPU 行非零、CPU 行为 0.0000。

| 模型 | 输入 (NHWC) | nnrt (NPU) p50 | cpu p50 | gpu (MS Lite) | kirin (MS Lite) |
|---|---|---|---|---|---|
| det-head `y5fu_320x_head_fp32.ms` | 1×320×320×3 | **5.3505 ms**（L2asFp16=48950.6） | 7.6638 ms | 8.1759 ms（落 CPU，fallback=GPU:fp16） | 7.8259 ms（落 CPU） |
| rec `rpv3_mdict_160_r3.ms` | 1×48×160×3 | **3.9883 ms**（L2asFp16=5042.7） | 9.1286 ms | 9.2547 ms（落 CPU） | 9.2411 ms（落 CPU） |
| cls-fp32 `litemodel_cls_96x_r1_fp32.ms` | 1×96×96×3 | **1.0012 ms**（L2asFp16=1.87） | 0.8395 ms | cap-fail | cap-fail |
| cls (fp16 变体) `litemodel_cls_96x_r1.ms` | 1×96×96×3 | 0.8756 ms（**落 CPU**，fp16 kernel 构不出，fallback=NNRT:...） | （行缺失，见下） | 0.9836 ms（落 CPU） | — |

比值（同栈 MS Lite）：det-head NPU/CPU = **1.43×**；rec NPU/CPU = **2.29×**；cls-fp32 NPU/CPU = **0.84×（NPU 更慢）**。

注意：
- 这是 **release 构建**（`-O2 -DNDEBUG`）下的 30× p50，与旧的 `bench30_matrix_20260918.log`
  （debug 构建）**不可并列**，旧数字作废。
- CPU 端已含高性能模式修复（`OH_AI_DeviceInfoSetPerformanceMode(HIGH)`，`ms_engine.cpp:179`），
  但线程数仍硬编码 4（`ms_engine.cpp:176`）→ CPU 仍未被完全榨干，见 §5。
- cls（fp16 变体）的 cpu 行在 probeAll 中缺失（疑似会话管理腾挪失败）；其 CPU 数字沿用 A12 的 0.780 ms。
- GPU/kirin 两列全部 `LANDED=CPU`：这份 `libmindspore_lite.so` 没编 GPU delegate，`KIRIN_NPU`
  设备类型也不可用 —— **与 ADR-004/008 一致，无变化**。

---

## 3. GPU：不是「调用不了」，是「MS Lite 这条路走不通；Vulkan 走得通但不赚」

Vulkan 探针（应用层）：`Maleoon 920C`，Vulkan 1.3.275，integrated，2 条 graphics+compute 队列，
`VK_KHR_shader_float16_int8` 在位，97 个 device extension，verdict=GPU_COMPUTE_FEASIBLE。

ncnnSlotProbe（引擎层，三后端里唯一能落 GPU 的通路）：

| 模型 | ncnn CPU | ncnn Vulkan | vkLayers | 判定 |
|---|---|---|---|---|
| det | total 18.3–22.8 ms | total 35.9–63.3 ms | **218/218** | match=1，GPU 更慢 |
| rec | total 47.2–58.3 ms | total 65.4–93.0 ms | **251/252**（1 层回落 CPU） | match=1，GPU 更慢 |
| cls | total 20.9–35.2 ms | total 65.6–84.8 ms | **78/78** | match=1，GPU 更慢 |

- `clsScores=0.0001|0.9998|0.0001` 与 MS 路径**逐位一致** → GPU 通路数值保真成立。
- rec 只有 251/252 层落 GPU（按 `vkLayers` 口径必须注明「GPU + 1 层 CPU 回落」，不是纯 GPU 数据点）。
- ncnn 侧计时是 3 reps pipeline 口径，**不是 30× p50**；「GPU 慢」的定量结论强度受此限制
  （与 ADR-009 §待办「Vulkan 30 次无插桩正式计时」一致，仍未做）。
- GPU 档阶段明细（rec vulkan）：rec 段 45.5–56.0 ms vs det 段 2.3–4.5 ms → 慢在 rec 的小算子提交开销，
  与 ADR-003 缩放律吻合。

---

## 4. CANN：OMG 缺口已补，项目自有模型三连跑通（**重大更新**）

`cannProbe`：NNRt 设备枚举 2 张全是 `ACCELERATOR`（`HIAI_F` id=8987859593747354028 与
`NPU_ohos.boot.hardware.kirin8020_v2_0`），CANN 版本 `108.631.120.010`，**无 GPU 设备**。

`.om` 离线加载结果（合成填充输入，`argmax` 无意义，只看能不能跑与多快）：

| 模型 | 文件 | compat / build / run | IO | 稳态耗时 |
|---|---|---|---|---|
| codelab SqueezeNet | `probe_hiai_imagenet.om` 2.50 MB | 0 / 0 / 0 | [1×3×227×227]→[1×1000×1×1] | **0.944 ms**（out_sum=1.0005 合法 softmax） |
| LPRNet | `probe_om_lprnet.om` 0.92 MB | 0 / 0 / 0 | [1×3×24×94]→[1×1×68×18] | 1.77–2.11 ms |
| LPRNet(npufix) | `probe_om_lprnet_npufix.om` 0.92 MB | 0 / 0 / 0 | 同上 | **1.37 ms** |
| **检测裸 head** | `probe_om_dethead.om` 1.01 MB | 0 / 0 / 0 | [1×3×320×320]→**3 输出头** [1×45×40×40]/[1×45×20×20]/[1×45×10×10] | 3.90–5.10 ms |
| **识别 rpv3** | `probe_om_rec.om` 5.04 MB | 0 / 0 / 0 | [1×3×48×160]→[1×1×20×78] | 5.19–5.63 ms |
| **分类** | `probe_om_cls.om` 0.84 MB | 0 / 0 / 0 | [1×3×96×96]→[1×3×1×1] | 0.95–0.97 ms |
| 完整检测器 | `probe_om_det.om` 1.95 MB | **compat=1 / build_rc=1 被拒** | — | — |
| 三个项目 `.ms` | y5fu_head/rpv3/cls `.ms` | 全部 compat=1 / build_rc=1 被拒（同一次运行对照） | — | — |

与 MS Lite 同栈对比（**口径不同：CANN 3 次原始采样 vs MS 30× p50，只作量级参考**）：

| 模型 | CANN `.om` | MS Lite→NNRT | 谁快 |
|---|---|---|---|
| det-head | 3.90–5.10 ms | 5.35 ms（30×p50） | 接近 |
| rec rpv3 | 5.19–5.63 ms | **3.99 ms** | **MS 更快** |
| cls | 0.95–0.97 ms | 1.00 ms（fp32 变体） | 接近 |

**结论**：CANN 路线从「只验证过 codelab 客人模型」升级为「项目自有三模型全部 `.om` 化并跑通」，
ADR-013 的「缺口=OMG」已闭合。但升为生产后端仍差三道门槛（ADR-014）：同输入张量 L2 对照、
30× p50、thermal 标注 —— CANN 探针目前只喂合成输入，识别正确性未经真图验证。
两条客观事实没变：完整 det 的 `.om` 上不去（只有裸 head 可以）；`.ms` 仍被 NNRt 直连拒收（格式门槛）。

---

## 5. 对「NPU 调用率不高」的解释（用户问题）

「调用率」由模型结构决定，不由调用方式决定，证据链已在 ADR-014 建立：

- 官方判据（`Cin`/`Cout` 双 16 倍数占比）：cls **19.2%**（10/52）/ rpv3 **55.6%** / det-head **70.6%** / LPRNet 81.2%
- 真机分区：LPRNet `NPU:2,CPU:1`（`ReduceMean` 被拒）、rec `NPU:3,CPU:2`、dethead `NPU:1,CPU:0`、cls `NPU:1,CPU:0`
- 本轮 `L2asFp16` 指纹证明 NPU 真在算（det-head 48950.6 / rec 5042.7），不是假委托
- 小图委托开销摊不薄 → cls 上 NPU 反而慢（0.84×）—— 这是 ADR-003 缩放律的又一次实测

**能试的下一步**（未做）：① CPU 线程扫描（现硬编码 4）；② rec 的 svtr mixer 段算子 NPU 化；
③ rec/cls 的 30× Vulkan 正式计时；④ ≥5 轮 thermal 标注复测。

---

## 6. 本轮复测与旧口径的一致性

- rec nnrt match=1（fp16/fp32 均对）；det nnrt match=0（fp16/fp32 均翻）—— 与 A15 逐项一致
- ncnn-cpu / ncnn-vulkan match=1，clsScores 逐位一致 —— 与 A9 一致
- MS Lite gpu/kirin 编译期/运行期回落 CPU —— 与 ADR-004/008 一致
- NATIVE PIPE 生产链（det=CPU/rec=NNRT/cls=CPU）读出 `苏ED5172` colour=蓝牌 —— 延续 ADR-015 两口径并存登记
  （ cls 模型输出=蓝牌；像素量法=真绿牌，未裁决）

## 7. 数据出处索引

| 数据 | 位置 |
|---|---|
| 本次原始完整日志（58753 行） | `_evidence/full-hilog-20260920.txt` |
| 本次链路清洗行（147 行，全部探针输出） | `_evidence/chain-20260920.txt` |
| NNRT-fp32 e2e 行 | chain 文件 L41–43（det）/ L59–61（rec）/ L77–79（cls 回落） |
| 30× p50 矩阵 | chain 文件 L132–147（`LprMatrix` tag） |
| ncnn Vulkan 三模型 | chain 文件 L102–118 |
| CANN `.om` 十连试 | chain 文件 L119–131 |
| 会话上限失败行 | chain 文件 L95–98（e2e）/ L142–143（probeAll） |
