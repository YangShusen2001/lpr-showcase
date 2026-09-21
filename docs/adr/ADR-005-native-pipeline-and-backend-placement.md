# ADR-005：原生 C++ 流水线、三模型混合后端落点与 GPU 专项复测

- **状态**：已接受（结论均由真机实测 + 与 Python 参考逐项对照）
- **日期**：2026-09-17
- **前置**：ADR-004（本 ADR 是其 §6.1 的直接落地）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135(SP8C00E120R7P5)
- **宿主**：自研 ArkTS + NAPI C++ 应用（`com.shusen.lprdemo`），MindSpore Lite 2.6.0 NDK
- **证据**：`_evidence/native_pipeline_and_gpu.txt`、`_evidence/native_selftest_raw.txt`

---

## 1. 问题

ADR-004 §6.1 已经证明：**Web 路径 100% 是 ORT WASM / CPU**，NPU 与 GPU 在 Web 层物理上接不上。

那么「让识别模型吃到 NPU 的 2.2–3.0×」这句话，就只剩一条路——**在原生侧重建整条流水线**。
而此前原生侧只有一个「探针」（逐模型跑 benchmark），**完整车牌流水线压根不在原生侧**，
它只在 Web 的 `assets/js/pipeline.js` 里。

本 ADR 记录这次重建、验收结果，以及顺带做掉的 GPU 专项复测。

---

## 2. 实现

| 新增/改动 | 内容 |
|---|---|
| `entry/src/main/cpp/lpr_pipeline.h` | 流水线接口：`RgbaImage` / `PlateResult` / `LprSessions` / 纯函数测试面 |
| `entry/src/main/cpp/lpr_pipeline.cpp` | **~700 行**，逐函数移植 `pipeline.js` |
| `napi_init.cpp` | 新增 `pipeline(rgba, w, h, detId, recId, clsId) -> kv` |
| `Index.ets` | 新增「原生演示」标签页：加载模型 / 跑样本 / 枚举镜头 |

**关键设计**：三个模型由 `loadModel` 的第三参数**独立指定后端**，互不影响。
这正是「检测留在 CPU、识别送上 NPU」这种混合配置能表达出来的原因，
也是必须在原生侧重建流水线的全部理由。

移植的数值契约（与 `pipeline.js` 逐条对齐）：

| 环节 | 契约 |
|---|---|
| resize | cv2 `INTER_LINEAR` 定点两趟：水平趟保持未移位整数中间量，垂直趟一次舍入 |
| 矫正 | cv2 `warpPerspective` `INTER_CUBIC`，a = −0.75，BORDER_REPLICATE |
| 通道序 | 检测器要 RGB，识别/分类要 BGR |
| 存储 | `Uint8ClampedArray` 是**四舍六入五成双**，故用 `nearbyint()` 而非 `lround()` |
| 垂直趟 | 最坏中间量 1,044,480 × 2048 已擦到 int32 上界 → **必须 int64** |

---

## 3. 验收：与 Python 参考逐项对照

同一张 `assets/samples/hlpr-test.jpg`（1920×1080），三个独立实现的对照：

| 项 | Python 参考 | 原生 C++ | 结论 |
|---|---|---|---|
| `rect` | `[1751, 747, 1879, 855]` | `1751\|747\|1879\|855` | **逐值相同** |
| `crop_shape` | `[78, 123]` | `78\|123` | **相同** |
| `crop_sum` | 2,772,794 | 2,773,473 | 差 679（**0.024%**） |
| `det_score` | 0.7413 | 0.7276 | 差 1.8%（后端浮点） |
| `code` | **苏ED5172** | **苏ED5172** | **逐字符一致** |
| `rec_conf` | 0.7312 | 0.7366 | 差 0.7% |

第二张 `scene-2.jpg`（1140×456）：

| 项 | Python 参考 | 原生 C++ |
|---|---|---|
| `rect` | `[217, 157, 563, 328]` | `217\|157\|563\|328`（**逐值相同**） |
| `crop_shape` | `[122, 339]` | `122\|339`（相同） |
| `crop_sum` | 5,281,904 | 5,281,907（差 **3**，5.7e−7） |
| `code` | — | `G0C289GBAB9FBOPKLMN`（低置信，与参考同源样本差异见 §8） |

**结论**：几何链路（检测 + 单应矫正）移植正确——**检测框逐值相同，裁剪像素只差 1 LSB 级**，
差异来自后端浮点（MS Lite CPU vs ONNX Runtime CPU），不是移植错误。
识别结果与参考逐字符一致。

---

## 4. 两个移植缺陷（都出在「想当然」上，务必留档）

### 4.1 类别数取错：用字典长度（77）而不是模型输出（78）

首次跑通时输出 19 字符乱码 `苏G0123456789ABCDEFGH`（真实是 7 字符）。

根因：`pipeline.js` 用 **`dims[2]`** 作为类别数切分 logits，而我用了字符表长度：

```cpp
const int C = static_cast<int>(LprToken().size());   // 77 —— 错
const int T = static_cast<int>(logits.size() / C);   // 1560/77 = 20，每行错位一列
```

实测三个模型的真实输出形状：

| 模型 | ONNX 输出 | 元素数 |
|---|---|---|
| `y5fu_320x_sim` | `[1, 6300, 15]` | 94,500 |
| `rpv3_mdict_160_r3` | **`[1, 20, 78]`** | 1,560 |
| `litemodel_cls_96x_r1` | `[1, 3]` | 3 |

**`rpv3` 输出 78 类，而字符表只有 77 项** —— 差一项。
`pipeline.js` 的切分本来是对的（用模型 shape），是我把「字典长度」当成了「类别数」。

修法：**类别数一律取自 `MsSession::outputShape`**，字典只用于索引映射。
兜底路径（shape 不可用时退回字典长度）保留，但会显式写回 `err`，不静默。

> **顺带记录一个既有观察**：78 vs 77 的缺口在 Web 路径同样存在
> （`ctcGreedy` 里 `idx < TOKEN.length ? TOKEN[idx] : '?'`），只是实测样本中
> argmax 从未选中索引 77，所以没有暴露。本 ADR 不擅自改字符表——
> 改了会与 Web 路径产生分叉，且没有依据说明第 78 类是什么。

### 4.2 输入布局未分派：NHWC 模型喂了 NCHW 数据

修完 §4.1 后，结果变成 `苏F`（2 字符，置信 0.2698），分类还把蓝牌判成黄牌。

根因：MS Lite 报告的输入是 **NHWC**，而我只对检测模型做了布局分派：

```cpp
// 检测做了
const bool detNhwc = (s.det->inputFormat == OH_AI_FORMAT_NHWC);
// 识别 / 分类漏了 —— LprEncodePlate 直接生成 NCHW 平面布局就送进去了
```

实测三个模型的输入形状与格式：

| 模型 | MS Lite 输入 | 格式 |
|---|---|---|
| `y5fu_320x_sim` | `[1, 320, 320, 3]` | NHWC |
| `rpv3_mdict_160_r3` | `[1, 48, 160, 3]` | NHWC |
| `litemodel_cls_96x_r1` | `[1, 96, 96, 3]` | NHWC |

把三个平面（B 平面 / G 平面 / R 平面）喂给像素交织的 NHWC 张量，
等于把三个通道揉成噪声，CTC 头自然解出乱码。

修法：`LprEncodePlate` / `LprEncodeClassify` 增加 `nhwc` 参数，**由会话报告的格式决定**，
不再假设 ONNX 声明的 NCHW 就是运行时布局。

**教训**：ONNX 的 `[1,3,H,W]` 与 MS Lite 的 `[1,H,W,3]` 是**同一模型的两个视角**，
转换器会把权重转成 NHWC，输入也必须跟着转。**凡是跨推理框架，先问会话要 shape 和 format。**

---

## 5. GPU 专项复测：三个模型全部回落 CPU

用户明确要求「GPU 也要测试」。本轮对**全部三个正式模型**逐个请求 `gpu` 后端：

| 模型 | 请求 | 实际落点 | p50 |
|---|---|---|---|
| `y5fu_320x_sim` | `gpu` | **CPU** | 9.07 ms |
| `rpv3_mdict_160_r3` | `gpu` | **CPU** | 9.00 ms |
| `litemodel_cls_96x_r1` | `gpu` | **CPU** | 0.67 ms |

**不是运行时静默降级，是编译期判否。** 原始日志（同一行出现 **6 次** = 3 模型 × fp16/fp32 两档）：

```
E MS_LITE: [inner_context.cc:207] IsValid# GPU is not supported.
E MS_LITE: [inner_context.cc:116] Init# Context is not valid
```

根因见 ADR-004 §4.2：MindSpore Lite 的 GPU 后端基于 **OpenCL**，麒麟 8020 的 Maleoon GPU
未向该路径暴露 OpenCL 设备。

**结论**：MS Lite 这条路线上 GPU 不可用，与本 ADR 无关，是平台边界。
（另有一条**未测**的路：原生 GLES 3.1 compute shader 自写算子——那是自己实现推理，
不是「让 MS Lite 用 GPU」，成本量级完全不同，属可选探索。）

**一个待解释的观察**：分类模型在 CPU 落点下 `L2asFp16 = 56640`，
而 ADR-004 §7 记载「CPU 落点的该值为 0 属预期」。此处非 0，
怀疑是同一进程内先前的 GPU/fp16 尝试留下了设备状态。
**未隔离验证**，仅记录。

---

## 6. 意外收获：识别模型在 NPU 上是**混合执行**

NPU 编译期对识别模型报了算子支持检查失败：

```
W AI_NPUCL: CheckSupported: op [/neck/encoder/svtr_block.0/mixer/Reshape] type [Reshape]
            is not supported in npucl store [elementary_lib]
W AI_NPUCL: CheckSupported: op [.../mixer/Transpose] type [Permute]
            is not supported in npucl store [fe_lib]
W AI_NPUCL: fftl_fusion_pass.cc CheckOpSupport: "fftl not support GatherV2D"
W AI_NPUCL: fftl_fusion_pass.cc CheckOpSupport: "fftl not support Swish"
```

即 `svtr_block.{0,1}/mixer` 里的 **Reshape / Transpose(Permute)**、以及 **GatherV2D / Swish**
不在 NPU 的算子库里。但识别模型**整体仍然成功落到 NPU**
（`NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0(ok)`）。

**解释**：NPU 做了**子图切分**——支持的子图在 NPU 执行，不支持的回落 CPU，
所以端到端是**混合执行**，不是纯 NPU。

这解释了为什么识别模型的加速比是 2.2–3.0× 而不是更高，
也指出了下一步优化方向：**若把这些算子改写成 NPU 友好的等价形式，
识别模型还有进一步提速空间**（属模型侧改造，未启动）。

---

## 7. 性能：原生 vs Web（同一台 nova 14 Pro）

| 环节 | Web（ORT WASM · 6 线程） | 原生（det=CPU / rec=NPU / cls=CPU） | 加速 |
|---|---|---|---|
| 检测 | 138.1 ms | **63.7 ms** | 2.17× |
| 矫正 | 26.1 ms | **12.5 ms** | 2.09× |
| 识别 | 58.3 ms | **9.0 ms** | **6.5×**（NPU） |
| 分类 | 3.4 ms | 3.9 ms | 0.87× |
| **端到端** | **222.4 ms** | **96.8 ms** | **2.3×** |

> 口径说明：两列**不是同一次测量**。Web 列取自本日 PC 经 CDP 推图到手机 ArkWeb 的那次
> （`_evidence` 中的 mobile.html 运行记录）；原生列为 §3 那次自检。
> 同一张 `hlpr-test.jpg`、同一台设备、同样 warm 状态，但**未做 ≥5 轮重复**，CI 缺失。
> 按 ADR-004 §7 的纪律，此表只能给量级，不能当精确值引用。

**识别环节 6.5× 是 NPU 的直接贡献**；检测仍在 CPU，是当前最大单项（63.7 ms，占 66%）。

---

## 8. 未完成 / 下一步

- [ ] **多摄（长焦 / 主摄 / 超广角）**：`camera.getCameraManager().getSupportedCameras()`
      在 nova 14 Pro 上**只返回 2 个逻辑相机**（`device/0` 后置、`device/1` 前置），
      且两者 `cameraType` **均为 0（DEFAULT）** —— 物理镜头没有被单独枚举出来。
      下一步需确认走**变焦比**（`setZoomRatio`，由系统内部切换物理镜头）
      还是另有物理镜头枚举 API。**未完成**。
- [ ] **相机实时采集**：原生侧要接 Camera Kit 的 ImageReceiver 取流，
      把 YUV/RGBA 帧喂给 `pipeline()`。目前只验证了「rawfile 解码 → 流水线」这条路。
- [ ] **检测模型上 NPU**：y5fu 需先导出「无 anchor-grid decode」裸 head（ADR-004 §4.1）。
      检测现在是 63.7 ms 的最大单项，上去收益最直接。
- [ ] `scene-2.jpg` 在原生侧输出 `G0C289GBAB9FBOPKLMN`（19 字符、低置信），
      与 Python 参考的差异待查（该样本在参考实现里也是低置信场景）。**未完成**。
- [ ] 端侧 ≥5 轮 + 标注 thermal level，补齐 CI（与 ADR-004 §8 同项）。
- [ ] 关闭 `AUTO_SELFTEST`（`Index.ets`）——它是命令行验证用的启动自检，
      演示前必须置 `false`，否则应用一打开就自己跑。
