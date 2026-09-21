# ADR-008：GPU 终审 —— Vulkan compute 可达（Maleoon 920C 枚举成功）

- **状态**：已接受（真机应用级探针实测）
- **日期**：2026-09-17
- **前置**：ADR-004 §4.1（GPU 两路判否）、ADR-005 §8（MS Lite gpu 档 LANDED=CPU）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135
- **证据**：`_evidence/vulkan_probe_app_20260917.json`、`_evidence/vulkan_probe_hilog_20260917.txt`
- **工具**：`lpr-harmony/LprDemo/entry/src/main/cpp/vulkan_probe.{h,cpp}`（本次新增，NAPI 导出 `vulkanProbe`）

---

## 1. 问题

GPU 问题此前已判死三条路（ADR-004/005），只余 Vulkan compute 未验证：

| 路径 | 结论 | 根因 |
|---|---|---|
| MindSpore Lite `gpu`/`kirin` 档 | 死 | OpenCL 路线，麒麟 8020 不暴露设备，编译期 `GPU is not supported` |
| ORT-Web WebGL EP | 死 | onnxruntime v1.29.0 上游已删（#29716/#31683） |
| ORT-Web WebGPU | 死 | ArkWeb `requestAdapter()` 返回 NULL |
| **原生 Vulkan compute** | **本文验证** | — |

## 2. 分层探针（先零风险后应用级）

**第一层（hdc shell，零风险）**：
- `param get` → **`const.SystemCapability.Graphic.Vulkan = true`**（HarmonyOS 无 getprop，用 param）
- `ls /vendor/etc/vulkan/` → **Permission denied**（≠ No such file → 目录存在）
- `/vendor/lib64`、`/vendor/etc` 整体不可列（SELinux），文件级探针到此为止

**第二层（应用级，本 ADR 核心）**：给 LprDemo 原生层加 `vulkanProbe` NAPI：
`dlopen` loader → `vkCreateInstance` → `vkEnumeratePhysicalDevices` → 读
`VkPhysicalDeviceProperties` + `VkQueueFamilyProperties`。不依赖 SDK 的 Vulkan 头
（ABI 手写最小子集，Vulkan ABI 自 1.0 冻结），任何失败防御性返回 JSON、不崩溃。

## 3. 实测结果（18:27，AUTO_SELFTEST 无人值守路径）

```
loader      = libvulkan.so  dlopen OK（OHOS so namespace 放行第三方 App）
instance    = VK_KHR_surface / VK_OHOS_surface / VK_KHR_portability_enumeration / …
GPU         = Maleoon 920C   vendorID 0x19e5（海思） deviceID 0x20011000 integrated
apiVersion  = 1.3.275        driverVersion 29.368.428
queues      = 1 family，graphicsComputeQueues=2（G+C 混合队列）
device ext  = 97 个，含 VK_KHR_shader_float16_int8 / 16bit_storage /
              shader_float_controls / spirv_1_4 / scalar_block_layout
verdict     = GPU_COMPUTE_FEASIBLE
```

**判读**：
1. **GPU 真实存在且可达**——Maleoon 920C（实测名，与坊间传的 910 不符，以实测为准）。
2. **compute 路径成立**：2 条 G+C 队列，Vulkan 1.3；compute 不需要 swapchain，
   `VK_OHOS_surface` 只在想把结果画出来时才用。
3. **FP16 shader 支持在位**：`VK_KHR_shader_float16_int8` + `16bit_storage` ——
   与 NPU/MS Lite 的 fp16 口径对齐，后续模型可直接 fp16 推理。
4. 同一轮自检 sanity：镜头 count=2；流水线输出 `苏ED5172` **与 rpv3 基线一致**
   （该串非人工真值，见 ADR-015；total 128ms：det 92.5 / rect 13.1 / rec 9.9 / cls 4.8
   —— det 预处理嫌疑再+1）。

## 4. 意义与下一步

GPU 从「物理不可能」改判「**可达、待跑通**」。落地路径按工程量排序：

| 方案 | 内容 | 工程量 |
|---|---|---|
| **A. ncnn-Vulkan 移植** | ncnn 有成熟 Vulkan backend；onnx → pnnx → ncnn；先跑 det 裸 head（纯 conv，最合适）或 cls | 中（交叉编译 ncnn for OHOS + Vulkan） |
| B. 自写最小 Vulkan compute | 手写 SPIR-V 卷积——不现实，仅作方案 A 失败后的说明项 | 大 |
| C. 只用 GPU 做预处理/后处理 | letterbox/decode 上 compute shader | 小，但不算「模型跑在 GPU」 |

**选定方案 A**：ncnn-Vulkan。验收标准：任一车牌模型（首选 y5fu 裸 head）
在 ncnn Vulkan 后端跑通且数值与 CPU/ONNX 参考一致，出真机 p50。

## 5. 工程坑（本轮）

- **musl 下 dlopen 在 libc，不存在 `-ldl.z`**：CMakeLists 里显式链接
  `libdl.z.so` 会报 `ld.lld: error: unable to find library -ldl.z`——删掉即可。
- hilog 单条有截断：探针 JSON 按 ~400 字符切块输出（ArkTS 侧循环 append）。
- 构建指纹复用：signed.hap 时间戳必须新于 unsigned.hap，否则是签名收尾崩
  （NODE_OPTIONS 陷阱，见 build.sh 注释）。
