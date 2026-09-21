# ADR-011：ncnn 接入 App —— 交叉编译、内存 API 陷阱与保真验证

- **状态**：已接受（真机跑通 + PC 双向对照）
- **日期**：2026-09-17
- **前置**：ADR-008（Vulkan 可达，GPU 路径的唯一载体是 ncnn）
- **后续**：ADR-015 —— ncnn 行的 `effectiveVulkan` 只是选项回显，不构成执行证据；
  矩阵行现已补 `vkLayers=X/Y`（带 Vulkan 实现的层数），X<Y 时该格是「GPU + 部分层
  回落 CPU」，不得当纯 GPU 数据点引用。
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135
- **证据**：`_evidence/ncnn_device_run_20260917.txt`、`_evidence/ncnn_ref_check.json`
- **工具**：`tools/ncnn_ref_check.py`；引擎源码 `cpp/ncnn_engine.{h,cpp}`

---

## 1. 目标与路径

ADR-008 证明 Maleoon 920C 的 Vulkan compute 可达，但 MS Lite 的 GPU 档走 OpenCL
且被编译期判否、ORT-Web 两路皆死 —— **ncnn 是唯一能在 OHOS 上跑 Vulkan 后端的成熟框架**。
本 ADR 完成接入的前半程：**交叉编译 + CPU 后端跑通 + 保真验证**；Vulkan 后端为下一步。

## 2. 交叉编译（OHOS NDK）

用 SDK 自带工具链（`native/build/cmake/ohos.toolchain.cmake` + 自带 cmake/ninja/clang）：

```
cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=<sdk>/native/build/cmake/ohos.toolchain.cmake \
      -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_SHARED_LIB=ON
ninja -j8
```

产物 **`libncnn.so` 5.46 MB**，ELF 校验 `e_machine=0xB7`（AArch64）。
依赖 `libomp.so`（SDK 的 `llvm/lib/aarch64-linux-ohos/`）+ `libc++_shared.so`；
SONAME 是 **`libncnn.so.1`** —— 打包时必须带上这个名字，否则运行时 `NEEDED` 解析失败。

模型转换用 **pnnx**：`y5fu_320x_head.onnx` → `.ncnn.param`（18.6 KB）+ `.ncnn.bin`（894 KB），
**218 层 / 245 blobs / 三输出 out0–out2（各 45 通道）**，FLOPS 386.264M。

## 3. 内存 API 陷阱（本轮最大的坑）

ncnn 的内存加载 API **同名不同义**，返回值语义也各不相同（`net.h:80-111`）：

| API | 语义 | 返回值 | 成功判据 |
|---|---|---|---|
| `load_param_mem(const char*)` | **文本** param，需 NUL 结尾 | `int` | **0 = 成功** |
| `load_param(const unsigned char*)` | **二进制** param（内部走 `load_param_bin`） | `size_t` | 消耗字节数，0 = 失败 |
| `load_model(const unsigned char*)` | 权重，**只引用不拷贝** | `size_t` | 消耗字节数，0 = 失败 |

两个踩坑点：
1. **把文本 param 喂给 `load_param(const unsigned char*)`** —— 它走二进制解析路径，
   必然失败。文本 param 只能用 `load_param_mem`。
2. **误判返回值**：`load_model` 返回的是**消耗字节数**（我们这里正好是 bin 大小 894352），
   按旧 API「0 = 成功」去判，会把成功当失败（首轮真机日志
   `NCNN LOAD FAIL load_model failed ret=894352` 就是这个误判）。
3. **内存必须存活**：`load_model` 不拷贝权重（注释明写 "external memory should be
   retained when used"），因此 param/bin 必须持有副本到网络用完 —— 引擎里用静态
   `g_param/g_bin` 承接，而不是调用方的临时 buffer。

## 4. 真机结果与保真对照

```
NCNN LOAD ok  layers=218  in=in0  out=out0|out1|out2
NCNN RUN  ok=1  o0l2=1076.9788 o0max=23.6007 (72000)
                o1l2=479.9452  o1max=18.1233 (18000)
                o2l2=232.6409  o2max=15.1823 (4500)
                p50 = 10.03 ms（CPU, 4 线程）
```

PC 侧用 **ONNX 同图、同布局（NCHW）、同预处理** 对照（`tools/ncnn_ref_check.py`）：

| 输出 | 设备 ncnn | PC ONNX | 相对差 |
|---|---|---|---|
| out0 | 1076.9788 / 23.6007 | 1077.1619 / 23.6179 | **0.017%** |
| out1 | 479.9452 / 18.1233 | 479.8680 / 18.1462 | 0.016% |
| out2 | 232.6409 / 15.1823 | 232.7599 / 15.2070 | 0.051% |

**结论：ncnn 移植保真**（L2 相对差 ≤0.05%、maxAbs 绝对差 ≤0.025），
与「同框架换后端」的差异同量级（MS Lite NPU vs CPU：0.08% / 0.0232）。

**一个必须记住的推论**：ncnn 与 ONNX 的差（maxAbs ~0.02）经 anchor（≤433）放大后
同样是 ~10px 级 —— 与 ADR-009 的 NPU 漂移同性质。**ncnn 的检测输出也不能直接喂给
单应矫正**（若接进流水线，须先在真机上验证 `苏ED5172` 是否仍然逐字符一致）。

## 5. 工程提示

- 库与模型布置：`entry/libs/arm64-v8a/{libncnn.so, libncnn.so.1, libomp.so}`（随 hap 打包），
  模型进 `rawfile/models/`（复制后 `attrib -r`，否则 restool 报 11204003）。
- 预处理**复用 `LprLetterBox` / `LprToNchw`**，不另写一份 —— 这是两侧可比的前提。
- ncnn 头文件要两个目录：源码 `ncnn/src` + 构建目录 `ncnn_build_ohos/src`（生成头）。
- pnnx 安装会因网络中断失败，pip 需带 `--retries 5 --timeout 60`。

## 6. 下一步

- [ ] `-DNCNN_VULKAN=ON` 重编（需宿主 glslang 参与 shader→SPIR-V 编译，已 clone 到位），
      引擎开 `use_vulkan_compute=true`，真机验证 **Vulkan 后端是否真的落到 Maleoon 920C**
      （ncnn 的 `net.opt.use_vulkan_compute` + `get_gpu_device()`），并测 p50。
- [ ] 若 Vulkan 后端可用：对比 CPU(10.03ms) / Vulkan / MS Lite NPU(5.04ms) 三条路径的
      延迟与数值，形成端侧三后端矩阵（论文 §VI 的核心表格）。
