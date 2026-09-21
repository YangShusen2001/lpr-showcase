#pragma once
#include <cstdint>
#include <string>
#include <vector>

/**
 * ncnn 引擎（A 阶段第二步：GPU 路径的载体）。
 *
 * 为什么要有它：ADR-008 证明 Vulkan compute 在麒麟 8020 上可达（Maleoon 920C，
 * 2 条 G+C 队列）。onnxruntime / MindSpore Lite 都够不到这条路径，
 * 而 ncnn 是唯一成熟的、能在 OHOS 上跑 Vulkan 后端的推理框架。
 *
 * 本文件先做 CPU 后端（交叉编译已验证），Vulkan 后端在下一步
 * （-DNCNN_VULKAN=ON 重编 + use_vulkan_compute=true）打开。
 *
 * 保真纪律：输入预处理**复用 lpr_pipeline 的 LprLetterBox / LprToNchw**，
 * 不另写一份 —— 这样设备侧 ncnn 的输入与 PC 侧 ONNX 参考逐字节同源，
 * 两侧输出可直接对比。
 */

/** 从内存加载 ncnn 模型（param 为文本，bin 为权重；均由 ArkTS 从 rawfile 读入）。
 *  useVulkan=true 时走 Vulkan 后端（需 NCNN_VULKAN=ON 构建的 libncnn.so），
 *  失败（无 libvulkan/无设备）时返回 false 并在 err 里带原因。 */
bool NcnnLoad(const std::vector<char>& param, const std::vector<char>& bin,
              bool useVulkan, std::string& err);

/** 释放当前网络（可重新加载）。 */
void NcnnRelease();

/** 是否已有可用网络。 */
bool NcnnLoaded();

/** GPU 实例状态：ok=1;count=..;name=..;driver=.. 或 ok=0;error=..
 *  create_gpu_instance 成功后可调用，用于自证落点（Maleoon 920C 而非回落 CPU）。 */
std::string NcnnGpuProbe();

/**
 * 用真实 RGBA 图像跑裸 head（可多轮计时）。
 * 返回扁平 kv：ok=1;o0l2=..;o0max=..;o0n=..;o1l2=..;o1max=..;o2l2=..;o2max=..;
 *              warmup=..;repeat=..;p50Ms=..;meanMs=..;error=
 * 三个输出（out0/out1/out2，各 [1,45,H,W]）的 L2/maxAbs 用于与 PC 参考对照。
 */
std::string NcnnRunRgba(const uint8_t* rgba, int w, int h, int warmup, int repeat);

/** 网络结构摘要（层数 / 输入输出名），加载后调用。 */
std::string NcnnInfo();
