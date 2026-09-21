/**
 * 原生 LPR 引擎（libentry.so）的 NAPI 接口声明。
 * 实现在 entry/src/main/cpp/{napi_init,ms_engine}.cpp。
 */
/** backend: "auto" | "nnrt" | "gpu" | "kirin" | "cpu" */
export const loadModel: (name: string, bytes: ArrayBuffer, backend: string) => string;
export const run: (id: number, input: Float32Array) => Float32Array;
/** 稳态计时 + 输出校验和（CPU-diff 协议）。返回 JSON。 */
export const bench: (id: number, warmup: number, repeat: number) => string;
export const listNnrtDevices: () => string;

/**
 * 原生完整流水线（检测 → NMS → 单应矫正 → 识别 → CTC → 分类），全在 C++ 里跑。
 *
 * 三个会话可以落在不同后端 —— 这正是必须在原生侧重建流水线的原因：
 * 检测模型上不了 NPU（ADR-004 §4.1），识别模型能上（2.2–3.0×），
 * 而 onnxruntime-web 根本够不到 NPU（ADR-004 §6.1）。
 *
 * 返回扁平 kv 串：
 *   ok=1;count=N;totalMs=...;error=
 *   p0=<code>,<detScore>,<recConf>,<layer>,<x1|x2|x3|x4>,<cropH|cropW>,
 *      <cls0|cls1|cls2>,<char|...>,<prob|...>,<tDet|tRect|tRec|tCls>,<cropSum>;
 */
export const pipeline: (
  rgba: ArrayBuffer, width: number, height: number,
  detId: number, recId: number, clsId: number) => string;

/**
 * GPU 终审探针（vulkan_probe.cpp）：dlopen Vulkan loader → vkCreateInstance →
 * vkEnumeratePhysicalDevices → queue families。返回 JSON，verdict 字段四选一。
 * 只打日志用，ArkTS 侧不解析（JSON 里有引号/逗号，KvSanitize 语法装不下）。
 */
export const vulkanProbe: () => string;

/**
 * ncnn 引擎（A 阶段第二步）：param/bin 由 ArkTS 从 rawfile 读入后传字节。
 * ncnnRun 用真实 RGBA 图跑裸 head，返回三输出 L2/maxAbs 与耗时（kv 串）。
 */
export const ncnnLoad: (param: ArrayBuffer, bin: ArrayBuffer, useVulkan?: boolean) => string;
export const ncnnRun: (rgba: ArrayBuffer, width: number, height: number, repeat: number) => string;
export const ncnnRelease: () => string;
