#include "ncnn_engine.h"

#include <net.h>
#include <gpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "lpr_pipeline.h"

namespace {

ncnn::Net g_net;
bool g_loaded = false;
bool g_use_vulkan = false;
/**
 * 权重内存必须存活到网络用完（ncnn 的内存版 load_model **只引用不拷贝**，
 * 见 net.h:106-108）。所以这里持有副本，而不是用调用方的临时 buffer。
 * std::vector 的堆内存由 operator new 分配，满足 32-bit 对齐要求。
 */
std::vector<unsigned char> g_param;
std::vector<unsigned char> g_bin;
/** GPU 实例全局只创建一次；ncnn 要求 create_gpu_instance 先于任何 Vulkan Net。 */
bool g_gpu_instance = false;

std::string Num(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.4f", v);
  return std::string(buf);
}

/** L2 + maxAbs over a ncnn Mat, read as raw floats (pack layout does not matter). */
void Stats(const ncnn::Mat& m, double& l2, double& maxAbs, size_t& n) {
  const float* p = reinterpret_cast<const float*>(m.data);
  n = static_cast<size_t>(m.total()) * m.elempack;
  double acc = 0.0;
  maxAbs = 0.0;
  for (size_t i = 0; i < n; i++) {
    const double v = p[i];
    acc += v * v;
    const double a = std::fabs(v);
    if (a > maxAbs) {
      maxAbs = a;
    }
  }
  l2 = std::sqrt(acc);
}

}  // namespace

bool NcnnLoad(const std::vector<char>& param, const std::vector<char>& bin,
              bool useVulkan, std::string& err) {
  if (param.empty() || bin.empty()) {
    err = "empty param or bin";
    return false;
  }
  g_net.clear();
  g_loaded = false;
  g_use_vulkan = false;

  g_net.opt.num_threads = 4;
  g_net.opt.use_fp16_packed = false;
  g_net.opt.use_fp16_storage = false;
  g_net.opt.use_fp16_arithmetic = false;

  if (useVulkan) {
    // SimpleVK：内部 dlopen("libvulkan.so")，无需链接（ADR-008 已验证可达 Maleoon 920C）
    if (!g_gpu_instance) {
      const int gi = ncnn::create_gpu_instance();
      if (gi != 0) {
        err = "create_gpu_instance failed ret=" + std::to_string(gi);
        return false;
      }
      g_gpu_instance = true;
    }
    if (ncnn::get_gpu_count() <= 0) {
      err = "no vulkan device";
      return false;
    }
    g_net.opt.use_vulkan_compute = true;
    g_net.set_vulkan_device(ncnn::get_default_gpu_index());
  } else {
    g_net.opt.use_vulkan_compute = false;
  }

  // 两个内存 API 的语义**不一样**（net.h:80-111，net.cpp:2636-2648）：
  //   load_param_mem(const char*)      —— 文本 param，要求 NUL 结尾，返回 0 表示成功
  //   load_param(const unsigned char*) —— **二进制** param（内部走 load_param_bin），
  //                                       返回消耗字节数；拿文本喂它必失败
  //   load_model(const unsigned char*) —— 权重，只引用不拷贝，返回消耗字节数，0 才是失败
  g_param.assign(param.begin(), param.end());
  g_param.push_back(0);  // load_param_mem 需要 NUL 结尾
  g_bin.assign(bin.begin(), bin.end());
  const int rp = g_net.load_param_mem(reinterpret_cast<const char*>(g_param.data()));
  if (rp != 0) {
    err = "load_param_mem ret=" + std::to_string(rp) +
          " (param len=" + std::to_string(param.size()) + ")";
    return false;
  }
  const size_t rm = g_net.load_model(g_bin.data());
  if (rm == 0) {
    err = "load_model consumed 0 bytes (bin len=" + std::to_string(bin.size()) + ")";
    return false;
  }
  g_loaded = true;
  g_use_vulkan = useVulkan;
  return true;
}

void NcnnRelease() {
  g_net.clear();
  g_loaded = false;
  g_use_vulkan = false;
  g_param.clear();
  g_bin.clear();
  if (g_gpu_instance) {
    ncnn::destroy_gpu_instance();
    g_gpu_instance = false;
  }
}

bool NcnnLoaded() {
  return g_loaded;
}

std::string NcnnGpuProbe() {
  if (!g_gpu_instance) {
    return "ok=0;error=gpu instance not created";
  }
  const int count = ncnn::get_gpu_count();
  if (count <= 0) {
    return "ok=0;error=no vulkan device;count=" + std::to_string(count);
  }
  const ncnn::GpuInfo& info = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
  return "ok=1;count=" + std::to_string(count) +
         ";name=" + info.device_name() +
         ";driver=" + info.driver_name() +
         ";api=" + std::to_string(info.api_version()) +
         ";fp16=" + std::to_string(info.support_fp16_packed() ? 1 : 0) +
         std::to_string(info.support_fp16_storage() ? 1 : 0) +
         std::to_string(info.support_fp16_arithmetic() ? 1 : 0);
}

std::string NcnnInfo() {
  if (!g_loaded) {
    return "ok=0;error=no network";
  }
  std::string in, out;
  for (size_t i = 0; i < g_net.input_names().size(); i++) {
    if (i > 0) in += "|";
    in += g_net.input_names()[i];
  }
  for (size_t i = 0; i < g_net.output_names().size(); i++) {
    if (i > 0) out += "|";
    out += g_net.output_names()[i];
  }
  return "ok=1;layers=" + std::to_string(g_net.layers().size()) +
         ";inputs=" + in + ";outputs=" + out +
         ";requestedVulkan=" + std::to_string(g_use_vulkan ? 1 : 0) +
         ";effectiveVulkan=" + std::to_string(g_net.opt.use_vulkan_compute ? 1 : 0) +
         ";gpuProbe=" + (g_use_vulkan ? NcnnGpuProbe() : "not requested");
}

std::string NcnnRunRgba(const uint8_t* rgba, int w, int h, int warmup, int repeat) {
  if (!g_loaded) {
    return "ok=0;error=no network";
  }
  if (rgba == nullptr || w <= 0 || h <= 0) {
    return "ok=0;error=bad image";
  }
  if (warmup < 0) warmup = 0;
  if (repeat < 1) repeat = 1;

  // 与 PC 参考同源的预处理：letterbox → NCHW(RGB, /255)
  RgbaImage img;
  img.width = w;
  img.height = h;
  img.data.assign(rgba, rgba + static_cast<size_t>(w) * h * 4);
  if (!img.Valid()) {
    return "ok=0;error=rgba size mismatch";
  }
  const LetterBoxed lb = LprLetterBox(img, 320);
  const std::vector<float> nchw = LprToNchw(lb.img, true);

  ncnn::Mat in(320, 320, 3);
  for (int c = 0; c < 3; c++) {
    ncnn::Mat ch = in.channel(c);
    const float* src = nchw.data() + static_cast<size_t>(c) * 320 * 320;
    for (int y = 0; y < 320; y++) {
      float* row = ch.row(y);
      std::memcpy(row, src + static_cast<size_t>(y) * 320, 320 * sizeof(float));
    }
  }

  double l2[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
  size_t nn[3] = {0, 0, 0};
  std::vector<double> times;
  times.reserve(repeat);

  for (int it = 0; it < warmup + repeat; it++) {
    ncnn::Extractor ex = g_net.create_extractor();
    auto t0 = std::chrono::steady_clock::now();
    ex.input("in0", in);
    ncnn::Mat o0, o1, o2;
    if (ex.extract("out0", o0) != 0 || ex.extract("out1", o1) != 0 ||
        ex.extract("out2", o2) != 0) {
      return "ok=0;error=extract failed at iter " + std::to_string(it);
    }
    auto t1 = std::chrono::steady_clock::now();
    if (it >= warmup) {
      times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      Stats(o0, l2[0], mx[0], nn[0]);
      Stats(o1, l2[1], mx[1], nn[1]);
      Stats(o2, l2[2], mx[2], nn[2]);
    }
  }

  double sum = 0;
  for (double t : times) sum += t;
  std::vector<double> sorted(times);
  std::sort(sorted.begin(), sorted.end());
  const double p50 = sorted.empty() ? 0 : sorted[sorted.size() / 2];
  const double mean = times.empty() ? 0 : sum / static_cast<double>(times.size());

  std::string kv = "ok=1";
  for (int k = 0; k < 3; k++) {
    kv += ";o" + std::to_string(k) + "l2=" + Num(l2[k]) +
          ";o" + std::to_string(k) + "max=" + Num(mx[k]) +
          ";o" + std::to_string(k) + "n=" + std::to_string(nn[k]);
  }
  kv += ";warmup=" + std::to_string(warmup) +
        ";repeat=" + std::to_string(repeat) +
        ";p50Ms=" + Num(p50) +
        ";meanMs=" + Num(mean) +
        ";error=";
  return kv;
}
