/**
 * NAPI surface for the native LPR engine.
 *
 *   loadModel(name, bytes: ArrayBuffer, backend: string) -> string  (flat kv)
 *   run(id: number, input: Float32Array)                 -> Float32Array | null
 *   bench(id: number, warmup, repeat)                    -> string  (flat kv)
 *   listNnrtDevices()                                    -> string  (JSON array)
 *   pipeline(rgba, width, height, detId, recId, clsId)   -> string  (flat kv)
 *
 * `pipeline` is the whole HyperLPR3 chain in native code (lpr_pipeline.cpp): it is
 * what makes "detector on CPU + recogniser on NPU" expressible at all, since the
 * NPU is unreachable from onnxruntime-web (ADR-004 §6.1).
 *
 * `backend` is "auto" | "nnrt" | "gpu" | "kirin" | "cpu"; see ms_engine.h.
 *
 * Deliberately tiny: the whole compute pipeline lives in C++ so the JS side only
 * ever ships the image in and a few KB of JSON out. The ArkWeb bridge can then
 * carry a small payload instead of multi-megabyte tensors.
 */
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include <hilog/log.h>
#include <napi/native_api.h>

#include "ms_engine.h"
#include "lpr_pipeline.h"
#include "vulkan_probe.h"
#include "ncnn_engine.h"
#include <gpu.h>

#define LPR_TAG "LprNative"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, 0xD001, LPR_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, 0xD001, LPR_TAG, __VA_ARGS__)

static std::vector<MsSession*> g_sessions;

// ---------------------------------------------------------------- helpers
static std::string JsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

static std::string ShapeToJson(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); i++) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string((long long)s[i]);
  }
  out += "]";
  return out;
}

static const char* FormatName(OH_AI_Format f) {
  switch (f) {
    case OH_AI_FORMAT_NCHW: return "NCHW";
    case OH_AI_FORMAT_NHWC: return "NHWC";
    case OH_AI_FORMAT_NHWC4: return "NHWC4";
    case OH_AI_FORMAT_NC4HW4: return "NC4HW4";
    case OH_AI_FORMAT_NC: return "NC";
    case OH_AI_FORMAT_NC4: return "NC4";
    case OH_AI_FORMAT_HW: return "HW";
    case OH_AI_FORMAT_HW4: return "HW4";
    default: return "OTHER";
  }
}

static napi_value MakeString(napi_env env, const std::string& s) {
  napi_value v = nullptr;
  napi_create_string_utf8(env, s.c_str(), s.size(), &v);
  return v;
}

/**
 * ArkTS forbids `any`/`unknown`, so `JSON.parse()` on the JS side will not compile
 * under the strict ArkTS linter. Every NAPI call therefore returns a flat
 * `key=value;key=value;` string instead — trivially parseable with split().
 * `;` and `=` inside a value are folded to `,` so the grammar stays unambiguous.
 */
static std::string KvSanitize(const std::string& in) {
  std::string out = in;
  for (char& c : out) {
    if (c == ';' || c == '=' || c == '\n' || c == '\r') {
      c = ',';
    }
  }
  return out;
}

static napi_value MakeNull(napi_env env) {
  napi_value v = nullptr;
  napi_get_null(env, &v);
  return v;
}

// ---------------------------------------------------------------- loadModel
static napi_value LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

  if (argc < 2) {
    return MakeString(env, "ok=0;error=loadModel needs (name, bytes, backend)");
  }

  char nameBuf[128] = {0};
  size_t nameLen = 0;
  napi_get_value_string_utf8(env, args[0], nameBuf, sizeof(nameBuf) - 1, &nameLen);
  std::string name(nameBuf, nameLen);

  void* data = nullptr;
  size_t dataLen = 0;
  napi_status   st = napi_get_arraybuffer_info(env, args[1], &data, &dataLen);
  if (st != napi_ok || data == nullptr || dataLen == 0) {
    return MakeString(env, "ok=0;error=model ArrayBuffer is empty");
  }

  std::string backend = "auto";
  if (argc >= 3) {
    char beBuf[64] = {0};
    size_t beLen = 0;
    napi_get_value_string_utf8(env, args[2], beBuf, sizeof(beBuf) - 1, &beLen);
    if (beLen > 0) {
      backend.assign(beBuf, beLen);
    }
  }

  std::vector<char> bytes(reinterpret_cast<char*>(data),
                          reinterpret_cast<char*>(data) + dataLen);

  std::string err;
  MsSession* s = MsLoad(bytes, backend, err);
  if (s == nullptr) {
    LOGE("loadModel(%{public}s) failed: %{public}s", name.c_str(), err.c_str());
    return MakeString(env, "ok=0;backend=" + KvSanitize(backend) + ";error=" + KvSanitize(err));
  }

  int id = (int)g_sessions.size();
  g_sessions.push_back(s);

  std::string kv = "ok=1;id=" + std::to_string(id) +
                   ";name=" + KvSanitize(name) +
                   ";backend=" + KvSanitize(s->backend) +
                   ";attemptLog=" + KvSanitize(s->attemptLog) +
                   ";inputName=" + KvSanitize(s->inputName) +
                   ";outputName=" + KvSanitize(s->outputName) +
                   ";inputShape=" + ShapeToJson(s->inputShape) +
                   ";outputShape=" + ShapeToJson(s->outputShape) +
                   ";inputFormat=" + std::string(FormatName(s->inputFormat)) +
                   ";outputFormat=" + std::string(FormatName(s->outputFormat)) +
                   ";inputElems=" + std::to_string((long long)s->inputElems) +
                   ";outputElems=" + std::to_string((long long)s->outputElems) +
                   ";error=";
  LOGI("loadModel ok: %{public}s", kv.c_str());
  return MakeString(env, kv);
}

// ---------------------------------------------------------------- run
static napi_value RunModel(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2] = {nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

  if (argc < 2) {
    LOGE("run needs (id, Float32Array)");
    return MakeNull(env);
  }

  int32_t id = -1;
  napi_get_value_int32(env, args[0], &id);
  if (id < 0 || id >= (int32_t)g_sessions.size()) {
    LOGE("run: bad session id %{public}d", id);
    return MakeNull(env);
  }

  napi_typedarray_type type;
  size_t length = 0;
  void* data = nullptr;
  napi_value ab = nullptr;
  size_t byteOffset = 0;
  napi_status st = napi_get_typedarray_info(env, args[1], &type, &length, &data, &ab, &byteOffset);
  if (st != napi_ok || data == nullptr || type != napi_float32_array) {
    LOGE("run: input must be a Float32Array (status=%{public}d type=%{public}d)", (int)st, (int)type);
    return MakeNull(env);
  }

  MsSession* s = g_sessions[id];
  if (length != s->inputElems) {
    LOGE("run: input length %{public}zu != expected %{public}zu", length, s->inputElems);
    return MakeNull(env);
  }

  std::vector<float> out;
  std::string err;
  if (!MsRun(s, reinterpret_cast<const float*>(data), out, err)) {
    LOGE("run failed: %{public}s", err.c_str());
    return MakeNull(env);
  }

  void* outData = nullptr;
  napi_value outAb = nullptr;
  napi_create_arraybuffer(env, out.size() * sizeof(float), &outData, &outAb);
  if (outData != nullptr && !out.empty()) {
    std::memcpy(outData, out.data(), out.size() * sizeof(float));
  }
  napi_value outArr = nullptr;
  napi_create_typedarray(env, napi_float32_array, out.size(), outAb, 0, &outArr);
  return outArr;
}

// ---------------------------------------------------------------- bench
static std::string Num(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.4f", v);
  return std::string(buf);
}

static napi_value BenchModel(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

  int32_t id = -1;
  napi_get_value_int32(env, args[0], &id);
  if (id < 0 || id >= (int32_t)g_sessions.size()) {
    return MakeString(env, "ok=0;error=bad session id");
  }
  int32_t warmup = 10, repeat = 50;
  if (argc >= 2) napi_get_value_int32(env, args[1], &warmup);
  if (argc >= 3) napi_get_value_int32(env, args[2], &repeat);

  MsBench b = MsBenchRun(g_sessions[id], warmup, repeat);
  std::string kv = "ok=" + std::string(b.ok ? "1" : "0") +
                   ";backend=" + KvSanitize(b.backend) +
                   ";warmup=" + std::to_string(b.warmup) +
                   ";repeat=" + std::to_string(b.repeat) +
                   ";meanMs=" + Num(b.mean) +
                   ";p50Ms=" + Num(b.p50) +
                   ";p95Ms=" + Num(b.p95) +
                   ";minMs=" + Num(b.minMs) +
                   ";maxMs=" + Num(b.maxMs) +
                   ";checksum=" + Num(b.checksum) +
                   ";checksumAsFp16=" + Num(b.checksumAsFp16) +
                   ";maxAbs=" + Num(b.maxAbs) +
                   ";outputDtype=" + std::to_string(b.outputDtype) +
                   ";outputElems=" + std::to_string((long long)b.outputElems) +
                   ";error=" + KvSanitize(b.error);
  LOGI("bench: %{public}s", kv.c_str());
  return MakeString(env, kv);
}

// ---------------------------------------------------------------- listNnrtDevices
static napi_value ListNnrt(napi_env env, napi_callback_info info) {
  std::vector<std::string> names = MsNnrtCandidates();
  std::string json = "[";
  for (size_t i = 0; i < names.size(); i++) {
    if (i > 0) {
      json += ",";
    }
    json += "\"" + JsonEscape(names[i]) + "\"";
  }
  json += "]";
  return MakeString(env, json);
}

// ---------------------------------------------------------------- pipeline
/** Steady-clock milliseconds; lpr_pipeline.cpp keeps its own copy in an anon ns. */
static double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/**
 * Plate codes and character lists are re-emitted into a flat `;`-separated string,
 * so any delimiter appearing inside a value would corrupt the grammar. The charset
 * cannot produce them, but a wrong-session mix-up could, hence the scrub.
 */
static std::string Scrub(const std::string& in) {
  std::string out = in;
  for (char& c : out) {
    if (c == ';' || c == '=' || c == ',' || c == '|') {
      c = '_';
    }
  }
  return out;
}

/**
 * pipeline(rgba: ArrayBuffer, width, height, detId, recId, clsId) -> flat kv string
 *
 * Runs the whole HyperLPR3 chain in native code: the sessions may sit on different
 * backends (detector on CPU because the NPU rejects it, recogniser on the NPU), and
 * that combination is the entire point of having the pipeline here instead of in
 * the WebView — onnxruntime-web cannot reach the NPU at all.
 *
 * Layout of the returned string (values never contain ';' or '='):
 *   ok=1;count=N;totalMs=...;err=
 *   p0=<code>,<detScore>,<recConf>,<layer>,<x1|x2...>,<cropH|cropW>,
 *      <cls0|cls1|cls2>,<char|char|...>,<prob|prob|...>,
 *      <tDet|tRect|tRec|tCls>,<rect x1,y1,x2,y2>
 *   p1=...
 */
static napi_value PipelineRun(napi_env env, napi_callback_info info) {
  size_t argc = 6;
  napi_value args[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 6) {
    return MakeString(env, "ok=0;count=0;totalMs=0;error=pipeline needs "
                           "(rgba, width, height, detId, recId, clsId)");
  }

  void* data = nullptr;
  size_t dataLen = 0;
  if (napi_get_arraybuffer_info(env, args[0], &data, &dataLen) != napi_ok ||
      data == nullptr || dataLen == 0) {
    return MakeString(env, "ok=0;count=0;totalMs=0;error=rgba ArrayBuffer is empty");
  }
  int32_t w = 0, h = 0, detId = -1, recId = -1, clsId = -1;
  napi_get_value_int32(env, args[1], &w);
  napi_get_value_int32(env, args[2], &h);
  napi_get_value_int32(env, args[3], &detId);
  napi_get_value_int32(env, args[4], &recId);
  napi_get_value_int32(env, args[5], &clsId);

  const int32_t ids[3] = {detId, recId, clsId};
  for (int i = 0; i < 3; i++) {
    if (ids[i] < 0 || ids[i] >= (int32_t)g_sessions.size()) {
      return MakeString(env, "ok=0;count=0;totalMs=0;error=bad session id");
    }
  }

  RgbaImage img;
  img.width = w;
  img.height = h;
  img.data.assign(reinterpret_cast<uint8_t*>(data),
                  reinterpret_cast<uint8_t*>(data) + dataLen);
  if (!img.Valid()) {
    return MakeString(env, "ok=0;count=0;totalMs=0;error=rgba size != w*h*4");
  }

  LprSessions s;
  s.det = g_sessions[detId];
  s.rec = g_sessions[recId];
  s.cls = g_sessions[clsId];

  std::vector<PlateResult> plates;
  std::string err;
  const double t0 = NowMs();
  if (!LprRunPipeline(img, s, plates, err)) {
    LOGE("pipeline failed: %{public}s", err.c_str());
    return MakeString(env, "ok=0;count=0;totalMs=0;error=" + KvSanitize(err));
  }
  const double totalMs = NowMs() - t0;

  // Every field is `key=value;` — the trailing ';' after `error=` is load-bearing:
  // without it the first plate record fuses into error's value (`error=p0=...`),
  // and the ArkTS side then reads `p0` as undefined.
  std::string kv = "ok=1;count=" + std::to_string(plates.size()) +
                   ";totalMs=" + Num(totalMs) +
                   ";error=;";
  for (size_t i = 0; i < plates.size(); i++) {
    const PlateResult& p = plates[i];
    std::string v = Scrub(p.code) + "," +
                    Num(p.detScore) + "," +
                    Num(p.recConf) + "," +
                    std::to_string(p.layer) + "," +
                    std::to_string(p.rect[0]) + "|" + std::to_string(p.rect[1]) + "|" +
                    std::to_string(p.rect[2]) + "|" + std::to_string(p.rect[3]) + "," +
                    std::to_string(p.cropH) + "|" + std::to_string(p.cropW) + "," +
                    Num(p.cls[0]) + "|" + Num(p.cls[1]) + "|" + Num(p.cls[2]) + ",";
    for (size_t k = 0; k < p.chars.size(); k++) {
      if (k > 0) {
        v += "|";
      }
      v += Scrub(p.chars[k]);
    }
    v += ",";
    for (size_t k = 0; k < p.charProbs.size(); k++) {
      if (k > 0) {
        v += "|";
      }
      v += Num(p.charProbs[k]);
    }
    v += "," + Num(p.tDetectMs) + "|" + Num(p.tRectifyMs) + "|" +
         Num(p.tRecogMs) + "|" + Num(p.tClsMs) +
         "," + std::to_string(p.cropSum);
    kv += "p" + std::to_string(i) + "=" + v + ";";
  }
  LOGI("pipeline: %{public}s", kv.c_str());
  return MakeString(env, kv);
}

// ---------------------------------------------------------------- vulkanProbe
/**
 * GPU 终审探针（ADR-008）：dlopen Vulkan loader → 枚举物理设备 → 找 COMPUTE 队列。
 * 返回 JSON；见 vulkan_probe.h。这里不套 KvSanitize —— JSON 是给 PC 侧看的，
 * ArkTS 侧只做切块打日志，不解析。
 */
static napi_value VulkanProbe(napi_env env, napi_callback_info info) {
  return MakeString(env, VulkanProbeJson());
}

// ---------------------------------------------------------------- ncnn（A 阶段第二步）
/**
 * ncnn 三件套：load / run / release。
 *
 * 走 ncnn 而不是 MS Lite 的原因：GPU（Vulkan compute）在麒麟 8020 上可达
 * （ADR-008：Maleoon 920C、2 条 G+C 队列），但 MS Lite 的 GPU 档走 OpenCL
 * 且被编译期判否；ncnn 是唯一能在 OHOS 上跑 Vulkan 后端的成熟框架。
 *
 * 接口刻意收窄：param/bin 由 ArkTS 从 rawfile 读入后传字节（C++ 侧不碰资源系统），
 * 图像预处理复用 lpr_pipeline 的实现，保证与 PC 参考同源。
 */
static napi_value NcnnLoadFn(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 2) {
    return MakeString(env, "ok=0;error=ncnnLoad needs (param, bin)");
  }
  void* pData = nullptr;
  size_t pLen = 0;
  void* bData = nullptr;
  size_t bLen = 0;
  if (napi_get_arraybuffer_info(env, args[0], &pData, &pLen) != napi_ok ||
      napi_get_arraybuffer_info(env, args[1], &bData, &bLen) != napi_ok ||
      pData == nullptr || bData == nullptr || pLen == 0 || bLen == 0) {
    return MakeString(env, "ok=0;error=empty param/bin buffer");
  }
  std::vector<char> param(reinterpret_cast<char*>(pData),
                          reinterpret_cast<char*>(pData) + pLen);
  std::vector<char> bin(reinterpret_cast<char*>(bData),
                        reinterpret_cast<char*>(bData) + bLen);
  std::string err;
  bool useVulkan = false;
  if (argc >= 3 && napi_get_value_bool(env, args[2], &useVulkan) != napi_ok) {
    return MakeString(env, "ok=0;error=useVulkan must be boolean");
  }
  if (!NcnnLoad(param, bin, useVulkan, err)) {
    LOGE("ncnnLoad failed: %{public}s", err.c_str());
    return MakeString(env, "ok=0;error=" + KvSanitize(err));
  }
  const std::string kv = NcnnInfo();
  LOGI("ncnnLoad: %{public}s", kv.c_str());
  return MakeString(env, kv);
}

static napi_value NcnnRunFn(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4] = {nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 4) {
    return MakeString(env, "ok=0;error=ncnnRun needs (rgba, w, h, repeat)");
  }
  void* data = nullptr;
  size_t dataLen = 0;
  if (napi_get_arraybuffer_info(env, args[0], &data, &dataLen) != napi_ok ||
      data == nullptr || dataLen == 0) {
    return MakeString(env, "ok=0;error=rgba buffer empty");
  }
  int32_t w = 0, h = 0, repeat = 5;
  napi_get_value_int32(env, args[1], &w);
  napi_get_value_int32(env, args[2], &h);
  napi_get_value_int32(env, args[3], &repeat);
  const std::string kv =
      NcnnRunRgba(reinterpret_cast<const uint8_t*>(data), w, h, 1, repeat);
  LOGI("ncnnRun: %{public}s", kv.c_str());
  return MakeString(env, kv);
}

static napi_value NcnnReleaseFn(napi_env env, napi_callback_info info) {
  NcnnRelease();
  return MakeString(env, "ok=1");
}

// ---------------------------------------------------------------- module
EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
  const int gpuInit = ncnn::create_gpu_instance();
  LOGI("NCNN INIT create=%{public}d gpu_count=%{public}d", gpuInit, ncnn::get_gpu_count());
  napi_property_descriptor desc[] = {
      {"loadModel", nullptr, LoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"run", nullptr, RunModel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"bench", nullptr, BenchModel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"listNnrtDevices", nullptr, ListNnrt, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"pipeline", nullptr, PipelineRun, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"vulkanProbe", nullptr, VulkanProbe, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnLoad", nullptr, NcnnLoadFn, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnRun", nullptr, NcnnRunFn, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnRelease", nullptr, NcnnReleaseFn, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
  return exports;
}
EXTERN_C_END

static napi_module lprModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterLprModule(void) {
  napi_module_register(&lprModule);
}
