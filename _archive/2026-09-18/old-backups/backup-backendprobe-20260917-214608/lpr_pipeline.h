#ifndef LPR_PIPELINE_H
#define LPR_PIPELINE_H

/**
 * Native HyperLPR3 pipeline — C++ transcription of assets/js/pipeline.js.
 *
 * Why this file exists: the Web demo ran the whole pipeline in JS on top of
 * onnxruntime-web, which on Kirin 8020 can only ever use the CPU (ADR-004 §6.1).
 * The NPU is reachable *only* through MindSpore Lite, i.e. only from native code.
 * To put the recogniser on the NPU while the detector stays on the CPU, the whole
 * pipeline has to live here — the NAPI surface of ms_engine.cpp alone only runs
 * one model at a time.
 *
 * Fidelity contract: this is a *port*, not a reimplementation. Every numeric step
 * mirrors pipeline.js, which in turn mirrors the upstream Python/C++ algorithm:
 *   - resizeLinear   -> cv2.resize INTER_LINEAR, OpenCV's two-pass fixed point
 *                       (horizontal pass kept unshifted, one rounding in vertical)
 *   - warpPerspectiveCubic -> cv2.warpPerspective INTER_CUBIC, a = -0.75, BORDER_REPLICATE
 *   - channel order  -> detector wants RGB, recogniser/classifier want BGR
 *   - Uint8ClampedArray stores round-half-to-even, so ClampU8 uses nearbyint()
 * The acceptance test is: same image in, byte-identical crop checksum and
 * character-identical plate code out, versus the browser path.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "ms_engine.h"

/** RGBA8 image. The only image format the pipeline accepts. */
struct RgbaImage {
  std::vector<uint8_t> data;  // width * height * 4, RGBA
  int width = 0;
  int height = 0;

  bool Valid() const {
    return width > 0 && height > 0 &&
           data.size() == static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
  }
};

/** One plate: detection box + rectified crop + recognition + classification. */
struct PlateResult {
  int rect[4] = {0, 0, 0, 0};  // x1, y1, x2, y2 in source-image coordinates
  float detScore = 0;
  int layer = 0;  // 0 = single, 1 = double (DOUBLE_LAYER)

  std::string code;
  float recConf = 0;
  std::vector<std::string> chars;
  std::vector<float> charProbs;

  int cropH = 0;
  int cropW = 0;
  /** RGB-only sum of the rectified crop — the cross-implementation fingerprint. */
  long long cropSum = 0;

  float cls[3] = {0, 0, 0};  // yellow, blue, green

  float tDetectMs = 0;
  float tRectifyMs = 0;
  float tRecogMs = 0;
  float tClsMs = 0;
};

/** The three sessions, each built on whatever backend it actually landed on. */
struct LprSessions {
  MsSession* det = nullptr;  // y5fu_320x_sim      -> CPU (NPU rejected, ADR-004 §4.1)
  MsSession* rec = nullptr;  // rpv3_mdict_160_r3  -> NPU (2.2-3.0x)
  MsSession* cls = nullptr;  // litemodel_cls_96x  -> CPU (NPU slower on small maps)
  int detSize = 320;
  float confThresh = 0.25f;
  float iouThresh = 0.5f;
};

/**
 * End-to-end: letterbox -> detect -> NMS -> rectify -> recognise -> classify.
 *
 * Returns false only on a hard failure (bad image, missing session, inference
 * error); "no plate found" is a successful run with an empty `out`.
 * `out` is sorted by detector score, best first.
 */
bool LprRunPipeline(const RgbaImage& img, const LprSessions& s,
                    std::vector<PlateResult>& out, std::string& err);

// ---------------------------------------------------------------- test surface
// Exposed so the port can be diffed against pipeline.js function by function
// instead of only end to end. See tools/harmony/ for the diff harness.

/** Letterboxed square image plus the geometry needed to map boxes back. */
struct LetterBoxed {
  RgbaImage img;
  float r = 1;
  int left = 0;
  int top = 0;
};

LetterBoxed LprLetterBox(const RgbaImage& src, int size);

/** NCHW float tensor; `swapRB` true = detector (BGR->RGB), false = recogniser. */
std::vector<float> LprToNchw(const RgbaImage& img, bool swapRB);

/** NHWC float tensor — MindSpore Lite's CPU backend reports this layout. */
std::vector<float> LprToNhwc(const RgbaImage& img, bool swapRB);

/** 14-column rows [x1,y1,x2,y2,score,kp0x..kp3y,layer], padding undone. */
std::vector<std::vector<float>> LprDecodeDetections(const std::vector<float>& raw, int rows,
                                                    float confThresh, float iouThresh,
                                                    float r, int left, int top);

/**
 * Bare-head decode (ADR-006 §5): the three rank-4 head tensors [1,45,40,40] /
 * [1,45,20,20] / [1,45,10,10] (onnx output order) become the same [6300,15] row
 * blob the ORIGINAL in-graph decode produced. Verified element-wise against the
 * original model by tools/verify_head_decode.py (maxAbsDiff=0.000061, MATCH).
 * Anchors are hardcoded (graph constants 1005/1118/1231); grids are generated.
 */
std::vector<float> LprDecodeBareHead(const std::vector<std::vector<float>>& heads);

/** Rectify the quad to an axis-aligned crop (rotating 90 deg if portrait). */
bool LprRotateCrop(const RgbaImage& src, const int marks[4][2], RgbaImage& out);

/**
 * Recognition input: aspect-preserving resize, [-1,1], zero right-padding.
 *
 * `nhwc` selects the memory layout and it is NOT cosmetic: MindSpore Lite reports
 * [1,48,160,3] (pixel-interleaved) for this model while ONNX declares [1,3,48,160]
 * (planar). Feeding planar bytes to a NHWC tensor interleaves the three channels
 * into nonsense and the CTC head then decodes garbage.
 */
std::vector<float> LprEncodePlate(const RgbaImage& crop, int imgH, int imgW,
                                  int limitedMaxWidth, int limitedMinWidth, int& outW,
                                  bool nhwc);

/** Classification input: square resize, [0,1], BGR. Same `nhwc` caveat. */
std::vector<float> LprEncodeClassify(const RgbaImage& crop, int size, bool nhwc);

/** Character set, index 0 = CTC blank. 44 entries. */
const std::vector<std::string>& LprToken();

/** CTC greedy decode over one [T] index row with its per-step probabilities. */
void LprCtcGreedy(const std::vector<int>& idx, const std::vector<float>& prob,
                  std::string& code, float& conf,
                  std::vector<std::string>& chars, std::vector<float>& probs);

#endif  // LPR_PIPELINE_H
