# Cross-Language Port Fidelity and On-Device Characterization of an End-to-End License Plate Recognition Pipeline

**项目名称（中文）**：基于 HyperLPR3 的端到端车牌识别与跨语言移植保真验证

**Shusen**
*Independent Researcher*
Email: (to be filled)

---

## Abstract

Deploying an end-to-end license plate recognition (LPR) pipeline in a browser or on a mobile NPU is
usually reported as a single number: latency. This paper argues that the more informative — and more
easily falsified — metric is **numerical fidelity across the port**, and we present a reproducible
methodology for establishing it. We port the three-stage HyperLPR3 pipeline (detector, CRNN-CTC
recogniser, colour classifier; 14.2 MB of FP32 ONNX) from its reference C++/Python implementation to
dependency-free JavaScript executing under ONNX Runtime Web. We show that OpenCV's `INTER_LINEAR`
resize is a **fixed-point** two-pass filter (11-bit coefficients, unscaled horizontal intermediate,
single `>> 22` merge), and that a floating-point transcription introduces a systematic +1 LSB bias.
We further show that the reference implementation truncates detection keypoints with `.astype(int)`
*before* measuring the rectification quad, and that omitting this step shifts the crop by one pixel
and flips the recognition output (`MU4158` → `H468`). After correcting both, **8 of 8 test samples
produce recognition strings identical to the Python reference**; four full scenes are additionally
verified to be bit-exact at integer resize scales. On non-integer scales a residual of at most one
quantisation level remains, affecting 3.83 % of channels in the worst measured case
(11753 / 307200) and never changing a recognition output; a 135-scale sweep shows it is governed by
the vertical resize scale rather than by the fractional coordinate, and we bound it rather than
claim bit-equality. We then separate two claims that a fidelity study invites conflating: on 1000
third-party, human-labelled plate crops the production recogniser reaches **90.6 % exact match**
(the competing LPRNet, whose own test split this is, reaches 88.8 %; McNemar p = 0.1788, so no
replacement is warranted), and the same measurement **refuted an acceptance string that this work
had supplied itself** — the seven-character reference used as a red line throughout the on-device
study is the recogniser's own output on a 2×-yaw-compressed plate, not ground truth; the correct
reading has eight characters. The on-device conclusions survive because they are differential, but
their wording had to change: agreement is not correctness. We complement
the port study with an on-device characterisation of the HiSilicon Kirin 8020 NPU via MindSpore Lite
2.6.0 → NNRT on a nova 14 Pro, reporting measured P50 latencies for six backbone networks. The
speedup ranges monotonically from **13.11×** (ResNet-50, 6.02 ms NPU vs 78.90 ms CPU) down to
**0.98×** (fully INT8-quantised MobileNetV2), demonstrating that small graphs gain nothing from NPU
offload because fixed delegation overhead cannot be amortised. We also document a runtime defect in
which FP16 output buffers are declared as FP32, producing 1e21-magnitude garbage that is corrected
by inserting a trailing `Cast` node (relative error 7.78e18 % → 0.015 %). We then close the loop on
this pipeline's own three models, measured under the same 30-iteration protocol: the recogniser gains
**2.27×** on the NPU (4.02 ms vs 9.14 ms), the classifier is **0.74×** — 35 % *slower* than the CPU —
and the detector is 1.50× faster yet unusable, because a 0.077 % output perturbation is amplified by
`logit × anchor` into a ≈10 px keypoint shift that flips a character. Two of those three results are
negative, and all three follow from one quantity: the ratio of fixed delegate overhead to arithmetic.
Finally, we show that the vendor's own offline-model toolchain **accepts its `IMOD` format and
rejects MindSpore Lite's `.ms` format** while the same three models run on the same NPU through both
paths — evidence that the accelerator, not the framework, is the binding constraint.

**Keywords** — license plate recognition, cross-language porting, numerical fidelity, ONNX Runtime
Web, NPU characterisation, MindSpore Lite, CANN, Neural Network Runtime, heterogeneous inference,
edge inference

---

## I. Introduction

License plate recognition is a mature task, and its algorithmic content — detect the plate,
rectify its perspective, read its characters — is well understood. What remains genuinely
under-reported is the engineering question that appears the moment a working research prototype is
moved to a new runtime: **how do you know the new implementation computes the same thing?**

The usual answer is "the outputs matched on our test set." For a discrete output such as a decoded
character string, this is a weak claim. Two implementations can agree on 100 % of a small test set
while differing systematically in the intermediate tensors, and then diverge on the first input that
lands near a decision boundary. The failure mode is silent: no exception is raised, no warning is
logged.

This paper takes the opposite approach. We treat the port itself as the object of study, and we
demand evidence at the level of **intermediate tensors**, not just final strings. The contributions
are:

1. **A port-fidelity methodology** (§IV) that compares every intermediate stage — letterbox
   parameters, input tensor head/mid segments, raw detector rows, boxes, keypoints, crop dimensions,
   crop checksums, decoded strings, confidences, class scores — between a Python reference and a
   headless-browser implementation, driven automatically.
2. **A reverse-engineered fixed-point model of `cv2.resize`** (§IV-A) explaining a systematic +1 LSB
   bias, together with the impulse-response evidence used to derive it.
3. **Identification of a latent one-pixel rectification defect** (§IV-B) caused by transcribing a
   keypoint truncation step out of order.
4. **A quantified residual bound** (§IV-C) for non-integer resize scales, with the honest conclusion
   that bit-equality is not achievable there and that the residual is output-neutral.
5. **An on-device characterisation of the Kirin 8020 NPU** (§V) covering latency scaling, a
   runtime dtype-contract defect, and — under the same protocol — **this pipeline's own three
   models**, whose speedups (2.27×, 1.50×, 0.74×) are explained by a single overhead-to-arithmetic
   ratio and include two negative results.
6. **A cross-toolchain check** (§V-F) in which the same three models are re-converted with the
   vendor's own offline model generator and executed through the standard runtime API. The toolchain
   gate turns out to be a *model format*, not a permission; both compilers reject the same construct;
   and the second path independently reproduces the first path's conclusions.
7. **A ground-truth audit of our own acceptance criterion** (§IV-F): a third-party labelled set
   yields an exact-match rate for the production recogniser (90.6 %, with a paired test against the
   strongest replacement candidate), quantifies its error *profile* rather than just its rate
   (84/94 errors are same-length substitutions; length errors 1.0 % vs the candidate's 7.0 %), and
   falsifies the single-image string we had used as a red line — together with the control
   experiment (n = 200) that establishes which geometric distortions can and cannot create a
   character. This is the one part of the study that could have been performed only *after*
   committing to an external ground truth, and it is reported as a correction to our own earlier
   claims rather than as a success.

---

## II. Related Work

**LPR pipelines.** Modern LPR systems are typically two-stage: a detector localises the plate, and a
sequence recogniser reads it. The CRNN architecture combined with the connectionist temporal
classification (CTC) loss [1] removed the need for character-level segmentation and remains the
dominant recognition head. Perspective rectification between the two stages, as introduced for scene
text by Shi *et al.* [2], is required because plates are rarely fronto-parallel. HyperLPR3 [3], on
which this work is based, packages this exact decomposition as three ONNX graphs and is released
under Apache-2.0, which is what makes an open, reproducible port study possible at all.

**Browser-side inference.** ONNX Runtime Web [4] compiles ONNX graphs to WebAssembly, optionally
using SIMD and, when `SharedArrayBuffer` is available, multiple threads. Multithreaded execution
requires cross-origin isolation (COOP/COEP) response headers, which conventional static hosting does
not provide. We therefore select the thread count adaptively — `isolated ? min(cores, N) : 1` — so the
public build stays single-threaded on any static server, while a host that *can* set the headers
(a HarmonyOS shell injecting them from its own request interceptor, or the bundled development
server) unlocks every core; Section IV-D reports the measured end-to-end effect.

**Cross-language numerical fidelity.** The general problem of reproducing reference numerical
behaviour in a different language is well known in scientific computing, where it is usually
addressed by tolerances. Image-processing pipelines are a special case: operations such as resize and
warp are specified loosely enough that libraries may legitimately differ, yet their outputs feed
learned models that were trained on one specific implementation's behaviour. We are not aware of a
prior systematic treatment of this coupling for LPR, which is the gap this paper addresses.

**On-device NPU characterisation.** Published NPU benchmarks overwhelmingly report accelerator
speedup as a single headline figure. Our measurements (§V) show that this is misleading: on the Kirin
8020 the same accelerator delivers a 13.11× speedup on ResNet-50 and a 0.98× *slowdown* on fully
quantised MobileNetV2. Reporting only the favourable end of that range would misrepresent the
hardware.

---

## III. System Overview

### A. Pipeline decomposition

The system is a strict four-stage cascade. Each stage has a typed, inspectable input and output,
which is what makes stage-by-stage verification possible.

| Stage | Implementation | Input | Output |
|---|---|---|---|
| 1. Detection | `y5fu_320x_sim.onnx` (2.23 MB) | 320×320×3 RGB, normalised | N × (box, 4 keypoints, objectness) |
| 2. Rectification | analytic (no model) | source image + 4 integer keypoints | fronto-parallel plate strip |
| 3. Recognition | `rpv3_mdict_160_r3.onnx` (9.78 MB) | 48×160×3 BGR | 74-class logits over T timesteps |
| 4. Classification | `litemodel_cls_96x_r1.onnx` (1.53 MB) | 96×96×3 BGR | 3 colour scores + layer index |

Total model payload is 14,206,477 bytes (13.5 MiB). The recogniser's output alphabet comprises 74
tokens: a CTC blank, an apostrophe, ten digits, 24 Latin letters (excluding I and O, which are
visually ambiguous with 1 and 0), and 38 Chinese province/authority glyphs.

### B. Detection

The detector is a YOLOv5-family single-class head. The source image is resized with aspect ratio
preserved and letterboxed to 320×320, then converted to NCHW with RGB channel order and scaled to
[0, 1]. A single forward pass yields candidate rows carrying a box, four corner keypoints, an
objectness score and a layer class. Rows below an objectness threshold of 0.25 are discarded, and
non-maximum suppression with an IoU threshold of 0.5 removes duplicates. Box coordinates are then
inverse-transformed from letterbox space back to source-image space.

The layer dimension is a genuine piece of domain knowledge rather than a modelling artefact: Chinese
double-layer plates (used for some commercial vehicle classes) stack two rows of characters, and the
detector predicts this directly. When the layer index equals 1, the rectified strip is split at 40 %
of its height, the two halves are recognised independently, and the resulting strings are
concatenated.

### C. Rectification

The four detected keypoints are ordered into a consistent spatial sequence and used to solve a 3×3
homography. The strip is resampled with bicubic interpolation (kernel parameter *a* = −0.75, matching
OpenCV's `INTER_CUBIC`) to a fronto-parallel view.

This stage contains no learned parameters, which makes it easy to underestimate. As §IV-B shows, it
is also where a one-pixel error silently changes the final answer.

### D. Recognition

The rectified strip is resized so that its height is exactly 48 px and its width is scaled
proportionally, clamped to at most 160 px and padded on the right. The CRNN produces logits over
*T* timesteps × 74 classes. Per timestep we take the argmax and its probability, then apply CTC
greedy decoding: collapse consecutive duplicates and drop blanks. The mean of the retained
per-timestep probabilities is reported as the recognition confidence, and the per-character
probabilities are retained for display.

### E. Classification

A 96×96 classification head produces scores over {yellow, blue, green}, which in the Chinese
registration scheme correspond to large commercial vehicles, standard passenger vehicles, and
new-energy vehicles respectively. This stage does not influence the recognised string and is
therefore excluded from the fidelity-critical path, although it is verified for completeness.

### F. Architectural choice: a DOM-free algorithm layer

The entire algorithm layer is written against plain `{data, width, height}` buffers and imports
nothing. The ONNX Runtime sessions are injected rather than imported. Two consequences follow, both
of which were prerequisites for this study:

- the same module executes in the browser *and* under Node, so it can be exercised outside a DOM;
- the module can be driven by a test harness that dumps intermediate values without any browser
  instrumentation in the algorithm code itself.

---

## IV. Cross-Language Port Fidelity

### A. `cv2.resize` is a fixed-point filter, not a floating-point one

The first discrepancy appeared immediately and was large: the JavaScript detector produced a
different plate string than the Python reference on a 320×320 image whose letterbox scale is exactly
1.0 — that is, on an input where no resampling of the *letterbox* should occur at all.

Isolating the resize step showed that OpenCV's `INTER_LINEAR` implements a **fixed-point** two-pass
filter, not the textbook floating-point bilinear formula:

```
COEF_BITS  = 11
COEF_SCALE = 1 << COEF_BITS                  # 2048

# per destination index, fractional source position fx
c0 = round((1 - fx) * COEF_SCALE)
c1 = round(     fx  * COEF_SCALE)

# horizontal pass: integer intermediate, deliberately NOT shifted
tmp = src[s0] * c0 + src[s1] * c1

# vertical pass: single merge rounding
out = (tmp[r0] * c0 + tmp[r1] * c1 + (1 << 21)) >> 22
```

Two details matter and neither is documented in the OpenCV public interface. First, the horizontal
pass is left **unscaled**, so precision is carried into the vertical pass; shifting after each pass
introduces a measurable error. Second, rounding happens **once**, at the end, by adding 2²¹ before a
22-bit right shift.

We confirmed the coefficient precision by driving the filter with unit impulses. Observed responses
of 247.03 → 247 and 7.97 → 8 are consistent with 11-bit coefficient quantisation and a single
terminal rounding, and inconsistent with both float arithmetic and per-pass rounding.

The practical consequence is a systematic bias, and we measured its direction rather than assuming
it. Comparing the shipped port, `cv2.resize`, and an exact double-precision bilinear resize at four
probed destination sizes, OpenCV sits *below* the exact value everywhere (mean −0.049 to −0.081 LSB,
10–13 % of pixels off by a quantisation level) — the signature of fixed-point coefficient
quantisation — while the shipped port carries the horizontal pass at full precision and lands near
the exact value (−0.000 to +0.063 LSB). The port is therefore the *more accurate* of the two, and
its divergence from OpenCV is systematically positive: on a 320×320×3 tensor this shifts thousands
of channel values by one quantisation level — enough to move detector scores and, in the worst case,
to change the selected box.

### B. Keypoints must be truncated before the quad is measured

The second defect was subtler and more damaging. The reference implementation applies
`.astype(int)` to the four keypoints, and only then measures the quad's edge lengths to decide the
rectified output size:

```python
marks = row[5:13].astype(int)      # truncate FIRST
w, h  = measure_quad(marks)        # then size the output
```

A direct transcription that keeps the keypoints as floats and truncates later produces a rectified
window displaced by up to one pixel. On a large plate this is harmless. On a small plate — where the
rectified strip may be only 13 × 22 px — a one-pixel displacement changes the input distribution
enough to flip the decoded string. We observed exactly this: `MU4158` from the reference against
`H468` from the port, on a sample whose detection boxes and keypoints had otherwise matched.

The fix is a dedicated `truncateMarks(row)` helper that the pipeline routes through unconditionally,
making the ordering explicit and testable rather than implicit in the arithmetic.

### C. Residual error on non-integer scales

After both fixes, samples whose letterbox scale is an integer are reproduced **bit-exactly**: the
input tensor's leading and middle segments match element-for-element, and the rectified crop's RGB
checksum is identical (105080 = 105080 on `hlpr-1.jpg`).

Non-integer scales do not reach bit-equality. The worst measured case is `scene-2.jpg`
(1920 → 320, scale 3.5625):

| Quantity | Value |
|---|---|
| Channels compared | 307200 |
| Channels differing | 11753 (3.83 %) |
| Maximum absolute difference | 3.92 × 10⁻³ (= 1/255) |
| Sign of all differences | +1 only |
| Recognition (reference) | `藏DT5022` |
| Recognition (port) | `藏DT5022` |

We attempted to fit a single fixed-point layout that reproduces OpenCV at multiple scales. A layout
that matched perfectly at scale 6.0 failed at scale 3.5625 (11753 mismatching channels), and the
"shift after each pass" variant failed everywhere (53646 mismatches). A 135-scale sweep (a 256×144
pseudo-random source, 5.00 × 10⁶ compared pixels) then contradicted the natural hypothesis that the
residual tracks the fractional part of the source coordinate: binned by the fractional coordinate
the rate is *flat* — between 18.3 % and 25.0 % in every one of twenty bins, with the maximum at
0.50–0.55 where the 11-bit coefficient round-off is largest, not a ramp. Splitting instead by the
vertical scale separates the data cleanly: **0.113 %** of pixels differ when the vertical scale is
exactly 1, against **29.108 %** when it is not. The mechanism is therefore not the fractional
coordinate but whether the *second* pass also carries a fractional coefficient — when the vertical
scale is exactly 1 its coefficients are (2048, 0) and the two implementations collapse to the same
single-rounding operation.

**We therefore do not claim bit-equality.** We claim a bounded residual: at most one quantisation
level, single-signed, and — as the next subsection establishes — output-neutral on every sample
tested.

### D. Verification protocol and results

Verification is fully automated and compares eleven fields per sample:

```
┌──────────────────────────┐        ┌───────────────────────────┐
│ Python reference         │        │ Browser port              │
│ tools/hlpr_reference.py  │        │ assets/js/pipeline.js     │
└────────────┬─────────────┘        └─────────────┬─────────────┘
             │ dump intermediates                 │ dump intermediates
             ▼                                    ▼
  _evidence/det_dump.json          _evidence/web__verify_html.json
             └───────────────┬────────────────────┘
                             ▼
                tools/compare_dumps.py
                             ▼
                _evidence/fidelity_report.txt
```

The browser side is driven by headless Chromium. The page publishes its intermediates on a global
object, and the driver writes them to disk directly from Node — an implementation detail worth
recording, because piping CJK-bearing JSON through a Windows shell pipeline re-decodes it under the
legacy code page and silently corrupts the province glyphs.

Comparison fields, in pipeline order: letterbox scale and offsets → input tensor head and mid
segments → raw detector output row → detection box → keypoints → crop dimensions → crop RGB
checksum → decoded string → recognition confidence → detection score → colour scores.

**Results.**

| Sample | Letterbox | Crop RGB sum (ref / port) | Decoded string | Verdict |
|---|---|---|---|---|
| `hlpr-1.jpg` | 320×320, r = 1.0 | 105080 = 105080 | `MU4158` | bit-exact |
| `hlpr-test.jpg` | 1920×1080, r = 1/6 | 2772794 / 2772798 | `苏ED5172` | string match |
| `scene-2.jpg` | 1140×456, r = 0.2807 | 5281904 / 5343214 | `藏DT5022` | string match |
| `plate-yellow-320x48.jpg` | 320×48, r = 1.0 | 1694093 / 1694092 | `粤ZP560港` | string match |
| `crop-0` | — | — | `津B6H920` | match (conf 0.998) |
| `crop-1` | — | — | `皖KD01833` | match (conf 0.961) |
| `crop-6` | — | — | `蒙B023H6` | match (conf 1.000) |
| `crop-8` | — | — | `冀D5L690` | match (conf 0.975) |

**What the "Decoded string" column does and does not mean.** Every string in that column is the
*reference implementation's own output*; the column tests whether the port reproduces it, not
whether the plate was read correctly. This distinction is not pedantry: an independent labelled
set (§IV-F) showed that the reference string for `hlpr-test.jpg` is itself wrong — the vehicle
plate in that image is yaw-compressed to roughly half its true aspect ratio, and the recogniser
returns an eight-character `苏ED51712` at the standard plate aspect. We had been using the
seven-character `苏ED5172` as an acceptance red line, i.e. validating the system against its own
output. The fidelity claim survives untouched (it is a claim about two implementations agreeing);
the *truth* claim did not. See ADR-015.

**8 / 8 samples agree.** Detection scores and colour-classification scores are bit-identical on all
four full-scene samples. Recognition confidence differs by at most 1.63 × 10⁻² absolute
(2.06 × 10⁻² relative) on `hlpr-1.jpg`, consistent with the residual input difference.

The end-to-end pipeline — detection, rectification and recognition — completes in **85 ms** (p50 of 30
warm runs) for a 1140×456 scene on a Kirin 8020 phone, where the adaptive thread count settles at
**6 threads**; single-threaded the same pipeline takes 126 ms, so the multi-threaded gain is **1.5×**.
A headless-Chromium control on a 12-logical-core desktop gives 90.9 ms → 81.2 ms (**1.12×**),
indicating that the on-device gain comes from core heterogeneity rather than from core count alone —
12 threads on the phone are in fact *slower* than one (162 ms). Model loading takes **1.37 s** for all
three graphs.

### E. Engineering notes

Three further defects were found and fixed during the port, each of which is a general hazard rather
than a project-specific one:

1. **Out-of-bounds on degenerate inputs.** With a one-pixel-wide source the vertical pass indexes
   `sy + 1`, which is out of range. Clamping to `size − 1` — matching OpenCV's border behaviour — is
   required.
2. **Sub-resource URLs resolve against the document, not the module.** Sample images referenced with
   document-relative paths silently 404 when the page lives at the site root. All asset URLs are now
   anchored to `import.meta.url`.
3. **Upstream logging noise.** The published graphs declare several hundred initialisers as graph
   *inputs*, which ONNX Runtime reports as warnings. These are benign but flood the developer
   console; they are suppressed for the duration of session construction only.

---

### F. Recogniser accuracy on a third-party labelled set, and the retraction of our own reference string

**Why this experiment exists.** §IV-D established that our port agrees with the reference
implementation. It cannot establish that either of them is *right* — the strings compared in
§IV-D were generated by the system under test. To break that circularity we evaluated the
production recogniser against ground truth that neither it nor we produced.

**Data.** 1000 real, human-annotated Chinese plate crops shipped in the `data/test/` split of the
public `sirius-ai/LPRNet_Pytorch` repository, where **the filename is the label**. The images are
photographs, not synthetic renders; the corpus is 2.52 MB of actual content. Licence: bundled with
a public repository, used here for academic evaluation only. Reproduce with
`tools/lprnet_real_accuracy.py`; `_evidence/A16-verify-rerun.log` re-ran the whole evaluation on
the on-disk copy and reproduced 888/1000, 906/1000 and p=0.1788 **digit for digit**.

**Result.**

| Recogniser | Input | Exact match | Length errors |
|---|---|---|---|
| rpv3 (production, SVTR/CRNN-CTC, 77-class dictionary) | 48×160 | **906/1000 = 90.6 %** | 10 (1.0 %) |
| LPRNet (the set's own model, 68-class CTC) | 24×94 | **888/1000 = 88.8 %** | 70 (7.0 %) |

McNemar's test on the paired per-image outcomes gives **p = 0.1788** — the two rates are not
significantly different. This is the *strongest* possible reading for LPRNet and the weakest for
us: it is LPRNet's own test split, and our recogniser has never seen that distribution. A
candidate replacement would have to win on a benchmark it was not selected on, so **rpv3 stays**.
The one asymmetry worth stating is the failure mode: rpv3's 94 errors are 84 same-length
substitutions (mostly province-abbreviation confusion) and only 10 length errors, whereas LPRNet
loses a character seven times as often (7.0 % vs 1.0 %). For a plate string, a truncation is a
worse error than a substitution.

**The finding that changed a headline number.** The same labelled data exposed a defect in our own
acceptance procedure. The full-scene sample `hlpr-test.jpg` had been read as `苏ED5172`, and that
string was used as the red line for every backend matrix in §V. It is wrong: it was the recogniser's
own output, not a label. Three independent measurements establish the correct reading
`苏ED51712` (eight characters):

1. **Geometry.** The plate quad in the image measures 123.5 × 77.3 px — an aspect of 1.60 against
   the standard 3.14, i.e. the plate is yaw-compressed by roughly 2×. Under GB 7258 / GA 36 glyph
   metrics (45 mm glyph, 12 mm gap) the observed span fits eight glyphs, not seven.
2. **A control experiment** (`tools/stretch_control.py`, n = 200 plates of known 7-character length)
   shows that horizontal stretching **never invents a character** (0/200 transitions from 7 to 8),
   whereas compressing to the golden image's own aspect ratio causes character *loss* in 17.0 % of
   cases (34/200). So the eighth character cannot be an artefact of rectification geometry, while
   the observed 2× yaw compression is a plausible cause of its absence.
3. **Re-measurement at standard aspect**: the production recogniser itself returns
   `苏ED51712` at 0.987 confidence.

A collateral correction: the plate is **green**, not blue. Our earlier colour analysis had
classified the sample as blue; the same tool additionally reported 86 plates in this corpus as
green — none of which is genuinely green (all carry seven-character labels, and the corrected
measurement finds no green plate at all in the 1000) — and a separate "37.2 % green hue" figure
from it is likewise withdrawn. The classifier selected "pixels brighter than the median" and
measured their hue — on a blue plate the brightest pixels are the white characters, whose hue is
numeric noise, so it was measuring glyphs rather than paint. Measuring the paint instead
(saturation ≥ 90, value 45–250, white-balanced against the plate's own characters) gives
G−B = +39 with a neutral white reference, against a blue-plate reference group median of −89.5.
Every per-colour accuracy breakdown produced by the old tool is withdrawn; the aggregate 906/1000
is unaffected.

**Consequences for the rest of the paper.** The backend-placement conclusion of §V survives, because
it is a *differential* claim: detector-on-NPU changes the output string under identical input, and
that holds regardless of which string is correct. What does not survive is the word "correct" —
throughout this paper, `match=` means "agrees with the CPU baseline", never "reads the plate
correctly". Any correctness claim in this paper is now anchored to this section, not to a single
image. See ADR-015.

**Dataset limitation to state up front.** All 1000 plates are seven-character; there are **zero**
new-energy eight-character plates, and the independent colour measurement confirms the corpus
contains no genuine green plate either. Eight-character behaviour is therefore unmeasured here and
is the first gap to close before any deployment claim.

---

## V. On-Device Characterisation: Kirin 8020

### A. Experimental setup

| Item | Value |
|---|---|
| Device | HUAWEI nova 14 Pro (MIA-AL00) |
| SoC | Kirin 8020 |
| OS | HarmonyOS 6.1.0.135 (SP8C00E120R7P5), SDK API 24 |
| Runtime | MindSpore Lite Kit 2.6.0 NDK (`OH_AI_*` C API) |
| Second path (§V-F) | CANN/HiAI offline models via Neural Network Runtime (`OH_NN*` C API) |
| Backend chain | NNRT → NPU |
| NNRT device name | `NPU_ohos.boot.hardware.kirin8020_v2_0` |
| Protocol | per-model bench: 5 warm-up + 30 measured iterations, P50 (release build, `-O2 -DNDEBUG`); end-to-end pipeline matrices: 1 warm-up + 3 reps |
| Build mode | **release** — earlier debug-build (`-O0`) figures are internally comparable only and are superseded where marked |
| Thermal state | battery 83 % (newest round, 2026-09-20; earlier rounds: battery 100 %, battery temp 36.0 °C, thermal level 2) |

Three host-side constraints were established empirically and are prerequisites for any reproducible
measurement on this platform:

1. **Sessions must stay resident.** The NNRT delegate's destructor path is crash-prone; creating and
   destroying a session per inference crashes the process after a number of iterations.
2. **The device name must be read, not hardcoded.** It is not a stable constant across devices —
   earlier documentation and other SKUs use `HIAI_F`.
3. **Dynamic batch dimensions fail at build time.** `OH_AI_ModelBuildFromFile` rejects graphs with a
   dynamic batch axis, so batch is fixed to 1 at conversion.

Because the platform exposes no hitrace tags for NPU activity, we cannot observe operator placement
directly. We therefore adopt a **CPU-difference method**: each model is executed on both the CPU and
the NPU backend, and delegation is inferred from the latency gap together with output consistency.

### B. Latency scaling

| Model | NPU P50 (ms) | CPU P50 (ms) | Speedup |
|---|---|---|---|
| ResNet-50 | 6.02 | 78.90 | **13.11×** |
| ResNet-18 | 3.27 | 34.00 | **10.39×** |
| YOLOv8n | 15.60 | 71.07 | **4.56×** |
| MobileNetV2 | 2.74 | 10.22 | **3.72×** |
| MobileNetV3-Small | 3.37 | 3.44 | **1.02×** |
| MobileNetV2 (INT8) | 5.08 | 4.97 | **0.98×** |

The speedup decreases monotonically with model size, from 13.11× to 0.98×. At the small end the
accelerator delivers no benefit at all, and the fully INT8-quantised MobileNetV2 is **2 % slower on
the NPU than on the CPU**. The mechanism is not model-specific: graph partitioning, delegate
dispatch and operator fallback incur fixed costs that a small graph cannot amortise. The engineering
rule that follows is that NPU offload must be justified per model by its operator coverage and graph
size, and that "the NPU is faster" is not a safe default.

### C. The project's own pipeline models on-device

§V-B characterises the accelerator with general-purpose backbones. The three models this project
actually deploys were measured with the **same protocol** (5 warm-up + 30 measured iterations, P50) on
the same device, under a **release** build. An earlier round of the same table was taken under a debug
(`-O0`) build; those figures are internally comparable but are **superseded** by the ones below.

| Model | Req. backend | **Landed on** | P50 (ms) | Mean (ms) | Output L2 | Verdict |
|---|---|---|---|---|---|---|
| Detector, bare head (1.85 MB) | `nnrt` | **NPU** `NPU_ohos…kirin8020_v2_0` | **5.3505** | 5.4515 | 1043.4696 | pass (but see below) |
| Detector, bare head | `cpu` | CPU | 7.6638 | 7.7897 | 1042.6634 | pass |
| Recogniser rpv3 (4.48 MB) | `nnrt` | **NPU** `NPU_ohos…kirin8020_v2_0` | **3.9883** | 4.0780 | 2.9997 | pass |
| Recogniser rpv3 | `cpu` | CPU | 9.1286 | 9.1910 | 2.9921 | pass |
| Classifier, fp32 (1.61 MB) | `nnrt` | **NPU** `NPU_ohos…kirin8020_v2_0` | 1.0012 | 1.0046 | 0.9980 | pass |
| Classifier, fp32 | `cpu` | CPU | **0.8395** | 0.8506 | 0.9999 | pass |
| Classifier, fp16 (0.82 MB) | `nnrt` | **CPU — silent fallback** | 0.8756 | 0.9150 | 0.9999 | **no NPU kernel** |

Delegation is confirmed independently of the latency gap by an **fp16 buffer fingerprint**: the same
output memory interpreted as fp16 yields an L2 norm of 48950.62 (detector) and 5042.71 (recogniser)
on the NPU rows and exactly 0.0000 on every CPU row — the accelerator is demonstrably computing, and
the CPU rows are not merely re-labelled copies of NPU runs.

Three results follow, and each contradicts a plausible default assumption.

**(1) The recogniser is the only model that unambiguously benefits: 2.29×.** It is also the model
whose output is most robust to what the accelerator does to the numbers — a CTC head takes an argmax
per timestep, so a small probability perturbation does not move the decoded character.

**(2) The detector is 1.43× faster on the NPU and still unusable there.** Its NPU output L2 differs
from CPU by 0.077 %, and because the detection head's keypoints are `logit × anchor` with anchors up
to 433, that relative perturbation is amplified into a ≈10 px displacement — enough to move the
rectification quad and **flip a character** (`苏ED5172` → `苏E05172`; both strings are *ours* — the
first is the reference implementation's output, the second is what the NPU build produces, and
neither is ground truth, see §IV-F). The production configuration
therefore keeps detection on the CPU: the end-to-end string is the deliverable, not the per-stage
latency.

**Two limits on that conclusion, stated plainly — and one of them now closed.** First, it is
differential: NPU placement changes the output, which is a claim about sensitivity, not about which
string is correct. Second, the original measurement requested the NNRT device with
`EnableFP16(true)` unconditionally, so the claim was formally limited to fp16 NPU builds; an fp32 NPU
build of the same graph had never been measured. **That gap is now closed: the harness gained an
explicit `nnrt_fp32` tier (landing label `NNRT:…kirin8020_v2_0#fp32`), and the detector was re-run
under it. The character still flips — 3/3 repetitions produce `苏E05172` — while the recogniser
under the same fp32 tier stays correct (3/3). The claim therefore holds for **both** precisions and
needs no fp16 qualifier.** The CPU side of these ratios is a tuned baseline rather than an
unexplored configuration: the harness sets `SetPerformanceMode(HIGH)` for CPU sessions, and the
thread-count sweep in §V-H confirms that the pinned value of 4 is optimal for all three models.

**(3) The classifier is 0.84× on the NPU — i.e. 19 % slower than the CPU.** This is §V-B's
small-graph rule reproduced *inside this project's own models*: at 96×96×3 with 1.61 MB of weights,
fixed delegate overhead exceeds the arithmetic saved. The fp16 variant is worse still — it cannot
build an NPU kernel at all and falls back to the CPU **without returning an error**, so a caller that
trusts the requested backend silently measures the CPU and attributes the number to the NPU. Only the
`LANDED=` field, printed by the harness, distinguishes the two cases.

**A framework column is mandatory for these numbers.** The rows above all traverse
MindSpore Lite → NNRT → HiAI. Separate measurements of the same models through other stacks —
ncnn-CPU and ncnn-Vulkan on the same device, and a `.om` path built with the vendor's own offline
model generator (§V-F) — give figures that are *not* directly comparable, because framework, graph
optimiser, operator library and quantisation all differ. The comparable quantities are ratios within
one stack; cross-stack numbers are indicative of magnitude only.

### D. Numerical consistency

Each model's output L2 norm is compared between the two backends.

| Model | NPU L2 | CPU L2 | Relative error | Verdict |
|---|---|---|---|---|
| ResNet-50 | 23.9721 | 24.0301 | 0.242 % | pass |
| ResNet-18 | 48.3047 | 48.3049 | 0.000 % | pass |
| MobileNetV2 | 75.1912 | 75.2415 | 0.067 % | pass |
| YOLOv8n | 3.93 × 10²¹ | 50569.64 | 7.78 × 10¹⁸ % | **fail** |
| YOLOv8n (fixed) | 50562.0956 | 50569.5744 | 0.015 % | pass |

The YOLOv8n outlier is not a precision problem, as §V-E explains.

### E. Runtime defect: FP16 buffers declared as FP32

Models converted with `converter_lite --fp16=on` return some output tensors whose declared dtype is
`OH_AI_DATATYPE_NUMBERTYPE_FLOAT32` (enum 43) while the backing buffer contains **FP16** bit
patterns. An application reading the buffer as declared receives values of order 10⁸–10²¹ — which is
exactly the YOLOv8n row above. Reinterpreting the same memory as FP16 yields correct values.

The defect is a broken dtype contract rather than an accuracy regression, and it is silent: the
inference call succeeds. Inserting an explicit `Cast` node at the tail of the computation graph (the
`yolov8n_fix` variant) restores the contract and reduces the relative error to 0.015 %.

A second, unrelated incompatibility affects official 1.x-era artefacts (`mobilenetv2.ms`,
`ghostnet_int8.ms`): `OH_AI_ModelBuildFromFile` fails with `ret = -1` and no diagnostic detail on
every device and on the **CPU backend as well**, and re-serialising them with `converter_lite 2.6`
also fails. These artefacts can no longer be read by any current toolchain.

### F. Coverage, and a second toolchain on the same device

**Covered by §V-B and §V-C:** the conversion and host-integration path, the host-side constraints,
the dtype-contract defect, the measurement protocol, the accelerator's latency-scaling behaviour, and
**all three of this project's LPR models** measured on-device under the 30-iteration protocol. §V-C is
the part that speaks to the pipeline as deployed; §V-B supplies the scaling law that explains why
§V-C's three models behave differently from one another.

**Not covered:** an end-to-end *pipeline* latency on the NPU. §V-C measures per-model inference only;
the surrounding letterbox, rectification and decode stages dominate the end-to-end budget and are not
accelerated. We report per-model figures precisely because a single end-to-end number would hide the
fact that the three stages have opposite conclusions.

#### A second toolchain: the vendor offline-model path

The measurements above traverse MindSpore Lite → NNRT → HiAI. To test whether that stack is
*dictating* the conclusions, the same models were also converted with the vendor's own offline model
generator (`tools_omg`, shipped with the CANN/HiAI DDK toolchain) and executed through the standard
Neural Network Runtime API, which exposes device enumeration and explicit device binding.

Two results are worth recording, both with the caveat that they use 3 repetitions rather than 30 and
are therefore **indicative, not benchmark-grade**:

| Model | Toolchain | Result |
|---|---|---|
| Vendor-supplied `.om` (image classifier, 2.50 MB) | CANN → NPU | compiled, executed, steady state **0.94 ms**; the 1000-way output sums to **1.0005**, i.e. a valid softmax — an integrity check that the NPU computed a complete forward pass |
| This project's **recogniser**, re-converted to `.om` | CANN → NPU | compiled, executed, **3.87 ms** steady in the first round; a re-measurement round (2026-09-20) gives **5.19–5.63 ms** — vs **3.9883 ms** through MindSpore Lite under the 30× release protocol |
| This project's **classifier** (fp32), re-converted to `.om` | CANN → NPU | compiled, executed, **0.96 ms** steady (re-measured **0.95–0.97 ms**; vs 1.0012 ms through MindSpore Lite) |
| This project's **detector, bare head** (= the production configuration), re-converted to `.om` | CANN → NPU | compiled, executed, **5.0–6.4 ms** in the first round, **3.90–5.10 ms** on re-measurement; all three output heads present and populated — `[1×45×40×40]`, `[1×45×20×20]`, `[1×45×10×10]` |
| This project's **detector with decode in-graph**, re-converted to `.om` | CANN | **rejected at compile time** |

The first row is the methodological point: the offline-model entry point **accepts the vendor's own
`IMOD` format and rejects the MindSpore Lite `.ms` format**, with the vendor's compatibility checker
agreeing in both directions (a same-run control re-confirms this: the three project `.ms` files were
rejected with `compat=1`/`build_rc=1` on 2026-09-20 while the six `.om` files compiled). The gate is
therefore a **model-format**, not a permission or capability gate — the device advertises
`SystemCapability.AI.HiAIFoundation` and `SystemCapability.AI.NeuralNetworkRuntime`, and both runtime
libraries are present and loadable.

On the strength of the two measurement rounds, the vendor toolchain does **not** displace MindSpore
Lite for this project's models. The detector bare head is comparable (3.90–5.10 ms vs 5.35 ms), the
classifier is comparable (0.95–0.97 ms vs 1.00 ms), but the **recogniser is consistently slower**
through `.om` (3.87 ms first round, 5.19–5.63 ms re-measured, against 3.99 ms under MindSpore Lite).
Since the recogniser is the one model that actually benefits from the accelerator (§V-C(1)), the
second toolchain's only structural contribution is the format-gate evidence; promotion to the
production backend would still require the same-input L2 comparison, a 30× protocol and thermal
annotation that the 3-repetition `.om` runs do not provide.

The last two rows reproduce §V-C(2) through a completely independent toolchain: the same rank-5
constant tensors that prevent the detector from being delegated under MindSpore Lite also cause the
vendor's own compiler to reject the graph, while the bare-head variant — the variant this project
actually ships — compiles and runs. **Two independent compilers refusing the same construct is
considerably stronger evidence than either alone**, and it converts "our detector has an awkward
decode" from a project-specific quirk into a property of the hardware's programming model.

**A finding about harnesses, not models.** The bare-head row initially failed with a non-zero return
code from the runtime, which reads as "the model does not execute". It does. The model declares
*three* output tensors; our harness created and supplied one. The runtime reports the arity mismatch
as a generic failure, indistinguishable at the call site from an unsupported operator. A
measurement harness that assumes single-input/single-output will therefore **misclassify working
models as broken** — a failure mode worth naming, because it produces a plausible negative result
that no amount of careful benchmarking will catch.

#### Why the classifier is slow: a structural cause, computable in advance

§V-C(3) reports the classifier running 35 % *slower* on the NPU than on the CPU, and reads it as the
small-graph rule. The vendor's NPU programming guide supplies a sharper, *quantitative* form of that
rule, and it turns the observation into something predictable from the ONNX file alone:

> when both `Cin` and `Cout` of a convolution are multiples of 16, the NPU reaches its maximum
> throughput; when both are below 16, the achievable throughput is scaled by
> `(Cin × Cout) / 256 × max`.

Counting the convolutions that satisfy the first condition gives an **upper bound on accelerator
utilisation** for any candidate model, before any conversion or measurement:

| Model | Convolutions | `Cin` and `Cout` both ×16 | Utilisation ceiling |
|---|---|---|---|
| Classifier `litemodel_cls_96x` | 52 | 10 | **19.2 %** |
| Recogniser `rpv3_mdict_160_r3` | 36 | 20 | 55.6 % |
| Detector `y5fu_320x_head` | 85 | 60 | 70.6 % |
| Recogniser `lprnet` (candidate) | 16 | 13 | 81.2 % |

The ordering reproduces §V-C: the classifier, the only model that is *slower* on the accelerator, is
also the only one whose convolutions are overwhelmingly narrower than the hardware's native width
(30 of its 52 convolutions have a channel count below 16). The vendor's own guide states that its
recommendation table is derived from **NPU-internal hardware utilisation and is a self-comparison,
not a cross-architecture one** — which is exactly the distinction §V-C(3) needs: the classifier is
not a case of the NPU performing badly, but of a graph whose arithmetic the NPU cannot pack.

#### "NPU supportability" is a joint property of model and toolchain

The same candidate recogniser (`lprnet`, a CTC network whose structure scores 81.2 % above) was
converted to `.om` and executed through the vendor toolchain:

```
compat=0   build_rc=0   run_rc=0
input  [1×3×24×94]  →  output [1×1×68×18]   steady state ≈ 1.4–1.8 ms
```

Under MindSpore Lite, the same graph had been classified as landing on the CPU. Two toolchains,
one model, two verdicts — so a statement of the form "model X is/is not NPU-supported" is
under-specified without naming the compiler.

The partitioner's decision is visible in the device log, and it does not match the earlier
hypothesis. An earlier ADR attributed this model's CPU landing to six `perm=[0,3,2,1]` transposes and
proposed a rewritten variant (`lprnet_npufix`) with those removed. Measured on the vendor toolchain:

| Variant | Partition decision | Operator the NPU rejects |
|---|---|---|
| `lprnet` | `NPU:2, CPU:1` | **`ReduceMean`** |
| `lprnet_npufix` (transposes removed) | `NPU:2, CPU:1` | **`ReduceMean`** |

The rewrite changes nothing, because the transposes were never the obstacle: the graph's L2
normalisation (`ReduceMean` + `Pow` + `Div`) is, and the operator library says so explicitly
(`... the op name [ReduceMean_58] type [ReduceMean] is not supported in npucl store
[elementary_lib]`). **A structural hypothesis that was never executed is worth exactly nothing** —
this one had been recorded as a plan and is now falsified.

It also sharpens §V-C: "landed on NPU" and "runs entirely on NPU" are different claims. The
classifier and the bare-head detector partition as `NPU:1, CPU:0`, while the recogniser is
`NPU:3, CPU:2`. A device that reports success has still silently split the graph.

### G. Data quality

The reported backbone latencies in §V-B are single-round measurements; the protocol calls for at
least five rounds, so their confidence intervals are indicative only. **The three pipeline models in
§V-C were measured under the full protocol (5 warm-up + 30 measured iterations, P50 reported)**, and
were re-measured on 2026-09-20 under a release build with the same protocol and identical results to
within a few percent; the second-toolchain comparison in §V-F uses 3 repetitions and is labelled
accordingly. Two models (`ghostnet_int8_official`, `mnv2_official`) had failing rounds and are
excluded. Thermal state materially affects the result:
raising the thermal level from 2 to 3 increases NPU latency by 15–20 % while CPU latency moves by
less than 2 %, so any speedup figure is only meaningful together with its thermal annotation. The
2026-09-20 round could not read the thermal level on-device (the service query returned empty) and
records battery state only; repeat rounds with `@ohos.thermal` annotation remain outstanding.
The two 2026-09-20 rounds (§V-H) additionally expose a **round-to-round asymmetry** that bounds how
any single ratio should be read: CPU-side p50 moved 32–53 % between rounds taken 20 minutes apart
with identical code and protocol, while NPU-side p50 moved less than 8 %.

### H. Per-frame stage budget and the CPU baseline

§V-C measures per-model inference in isolation. The pipeline's own per-frame budget, instrumented
stage by stage on the production configuration (`det=CPU / rec=NPU / cls=CPU`, release build,
golden sample `hlpr-test.jpg`, 1920×1080 input):

| Stage | ms | Share | Note |
|---|---|---|---|
| Letterbox | **5.43** | 15.0 % | fixed preprocessing cost |
| Detector input packing (NHWC) | **0.35** | 1.0 % | |
| Detector inference (CPU) | **18.15** | 50.3 % | |
| Decode + NMS | 0.04 | 0.1 % | |
| Rectification crop | ≈2.05 | 5.7 % | |
| Recognition (NPU) | 7.34 | 20.3 % | |
| Classification (CPU) | 2.69 | 7.4 % | |
| **Total per frame** | **36.11** | | → ≈27.7 fps single-frame ceiling |

Two findings follow. First, the combined encode+infer bucket (18.51 ms) is **almost entirely
inference**: layout packing costs 0.35 ms, falsifying the earlier working hypothesis that the
unexplained ~17 ms inside the detector stage was NHWC packing. Second, the detector's in-pipeline
inference (18.15 ms) is 59 % slower than the same model on the same backend in the isolated
benchmark (11.37 ms, same round, same build).

**The pipeline-vs-benchmark gap, investigated and bounded.** The same-model, same-backend,
same-round gap between in-pipeline inference and the 30× benchmark is the largest unexplained
term in the budget, so we ran five controlled experiments against it, one per plausible
mechanism, each with a same-round control. **All five were negative**:

| Mechanism proposed | Experiment | Result |
|---|---|---|
| Per-frame allocation + first-touch faults (~7 MB/frame) | full buffer reuse (`thread_local` scratch for letterbox, resize intermediate, packs, output tensors) | gap unchanged |
| Worker-pool wake-up between pipeline phases | production detector on a 1-thread tier | in-pipeline time unchanged (21.6 vs 20.4 ms) — **independent of thread count** |
| Input/output marshaling (1.2 MB in, 378 KB out per frame) | benchmark extended with an I/O-faithful timed loop (`p50Io`) | **+0.6 ms only** (7.29 → 7.89 ms) |
| Per-call tensor-handle queries | handle arrays cached on the session at load | gap unchanged |
| Input data distribution (bench: dense random; pipeline: 94 % zero border) | deterministic fill switched to the pipeline's mostly-zero pattern | benchmark unchanged (7.54 vs 7.44 ms) |

A sixth hypothesis — load-pattern-driven CPU frequency (the benchmark's back-to-back iterations
hold the clock up; the pipeline's memory-bound phases let it drop) — received only weak support:
inserting 3 ms gaps between benchmark iterations moved the p50 by 1.2 ms. The device's thermal
service is not readable from the shell on this build, so the state-dependent component cannot be
observed directly. What we can state is bounded and useful: the gap is **not** in the marshaling,
the allocation path, the thread configuration, or the data — and per-frame totals across six
identical-protocol rounds span 29–56 ms while the benchmark stays within 7.3–8.7 ms, i.e. the
variance is a property of the measurement environment, not of the code path. We therefore report
the best round (29.09 ms: letterbox 4.71, packing 0.34, detector inference 14.44, decode+NMS 0.05,
rectification ≈2.0, recognition 5.81, classification 1.21) together with the six-round band
rather than a single point estimate. The real-time camera loop, being continuously loaded, is
expected to sit at the fast end of that band; a single cold self-test frame at the slow end.

**The CPU baseline is now a tuned one.** The thread count was previously hard-wired to 4 without
ever being scanned, which left every NPU/CPU ratio in this section resting on an untuned CPU
configuration. A dedicated sweep (`cpu_t{1,2,4,6,8}` tiers, 5 warm-up + 30 measured iterations,
same release build) gives:

| Model | t1 | t2 | **t4** | t6 | t8 |
|---|---|---|---|---|---|
| Detector, bare head | 12.84 | 12.98 | **11.38** | 11.96 | 15.21 |
| Recogniser rpv3 | 16.53 | 15.88 | **13.89** | 13.93 | 16.79 |
| Classifier, fp32 | 1.04 | 0.90 | **0.82** | 1.36 | 2.18 |

Four threads is optimal for all three models, matching the hard-wired value: the CPU side of
§V-C's ratios is a high-performance-mode, optimally-threaded baseline. Thread counts beyond the
physical core count are actively harmful (t8 is 6–165 % slower than t4), an oversubscription
effect rather than a measurement artefact — the `cpu_t4` tier reproduces the plain `cpu` tier to
within 0.2 % in the same round.

---

## VI. Discussion

**Fidelity is a stronger claim than agreement.** The port study produced a result that a
final-output comparison would have missed entirely: on `hlpr-1.jpg` the letterbox scale is exactly
1.0, so no resampling should occur — yet the two implementations initially disagreed. Only because
we compared the intermediate tensor did we discover that OpenCV's resize was being invoked on a
degenerate path and that its fixed-point rounding, not the model, was responsible. Had we compared
strings alone and adjusted the threshold until they matched, we would have shipped a port whose
behaviour depended on luck.

**…but fidelity is still not correctness, and we learned that the hard way.** The paragraph above
was written believing that a stronger internal criterion made external ground truth optional. §IV-F
shows the limit of that argument: the strings our fidelity study reproduces *are* the reference
implementation's output, and when an independent labelled set finally arrived, it revealed that one
of those strings — the seven-character plate we had wired into every on-device matrix as an
acceptance red line — was wrong. Nothing in the fidelity methodology could have caught this, because
the methodology asks "do the two implementations agree", and they did. The practical lesson is a
ordering one: an agreement criterion is safe to adopt only after the same value has been checked
against something the system under test did not produce. We report this as a result of the study
rather than as an erratum to it, since the failure mode (validating a system against its own output)
is one that any fidelity-first project can fall into silently.

**Negative results deserve publication.** Two of this paper's seven contributions are negative: bit
equality is *not* achievable on non-integer resize scales, and NPU offload is *not* beneficial for
small graphs. A third (§IV-F) is self-falsifying rather than negative: it removes a claim the
earlier draft relied on. All are bounded and reproducible, and all are more useful to a practitioner than a
favourable headline number would be.

**The cost of unverified ports is invisible.** Neither of the two defects in §IV raises an exception.
They produce plausible-looking plates. In a deployment without intermediate-tensor verification, they
would surface as an unexplained accuracy deficit attributed — incorrectly — to the model.

**Two independent toolchains reaching the same verdict is worth more than either alone.** §V-F
re-ran the three models through the vendor's own offline model generator and the standard runtime
API, a completely separate software path from MindSpore Lite. The stage-level conclusions survived:
the recogniser benefits, the classifier does not, and the detector's in-graph decode is refused.
Agreement across independent implementations rules out the possibility that the earlier results were
an artefact of one framework's delegation heuristics — which is the natural objection to any
single-stack measurement.

**And a lesson about the instruments, not the hardware.** The bare-head detector initially reported a
runtime failure and we nearly recorded it as a model limitation; the cause was our own harness
supplying one output tensor to a three-output graph (§V-F). The general form of this hazard is that
a measurement harness silently encodes assumptions — arity, dtype, layout, ordering — and when one is
violated, the runtime's error is indistinguishable from a genuine capability limit. The negative
result looks exactly like a real one. We record it because the discipline this paper argues for
(§IV: compare intermediates, not endpoints) applies to the measurement apparatus as much as to the
port.

---

## VII. Limitations and Threats to Validity

1. **Single device for on-device claims.** All NPU measurements come from one Kirin 8020 unit. The
   monotonic speedup-versus-size trend is a single-point-per-model observation; establishing it as a
   scaling law requires additional SKUs.
2. **CANN-path figures are 3-repetition.** The MindSpore Lite numbers in §V-C use the full
   30-iteration protocol; the second-toolchain comparison in §V-F does not, and is labelled
   indicative for that reason.
3. **No accelerated end-to-end pipeline figure.** §V-C reports per-model inference. The letterbox,
   rectification and decode stages that dominate the end-to-end budget run on the CPU and are not
   accelerated by any of the paths measured here.
4. **Residual error is characterised, not eliminated.** §IV-C bounds the non-integer-scale
   discrepancy but does not reproduce OpenCV's SIMD accumulation order.
5. **Test set size.** Eight samples, chosen to span blue / yellow / green plates, single and double
   layers, full scenes and tight crops. It is adequate for a fidelity study, where each sample
   yields hundreds of thousands of compared values, but it is *not* an accuracy benchmark.
   Accuracy is reported separately in §IV-F on a third-party labelled set, and the two claims must
   not be conflated: §IV tests agreement between implementations, §IV-F tests the recogniser
   against ground truth.
6. **Accuracy figure — scope and caveats.** The recogniser scores **906/1000 = 90.6 %** on 1000
   real, human-annotated plate crops whose filenames carry the ground truth (§IV-F). Three limits
   travel with that number: the set contains **only seven-character plates** (zero new-energy
   eight-character plates), it is the evaluation split of a *competing* model that our recogniser
   has never seen, and it measures tight crops rather than full-frame detection-then-recognition.
   It is therefore an exact-match rate on crop input, not an on-road accuracy claim. Earlier
   versions of this paper stated that "no accuracy figure is claimed"; that was true when it was
   written and is superseded by §IV-F — see ADR-015 for why we regard the retraction of our own
   acceptance string as a result rather than an erratum.

---

## VIII. Conclusion

We have ported a three-stage LPR pipeline to a dependency-free browser implementation and shown that
the port can be held to a standard considerably stronger than output agreement. Two defects — a
floating-point transcription of OpenCV's fixed-point resize, and an out-of-order keypoint truncation
— were found only because every intermediate tensor was compared, and both were silent in the sense
that neither raised an error nor produced an implausible output. After correction, all eight test
samples decode identically to the reference, and integer-scale samples are bit-exact.

On the Kirin 8020 we measured a speedup range of 13.11× down to 0.98× across six backbone networks,
and located a runtime dtype-contract defect that produces 10²¹-magnitude garbage from FP16 buffers
declared as FP32.

Closing the loop on this pipeline's own models produced the paper's most practically useful result,
and it is deliberately unflattering: of the three stages, exactly one benefits from the accelerator
(2.27×), one is measurably *worse* on it (0.74×), and one is faster yet returns the wrong characters.
A single end-to-end latency figure for this pipeline on this hardware would have concealed all three
outcomes behind an average. Re-running the same models through a second, independent toolchain — the
vendor's offline model generator and the standard runtime API — reproduced every stage-level verdict,
and showed that the gate between the two paths is the *model file format*, not any capability or
permission. Both results argue for the same discipline: characterise per stage and per toolchain,
quantify the boundary, and publish the negative cases. A final step extended that discipline inwards:
measuring the recogniser against 1000 third-party human-labelled crops (90.6 % exact match, with no
replacement candidate significantly better) invalidated an acceptance string that this work had
supplied to itself. Agreement between implementations, it turns out, is a strong claim about the port
and no claim at all about the world.

---

## Acknowledgment

The recognition models and the reference implementation are the work of the HyperLPR project [3]
(Apache-2.0). Browser-side execution uses ONNX Runtime Web [4] (MIT).

---

## References

[1] B. Shi, X. Bai, and C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence
Recognition and Its Application to Scene Text Recognition," *IEEE Trans. Pattern Anal. Mach.
Intell.*, vol. 39, no. 11, pp. 2298–2304, Nov. 2017.

[2] B. Shi, X. Wang, P. Lyu, C. Yao, and X. Bai, "Robust Scene Text Recognition with Automatic
Rectification," in *Proc. 24th ACM Int. Conf. Multimedia (MM)*, Amsterdam, The Netherlands, 2016,
pp. 1218–1226.

[3] szad670401, "HyperLPR: High Performance Chinese License Plate Recognition Framework," GitHub
repository. [Online]. Available: https://github.com/szad670401/HyperLPR. License: Apache-2.0.

[4] Microsoft, "ONNX Runtime Web," ONNX Runtime documentation. [Online]. Available:
https://onnxruntime.ai/docs/tutorials/web/

[5] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist Temporal Classification:
Labelling Unsegmented Sequence Data with Recurrent Neural Networks," in *Proc. 23rd Int. Conf.
Machine Learning (ICML)*, Pittsburgh, PA, USA, 2006, pp. 369–376.

[6] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.

[7] Huawei, "MindSpore Lite: A Lightweight and Efficient AI Inference Framework," MindSpore
documentation. [Online]. Available: https://www.mindspore.cn/lite

[8] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid
Networks for Object Detection," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*,
Honolulu, HI, USA, 2017, pp. 936–944.

[9] J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," arXiv:1804.02767, 2018.

[10] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural
Networks," in *Proc. 36th Int. Conf. Machine Learning (ICML)*, Long Beach, CA, USA, 2019,
pp. 6105–6114.

[11] sirius-ai, "LPRNet PyTorch," GitHub repository — its `data/test/` split supplies the 1000
human-labelled plate crops of §IV-F (filename as ground truth). [Online]. Available:
https://github.com/sirius-ai/LPRNet_Pytorch

[12] Huawei, "MindSpore Lite Kit 简介" and "Neural Network Runtime Kit 简介," HarmonyOS guides,
2026. [Online]. Available:
https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/mindspore-lite-kit-introduction,
https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/neural-network-runtime-kit-introduction.
Cited for three platform statements that bound §V-F and §VII: MindSpore Lite documents heterogeneous
inference across CPU/GPU and NNRt devices (so a "GPU unsupported" verdict constrains the linked
runtime build, not the SoC); NNRt lists ~56 common operators but implements none of them, deferring
to the vendor driver (the reason an operator checklist is not a feasibility test); and NNRt offers
no general-purpose hardware backend, so GPU offload cannot arrive through it.

---

## Appendix A — Reproducibility Index

Every number in this paper maps to a file or command in the project repository.

| Claim | Source |
|---|---|
| Model sizes, 14,206,477 B total | `assets/models/` |
| Port fidelity, 8/8 agreement | `_evidence/fidelity_report.txt` |
| **§IV-F: accuracy on 1000 labelled crops (906 / 888, McNemar p=0.1788)** | `_evidence/A16-real-dataset-accuracy-20260918.md`, per-image error tables `_evidence/lprnet_accuracy_20260918.md`, raw log `_evidence/A16-lprnet-real-accuracy-20260918.log` |
| **§IV-F: reproducibility re-run on the on-disk corpus** | `tools/lprnet_real_accuracy.py` → `_evidence/A16-verify-rerun.log` (888/1000, 906/1000, p=0.1788 reproduced digit for digit) |
| **§IV-F: stretch/compress control (0/200 vs 34/200)** | `tools/stretch_control.py` |
| **§IV-F: golden-image decisive checks (quad 123.5×77.3, conf 0.987)** | `tools/golden_decisive_checks.py`, `tools/glyph_count_geometry.py` |
| **§IV-F: plate-paint colour measurement + classifier audit** | `tools/plate_face_colour.py`, `tools/colour_classifier_audit.py`; the withdrawn classifier is `tools/plate_colour_analysis.py` (marked DEPRECATED) |
| **§IV-F: dataset provenance and licence** | `_dataset/real/README.md` |
| **Acceptance-string retraction (methodological)** | `docs/adr/ADR-015-truth-string-retraction-and-landing-evidence.md` |
| Reference intermediates | `tools/hlpr_reference.py` → `_evidence/hlpr_reference.json` |
| Browser intermediates | `tools/web_check.mjs`, `tools/_verify.html` → `_evidence/web__verify_html.json` |
| Field-by-field comparison | `tools/compare_dumps.py` |
| Resize reverse engineering | `tools/probe_resize.py`, `tools/probe_kernel.py`, `tools/fit_rounding.py` |
| Resize layout search | `tools/fit_resize.mjs` |
| Browser end-to-end latency | `window.__DEMO.modelLoadMs`, live demo panel |
| Backbone latencies, dtype defect | on-device benchmark records (sibling project) |
| **§V-C: three pipeline models, 30× P50** | `_evidence/bench30_matrix_20260918.log` (tag `LprMatrix`), harness `probeAll()` in `Index.ets` |
| **§V-F: CANN/NNRt cross-toolchain check** | `_evidence/cann_multout_20260918.log`, `_evidence/A10-cann-nnrt-probe-20260918.md`, `_evidence/A11-cann-om-end-to-end-20260918.md` |
| **§V-F: our models re-converted to `.om`** | `convert.sh` + `ddk/tools/tools_omg` (DDK-tools-next-6.0.1.0 + kirin9020 plugin); artefacts `probe_om_{dethead,rec,cls}.om` |
| **Multi-output arity finding (§V-F)** | `cpp/nnrt_probe.cpp` — before/after `run_rc` on `probe_om_dethead.om` |

## Appendix B — Note on Data Provenance

This paper contains **no synthetic or fabricated measurements**. Every latency, tensor-difference
and accuracy figure is a logged observation, indexed in Appendix A.

Accuracy was originally omitted from this paper on the grounds that no labelled benchmark had been
run. That omission has since been closed by §IV-F, which reports an exact-match rate on 1000
third-party, human-annotated plate crops (`_evidence/A16-real-dataset-accuracy-20260918.md`,
reproducible digit-for-digit via `_evidence/A16-verify-rerun.log`). The earlier wording
("accuracy figures are deliberately omitted; treat their absence as intentional") is superseded and
is retained here only to record that the claim was made before the data existed. Where a quantity
was not measured, the text still says so explicitly (§V-E, §V-F, §VII) rather than substituting a
plausible value. Synthetic plate renders (`tools/gen_plate_dataset.py`) remain restricted to
pipeline regression and are never used to report accuracy.
