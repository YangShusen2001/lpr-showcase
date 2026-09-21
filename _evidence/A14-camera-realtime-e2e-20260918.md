# A14 — 相机实时端到端（三档后端）+ 帧率瓶颈定位与修复

- 日期：2026-09-18
- 设备：nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1 · 序列号 `4CY9K25614046328`
- 页面：`LprDemo/entry/src/main/ets/pages/CameraPage.ets`（本轮新增）
- 构建：`buildMode=release`（**关键，见 §3**）
- 日志：`cam_gear0_prod_20260918.log` / `cam_gear1_npu_20260918.log` / `cam_gear2_gpu_20260918.log`

---

## 1. 做了什么

新增「相机实时识别」页：**双路预览** —— XComponent 显示，第二路 `ImageReceiver` 取帧分析。

```
CameraInput
  ├─ PreviewOutput#1 ─→ XComponent Surface      只负责显示
  └─ PreviewOutput#2 ─→ ImageReceiver           只负责分析
                          ↓ imageArrival
                       readLatestImage → getComponent(JPEG)
                          ↓
                       lpr.cameraFrameAsync(nv21, w, h, stride, rotation, detId, recId, clsId, mode, recSlot, clsSlot)
                          ↓ native 一趟做完
                       NV21→RGBA(+旋转) → letterbox → det → NMS → 矫正 → rec → CTC → cls
```

三个档位（一次只切一件事，任两档差异只能来自后端）：

| 档位 | det | rec | cls |
|---|---|---|---|
| 生产 | CPU | NPU | CPU |
| 全 NPU | NPU | NPU | NPU |
| 全 GPU | ncnn-Vulkan | ncnn-Vulkan | ncnn-Vulkan |

**功能已验证**：相机实时读到真实车牌 —— `苏A1085K`、`浙J3U1P3`（帧间抖动，未对准时 `（未检测到）`，符合预期）。

---

## 2. 帧率结果（release 构建，滑动窗口 p50）

| 档位 | fps p50 | fps 范围 | frame ms | conv ms | infer ms | 样本 | 采样时视野内有车牌？ |
|---|---|---|---|---|---|---|---|
| 生产 det=CPU rec=NPU cls=CPU | **23.8** | 16.7 – 27.8 | 42.0 | 3.1 | 32.3 | 27 | 否（28/28 `count=0`） |
| **全 NPU** | **34.5** | 30.3 – 38.5 | 29.0 | 8.5 | **12.7** | 20 | 否（20/21 `count=0`） |
| 全 GPU (ncnn-Vulkan) | 12.8 | 12.1 – 38.5 | 78.0 | 7.2 | 53.9 | 8 | 否（9/9 `count=0`） |

**全 NPU 档最快（34.5 fps）**，GPU 档最慢（12.8 fps）—— Vulkan 在麒麟 8020 上对小模型的 dispatch 开销压不过 NPU（与 ADR-008 的结论一致）。

> ⚠️ **上表口径（重要）**：三段采样的**绝大多数帧都没有车牌在视野内**（`count=0`），此时流水线只跑检测器就返回，`rec`/`cls` 根本没执行。所以这些 fps 是**「检测器 + 取帧」的成本**，**不是完整三模型流水线的 fps**。
>
> 唯一一帧含车牌的 NPU 样本（`count=1`）给出了完整链路的真实数字：
>
> | 阶段 | 仅检测 | 检测+识别+分类 |
> |---|---|---|
> | infer | 12.7 ms | **19.9 ms** |
> | 分段 | — | `det 12.67 \| 0.58 \| rec 4.34 \| cls 2.32` |
>
> 即**完整流水线约 +7 ms**（rec 4.3 + cls 2.3 + 检测后处理）。**"对准车牌时的 fps 必然低于上表"**，实测同会话在生产档看到过 16.7 fps 的低点。

> ⚠️ 口径说明之二：生产档在同一会话内曾测到 33–38 fps，本表 23.8 是另一段采样的结果。手机存在热/负载波动，**同一档位跨时段可比性有限**，档位间的相对关系才是结论。

---

## 3. 瓶颈定位：两个真实缺陷

### 3.1 取帧转换在 JS 层，占掉一帧的大头

分段计时（`split`）显示：**`buildPixelMap`（NV21→RGBA 建 PixelMap）占 16–30 ms**，`pm.rotate` 再占 5–8 ms，而推理只要 10 ms。

```
FRAME n=60 fps=22.73 split=0|1|30|5|1     ← 取帧|组件|建PixelMap|旋转|读像素
STAGE n=60 conv=... infer=11.9
```

**修法**：把 NV21→RGBA **连同 90° 旋转**搬进 C++（`LprNv21ToRgba`，`lpr_pipeline.cpp`），并新增合并入口 `lpr.cameraFrameAsync`，一次调用做完「转换+旋转+推理」，顺带省掉两次跨语言拷贝。

效果：**conv 21–38 ms → 2–8 ms**；生产档 fps **15 → 23.8（同会话内曾到 35）**。

### 3.2 `buildMode=debug` 把 native 编译成了 `-O0`

`entry/.cxx/.../debug/arm64-v8a/build.ninja` 的 `FLAGS` 是：

```
... -O3 -D__MUSL__ -O0 -g -fno-limit-debug-info ...
```

`cppFlags: "-O3"`（`entry/build-profile.json5`）**被 debug 的 `-O0` 覆盖**，因为它在命令行更靠后。改 `buildMode=release` 后是 `-O3 ... -O2 -DNDEBUG`，同一段代码 infer 从 ~30 ms 降到 ~10 ms（NPU 档）。

> 这条对**所有**既有基准数字都成立：`bench30_matrix_20260918.log` 与 `E2EMATRIX` 的历史数据都是 debug 构建下测的。**它们内部可比（同一构建），但不能与 release 数字直接并列。**

### 3.3 档位切换按钮实际点不动（已修）

`switchGear` 原本写成 `if (this.gear === g || this.busy) return;` —— 30 fps 下 `busy` 几乎恒为真，点击被静默吞掉。改成置 `switching` 标志、由帧循环让路（切档期间丢帧，避免新 mode/slot 配旧会话）。

### 3.4 GPU 档每帧失败（已修）

`infer 失败: no cached model (load first)` —— 检测器的 ncnn 模型是 native 侧的**全局单例**（`g_param`/`g_bin`），不是槽位，必须由 `lpr.ncnnLoadAsync` 装载；相机页此前只装了 rec/cls 两个槽位。补上后 GPU 档正常出结果。

---

## 4. 边界（不吹）

1. **GPU 档的"GPU"只对 rec/cls 成立**：det 走 ncnn-Vulkan（真 GPU），但 MS 会话仍以 CPU 占位；MS Lite 的 GPU 档是编译期判否后静默回落 CPU（ADR-012）。
2. **帧格式是 NV21 而非 JPEG**：本机 24 个预览档全是 `YUV_420_SP(1003)`，一个 JPEG 都没有。取帧按 profile 声明的格式选，实际交付格式由 `ImageReceiver` 决定。
3. **`getSupportedFrameRates()` 返回空**（`n=0`），无法声明"相机上限"。实测能到 62 fps，说明流本身高于 30，但这条**没有官方口径支撑**。
4. **未做精度验收**：相机读到的车牌串与静态图基准（`苏ED5172`）不是同一张图，不能据此声称精度。速度与功能已验，精度未验。
5. **丢帧是有意的**：`busy` 时新帧直接 release 不排队 —— 排队会让 fps 变成"输入帧率"的假象。
6. 🔴 **上表 fps 不是"完整流水线 fps"**：采样时视野内没有车牌，`count=0` 让 rec/cls 被跳过。**对外引用这三个 fps 必须同时说明这一点**，否则等于拿"只跑了检测器"的数字冒充三模型流水线。完整链路（含车牌）NPU 档实测 `infer=19.9 ms`，仅一帧样本，**不足以给出 fps p50**。
7. **`count=0` 的两种可能未区分**：既可能是视野内真没车牌，也可能是检测器在该分辨率/角度下没检出。本轮**没有做"同一场景有车牌 vs 无车牌"的对照采样**，所以不能说"检测器在这些帧上是对的"。

---

## 4.1 相机出帧上限：基准档实测（本轮新增，决定性）

**动机**：`arrive ≈ done` 时无法区分"相机只给这么多"和"我们算得慢导致队列满、相机跟着降速"。唯一干净的测法是**完全跳过推理**。

**做法**：`CameraPage.ets` 新增第 4 档 `GEAR_RAW = 3`「基准：只取帧不推理」—— `consume()` 里拿到帧、`release()`、直接返回，**不调 `cameraFrameAsync`**。按钮行由 `ForEach(GEAR_NAMES)` 渲染，第 4 个按钮自动出现。

**实测**（release，nova 14 Pro，连续 5 个结算窗口）：

**⚠️ 2026-09-18 追加更正（场景相关，本条推翻「固定 20 fps」的说法）**

上表那组 19.9 fps 是在**笔记本屏幕偏暗的场景**下测的。同日在**明亮场景**（笔记本屏幕显示亮色车牌图、取景框对着发光屏）复测同一档位：

| 场景 | arrive | done | dropped_in_win | 结论 |
|---|---|---|---|---|
| 暗（笔记本屏偏暗/待机） | 19.92–20.06 | 19.92–20.06 | 0 | 相机给 ~20 fps |
| **亮（屏幕显示亮色车牌图）** | **26.92–27.12** | **26.92–27.12** | **0** | 相机给 **~27 fps** |

→ **相机出帧率是场景相关的**：暗光下自动曝光会拉长曝光时间（帧间隔变长）→ 掉到 ~20 fps；光照充足时能给到 **~27 fps**。所以「相机固定上限 20 fps」是**错的**，正确表述是「**相机出帧率随曝光时间变化，实测范围 20–27 fps**」。

**推论也要跟着改**：`arrive=27 / done=13` 的差额（有车牌时每窗口丢 27–30 帧）**不是相机限制**，而是**推理真的跟不上** —— 此时瓶颈回到推理侧，还有优化空间（`det=CPU` 的 12.7 ms 是大头）。

**原实测**（暗场景，release，连续 5 个结算窗口）：

| 窗口 | arrive | done | dropped_in_win |
|---|---|---|---|
| 1 | 19.94 | 19.94 | 0 |
| 2 | 20.03 | 20.03 | 0 |
| 3 | 19.92 | 19.92 | 0 |
| 4 | 19.92 | 19.92 | 0 |
| 5 | 20.06 | 20.06 | 0 |

**结论**：

> **相机流在暗场景下给 ~20 fps**（帧间隔 50 ms）。整条推理链路（`conv 2–8` + `det 12.7` + `crop 0.58` + `rec 4.34` + `cls 2.32` ≈ **20 ms**）只占一半预算。
>
> 所以 `arrive=20 / done=15–17` 的差额**不是我们的问题**，是推理开销吃掉了部分帧间隔。而先前报的 `34.5 fps / 55 fps` 是 `1000 / p50(frameMs)` 算出的**服务时间理论值，不是吞吐**，**不能对外当 fps 引用**。

**因此"帧率还是低"的答案是**：暗光下 20 fps 是**相机预览流的限制（曝光时间）**；**光照充足时相机能给 ~27 fps，此时瓶颈回到推理侧**（`det=CPU 12.7 ms` 是大头）。要继续提速有两条路：① 改取帧路径（更高帧率 profile / `VideoOutput` 替代 `ImageReceiver`）；② 在明亮场景下把 `det` 也挪到 NPU（当前生产档 `det=CPU`）。

**口径更正**：本节之前所有 `fps p50` 数字（生产 23.8 / NPU 34.5 / GPU 12.8）**均为理论值**，仅可用于比较各档位的**相对开销**，**不可作为吞吐引用**。真实吞吐 = **到达率/完成率**，实测 20 fps（暗）～ 27 fps（亮）。

---

## 4.2 完成率腰斩缺陷 + 三加速器最优分配（2026-09-18 追加，release 构建）

### 4.2.1 真缺陷：完成率被"干等下一帧"砍掉一半

**症状**：`arrive ≈ 24.0 fps` 但 `done ≈ 12.5 fps`，每 2 s 窗口丢 24–25 帧 —— 正好一半。

**根因**（不是推理慢）：`listenFrames` 的丢帧分支写了
`recv.readLatestImage().then(img => img.release())`，把接收器缓冲里的帧**抽走释放**。
于是当前帧算完后，缓冲是空的，只能**干等下一次 `imageArrival`**。
单帧工作耗时（24–47 ms）只要略微超过帧间隔（1000/24 ≈ 41 ms），
完成率就被砍到到达率的一半。

**修复**：busy 期间**不读不释放**，留一帧在缓冲里（capacity 8 足够，最多积 2 帧）；
新增 `finishFrame()`，一帧处理完**立刻接手**缓冲里最新那帧。

| | 修复前 | 修复后 |
|---|---|---|
| arrive | 24.0 fps | 24.0 fps |
| **done** | **12.5 fps** | **23.9–24.1 fps** |
| 每窗口丢帧 | 24–25 | **0** |

持续完成率（43.5 s，n=30→990）：**12.5 → 22.1 fps（1.77×）**。

**附带修正**：丢帧计数器语义。加了"接手"之后，busy 期间到达的那帧**不再真丢**
（它成为缓冲里最新的一帧被处理）。旧写法每遇 busy 就 +1，多算了约 1368：
日志 `n=2760` 完成，读数卡却写"到达 2990 / 丢帧 1598"（1390 ≠ 2760）。
现在只有**已被停帧、又被新帧顶掉**的那帧才算丢 → 恒有 `到达 = 完成 + 丢帧`。

### 4.2.2 相机上限的干净证明（同会话、同场景）

| 档位 | arrive | done | 丢帧 |
|---|---|---|---|
| **基准档**（只取帧、零推理） | 14.94–15.09 | 14.94–15.09 | 0 |
| **生产档**（det+rec+cls 全跑） | 14.68–15.32 | 14.68–15.32 | 0 |

两者**完全相等** → 相机自己就只给 15 fps，全流水线一帧不落地跟上了。
**推理侧已彻底不是瓶颈。**

> 🚫 **2026-09-18 本节结论撤回（ADR-015 §2），待复测**。撤回理由三条，都可核对：
> ① 所谓"基准档 = 只取帧、零推理"**仍然支付整帧 NV21→RGBA 转换** —— 它测的是
> 「取帧+转换上限」，不是「相机上限」；② 同一份证据 §4.1 在**同一暗场景**报到达率
> `19.92–20.06 fps`，与本节的 `14.94–15.09` 互相矛盾（同场景同档位不可能既 20 又 15），
> 说明复现条件没有锁死；③ `frameMs = decodeMs + inferMs` 不含 ArkUI 重渲染与 napi 往返
> （`record()` 每帧写 6–8 个 `@State`，每次触发整页重渲染），因此由 `frameMs` 反推的一切
> 数字系统性乐观。
> 结案路径：新增「只数帧、完全不 `readLatestImage`」档 + 固定片元回放台。
> 在此之前，下文 §4.2.6 的档位对比与"det=CPU 12.7 ms 是大头"两条**同时挂起**。

### 4.2.3 "关掉显示路会不会更快" —— 实测：不会

现在是双路预览（显示路 960×960 + 分析路 640×480），基准档测的 15 fps 仍带着显示路，
所以"双流互抢"没被排除。摘掉显示路（`SKIP_PREVIEW_DISPLAY`）复测：

| | arrive（生产档） | arrive（基准档） |
|---|---|---|
| 显示路 ON | 14.7–15.3 | 14.9–15.1 |
| 显示路 OFF | 14.8–15.4 | 14.9–15.1 |

**无差别 → 双流互抢被排除。** 15.00 fps（抖动仅 ±0.1）是相机在暗场景把帧率
**砍半换曝光**（30→15）的典型行为，是传感器决策，不是我们的开销。开关保留（默认关），
供下次一分钟内复查同一假设。

### 4.2.4 帧率协商：能查、能设，但设了更慢

`PreviewOutput.getSupportedFrameRates()` **必须等 `session.start()` 之后**才非空
（建流前返回 `n=0`，一度让人以为设备不支持）。设备报 **`[1-30, 60-60]`**。

| 设置 | arrive |
|---|---|
| 不设（相机自动） | 22.1 fps |
| 强制 `30-30` | 18.1–19.4 fps |
| 强制 `60-60` | 18.5 fps |

⚠️ 但这三次测量**跨时间、场景在漂移**（同一设置下后来复测只有 15 fps），
所以**不能据此断言强制帧率更差** —— 只能说**没有观察到可靠收益**。
结论：保持不设，只保留查询日志作为能力证据。

### 4.2.5 三加速器最优分配（加速器矩阵，release）

`加速器矩阵` 探针逐模型 × 逐后端跑，只认 `LANDED=`（实际落点）：

| 模型 | 请求 | **实际落点** | p50 |
|---|---|---|---|
| det head (`y5fu_320x_head_fp32.ms`) | `nnrt` | **NPU** | **6.46 ms** |
| 同上 | `gpu` / `kirin` / `cpu` | **CPU**（静默回落） | 7.13–7.26 ms |
| **rec** (`rpv3_mdict_160_r3.ms`) | `nnrt` | **NPU** | **3.91 ms** |
| 同上 | `gpu` / `kirin` / `cpu` | **CPU**（静默回落） | **9.0 ms**（2.31×） |
| cls (`litemodel_cls_96x_r1_fp32.ms`) | `nnrt` | **NPU** | 0.96 ms |
| 同上 | `cpu` | CPU | **0.87 ms** ← CPU 更快 |
| cls (`litemodel_cls_96x_r1.ms`，非 fp32) | `nnrt` | **CPU**（回落） | 0.87 ms |

**两条硬结论**：
1. **MS Lite 的 `gpu` / `kirin` 后端是假的** —— 全部静默回落 CPU。真正的 GPU 通路
   只有 ncnn-Vulkan（A9：小图无加速，rec 52–72 ms，远慢于 NPU 的 7.5 ms）。
2. **NPU 只在 rec 上有大收益（2.31×）**；det 小幅受益；cls 反而是 CPU 更快。

### 4.2.6 档位 A/B：默认档改为「全 NPU」

同会话背靠背，release 构建，真机 nova 14 Pro：

| | 生产档 `det=CPU` | **全 NPU `det=NPU`** |
|---|---|---|
| det p50 | 24.86 ms | **12.15 ms**（2.05×） |
| crop p50 | 0.53 | 0.81 |
| rec p50 | 6.97 | 7.51 |
| cls p50 | 1.24 | 1.98 |
| **infer p50** | 34.6 ms | **23.0 ms**（1.50×） |
| **端到端** | 36.6 ms | **29.4 ms**（1.24×） |
| 抖动 | 22.4–104.3 ms | **19.1–26.6 ms** |
| 识别结果 | `皖A40675` ×21 | `皖A40675` ×10（**一致，零翻字符**） |

**纠正两条历史判断**：
- ❌「det 必须留在 CPU，因为 NPU 拒收」→ 矩阵实测 det head 在 `nnrt` 上
  `LANDED=NPU`（6.46 ms，比 CPU 的 7.13 ms 还快）。
- ❌「det 与 rec 同抢 NPU 会互相拖累」→ rec 只慢 0.5 ms，det 省 12.7 ms。

#### ⚠️ 2026-09-18 晚间更正：**默认档不能改，`det=NPU` 会翻字符**

改动一度把默认档切到 `GEAR_NPU`（`det=NPU`），理由是上面这组 A/B 的
速度优势（infer 34.6 → 23.0 ms）加上「两档都读对 `皖A40675`」。
**这个理由是错的，已撤回** —— 用**期望串验收**（不是「碰巧读对一个相机样本」）复验：

`AUTO_SELFTEST=true` + release 构建 + 黄金图 `assets/samples/hlpr-test.jpg`
（期望串 `苏ED5172`），`backendProbe` / `e2eMatrix` 逐字符比对，
原始日志 `A15-expected-string-matrix-release-20260918.log`：

| 角色 | 后端 | 实际落点 | 期望串比对 |
|---|---|---|---|
| **det** | `cpu` / `gpu` / `kirin` | CPU | **111 ✅** |
| **det** | **`nnrt`** | **NPU** | **000 ❌ 全翻**（`苏ED5172` → `苏E05172`，D→0） |
| **rec** | `nnrt` | NPU | **111 ✅** |
| rec | `cpu` / `gpu` / `kirin` | CPU | 111 ✅ |
| cls | 全部 | CPU | 111 ✅ |
| cls-fp32 | `nnrt` | NPU | 111 ✅ |

同一次运行的 `backendProbe`（4 条检测路径 × 3 次）逐条一致：
`ms-cpu` 111 / **`ms-nnrt` 000** / `ncnn-cpu` 111 / `ncnn-vulkan` 111。

**结论（决定性）**：
- **`det=NPU` 是整张矩阵里唯一会翻字符的落点**，release 构建上 3/3 + 45 条历史，
  全部 `match=0`。机制：NPU 的 0.08% 数值漂移经 anchor（≤433）放大成 ≈10 px 位移，
  矫正对 1 像素都敏感。
- 因此**最优且安全的分配就是生产档 `det=CPU / rec=NPU / cls=CPU`** ——
  它同时是「唯一安全」和「已是三个角色各自最优」的配置：
  det 不能上 NPU（翻字符）；rec 必须上 NPU（2.31×）；cls 留 CPU 更快且非 fp32 本就上不去。
- **「全 NPU 档」那 1.5× infer 提速不可用** —— 是拿正确性换的。该档保留为**测速用档**，
  不作默认。
- **默认档已改回 `GEAR_PROD`**，装机复测 `det=CPU rec=NPU cls=CPU`、
  `arrive = done = 23.0–23.2 fps`、丢帧 0。

**方法论教训（比结论更重要）**：一次相机 A/B 读到两档同串 `皖A40675`（21 次 vs 10 次），
看起来"精度无损"。但那是**单个随机样本**，而全矩阵在黄金图上 45+3 次全部翻字符。
**换档/换模型后的验收对象必须是期望串，而不是"碰巧读对一个"。**
把「相机上碰巧读对」当成精度证据，是拿演示运气冒充验收。

### 4.2.7 边界（不吹）

- 4.2.4 的帧率三档是**跨时间**测的，场景（光照）在漂移 → 不可作为强制帧率的优劣结论。
- A/B 的识别一致性只覆盖**当前样本**（`皖A40675`，各档 10–21 次采样）；
  **该证据不足以支持"det 可以上 NPU"**，见上面的晚间更正 ——
  期望串验收在 release 上给出 000（全翻），默认档已撤回生产档。
- `infer` 10.6–11.7 ms 是**当前明亮场景**（相机 24 fps）下的数；暗场景相机降到 15 fps 时，
  同档位曾测到 `infer` 19–27 ms（NPU 调用开销占比升高）。
- 4.2.5 的 p50 是**单模型微基准**（`benchAsync`，含 warmup），不等于流水线内该段的耗时
  （流水线内 det 段还含 letterbox / decode / NMS）。

---

## 5. 复现

```bash
cd /c/Users/26671/lpr-harmony
bash build.sh assembleHap --mode module -p product=default -p buildMode=release --no-daemon

HDC="D:/IDE/DevEco_Studio/sdk/default/openharmony/toolchains/hdc.exe"
cd LprDemo/entry/build/default/outputs/default
"$HDC" shell "aa force-stop com.shusen.lprdemo"
"$HDC" install entry-default-signed.hap
"$HDC" shell "hilog -r"
"$HDC" shell "aa start -a EntryAbility -b com.shusen.lprdemo"
sleep 4
"$HDC" shell "uitest uiInput click 715 540"     # 「相机实时识别」
sleep 25                                        # 采集
"$HDC" shell "hilog -x" | grep LprCamera
```

档位按钮（`uitest uiInput click`）：生产 `658 2595` · 全 NPU `658 2595`（同上行左）· 全 GPU `913 2595`。
坐标取自 `uitest dumpLayout`，随布局变化需重取。

关键日志标签：`LprCamera`（页面）· `PIPE BACKENDS`（各模型实际落点，native 侧）。
