# 端侧加速的其它路线：CANN / HiAI Foundation / NNRt 全图（2026-09-18）

> 触发问题：**"还有别的办法吗？CANN 框架的呢？"**
> 结论先说：**CANN 这条路是真的、设备上也确实开着**（库能加载、设备能枚举、版本可读），
> **但它不吃我们手上的模型格式** —— 门槛不在权限，在 `.om` 转换器。
> 而真正值得记的一条是：**我们现在的 NPU 通道，本来就是 CANN/NNRt 的官方底座**（下面 §4 有逐字证据）。

---

## 1. 先划清范围：GPU 只有一条路，而且已经走完了

| 路线 | 判定 | 依据 |
|---|---|---|
| **ncnn Vulkan** | ✅ 已实现，三模型全通 | **A9**：det/rec/cls × ncnn-Vulkan 全部 `match=1`，`effectiveVulkan=1` + 枚举到 `Maleoon 920C` |
| MindSpore Lite 的 GPU 档（OpenCL） | ❌ **编译期判否** | ADR-004 §4.2：`IsValid# GPU is not supported` |
| NNRt 的 GPU | ❌ **不存在** | **A10 实测**：NNRt 枚举出 2 张设备，**两张都是 `ACCELERATOR`**，没有任何 GPU 设备 |
| MNN / TNN 的 OpenCL 后端 | ❌ 不适用 | Maleoon 未向该路径暴露 OpenCL（ADR-004）；且 MNN 的 HIAI 后端官方 wiki 标注「适用于 **Android** 系统，Kirin 芯片」 |

**所以 GPU 侧没有"别的办法"了** —— Vulkan 是唯一通路，已经做过，结论还是
「通路成立、数值保真，但小图不加速」（A9 §4）。再去试别的 GPU 框架，只会在同一块
Maleoon 上重复同一组数字。

## 2. NPU 侧有四条路，只有一条不需要额外工具链

| # | 路线 | 需要什么 | 今天的状态 |
|---|---|---|---|
| **A** | **MindSpore Lite + NNRT delegate**（项目现状） | 只需 `.ms` 模型（MS 转换器可在 PC 侧跑） | ✅ **已跑通**：rec 拿 2.2–3.0×（A5/A6） |
| **B** | **NNRt 直连**（`libneural_network_runtime.so`，**标准 NDK，无需 HMS**） | `.om` 离线模型，或用构图接口在线建模 | ⚠️ 库/设备都可用；**`.ms` 被拒**（A10） |
| **C** | **HiAI Foundation / CANN Kit**（`libhiai_foundation.so`，HMS 侧） | 同 B，外加显式设备序、调优缓存、单算子、AIPP | ⚠️ 库能加载、CANN 版本可读；离线入口同样要 `.om` |
| **D** | 三方框架的华为 NPU 后端（MNN/TNN/Paddle-Lite 的 HIAI） | 官方确已对接 CANN DDK，但定位是 **Android** | ❌ 对 HarmonyOS NEXT 不适用 |

**B 与 C 的关系**：C 是 B 之上的增强层（同一份 `neural_network_core.h` 的编译/执行对象，
HiAI 额外提供 `HMS_HiAIOptions_SetModelDeviceOrder(compilation, HiAI_ExecuteDevice[], n)`
——**显式指定硬件顺序**，这正是本项目一直想要的"硬落点证据"手段）。

## 3. 真正的门槛：`.om` 转换器（不是权限，是工具链）

官方定义（NNRt 头文件原文）：

> *"Offline model is a type of model that is offline compiled by **the model converter provided by a device vendor**. So that the offline model can only be used on the specified device."*

对应工具是 **OMG（offline model generator）**，官方文档写明：

> *"you can use the offline model generator (OMG) to convert Caffe, TensorFlow, ONNX, MindSpore models into offline models (OMs) … The OMG is located in the `tools/tools_omg` directory of the tools (**obtained from Tool Download**). It can run on **64-bit Linux**."*

三条硬约束叠在一起，就是今天走不通的原因：

1. **要华为开发者账号**去 Tool Download 取 DDK/CANN 工具包；
2. **要 64 位 Linux**（本机是 Windows → 得靠 WSL2）；
3. **`.om` 是版本敏感的**：它绑定转换时的 CANN 版本。设备侧 CANN 版本我们已读出来是
   **`108.631.120.010`**；版本不匹配会被 `HMS_HiAICompatibility_CheckFromBuffer` 判为
   `INCOMPATIBLE` —— 我们手上**没有** `.om`，所以无法验证这条链路的终点。

## 4. 🔑 本轮最重要的发现：现有 NPU 通道**就是**官方底座

A10 探针从 NNRt 枚举出的设备名，与 MS Lite 日志里落到 NPU 时的设备名**逐字相同**：

```
NNRt  枚举  ：d1 = NPU_ohos.boot.hardware.kirin8020_v2_0   (type=ACCELERATOR)
MS Lite 日志：model built on NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0
```

也就是说：**MindSpore Lite 的 NNRT delegate 走的就是 NNRt，而 NNRt 在麒麟上落到 HiAI。**
这条证据把"我们到底有没有真的用上 NPU"从"日志里的字符串"升级成"设备枚举出来的实名设备"，
**两条独立来源互证**。

连带两个推论：
- 想靠"直连 NNRt"把 NPU 再快一截，**上限很小** —— 省掉的只是一层胶水，而 rec 段已经 7.5 ms。
- 项目此前 ADR-004 记的「`kirin` 档不可用（缺 `libhiai_ir_infershape.so`）」**与本轮不矛盾**：
  那条说的是 MindSpore Lite **内部**的私有档位；NDK 对外的 `libneural_network_runtime.so` /
  `libhiai_foundation.so` 是另一套（都躺在 `/system/lib64/ndk/`，都验过能 `dlopen`）。
  文档里要把这两件事分开写，否则看起来像自相矛盾。

## 5. 不买转换器的替代路线：在线构图

CANN Kit 的官方能力页明确写着：

> *"支持通过构图接口，将 Caffe、TensorFlow、**ONNX** 等模型**在线构图**为 HarmonyOS 内部统一格式的 IR 图。"*

NNRt 侧对应的是 `OH_NNModel_AddTensor` / `OH_NNModel_AddOperation` +
`OH_NNCompilation_Construct(model)`（而不是 `...WithOfflineModel*`）。官方 NNRt 简介也印证了
这条路是给推理框架用的：

> *"在线构图：AI 推理框架需要调用 NNRt 的构图接口将推理框架的模型图转换为 NNRt 内部模型图。"*

**代价**：要自己写一个 ONNX → NNRt 算子的映射器。规模（实测）：
ONNX 节点 **det 416 / rec 444 / cls 103**（转 ncnn 后对应 218 / 252 / 78 层），
而 NNRt 的算子枚举是固定集合。这是"几天"量级的工作，不是"一键转换"。

## 6. 建议（按性价比排序）

| 优先级 | 做什么 | 理由 |
|---|---|---|
|  首选 | **不追新路线，把 A 路线的口径做扎实** | 现状已是官方底座（§4），性能上再无便宜可捡；缺的是 **30 次 p50 正式基准 + thermal 标注**，那是论文真正要的 |
| ⭐⭐ 想真正拿下 CANN | **WSL2 + 开发者账号取 DDK/OMG → 转一个模型试 `.om`** | **终点已证明可达**（下文 §8）：探针里 `compat`/`build_rc`/`run_rc` 三关与 IO 自省**全部写好且实测通过**，拿到自己转的 `.om` 只是换一个文件 |
| ⭐⭐⭐ 想彻底摆脱转换器 | **写 ONNX→NNRt 构图映射器** | 学术上最"爽"（自研 delegate），但工作量最大，且性能上限已知不高 |
| ❌ 不建议 | 换框架（MNN/TNN）找 GPU 加速 | 同一块 Maleoon、同一条 Vulkan/OpenCL 结论；MNN 的 HIAI 还是 Android 定位 |

## 7. 复现命令（本轮新增的探针）

App 内「探针控制台」→ **CANN/NNRt 探针** 按钮，或等自检链跑到 `CANN BEGIN`。
返回的关键字段：`cann_version` / `device_count` / `d0,d1`（名字+类型）/
`hiai_compat_code`（0=compatible,1=incompatible）/ `construct` / `set_device_rc` /
`build_rc`（0=SUCCESS）/ `executor` / `in0,out0`（名字/类型/形状）/ `run_rc` / `run_ms_each`。

代码：`entry/src/main/cpp/nnrt_probe.{h,cpp}`（dlopen + 逐级回退，失败带 `dlerror`）。

## 8.  后续更新（同日）：终点已证明可达 —— 用真 `.om` 全绿跑通

上面 §3 的结论"要过这一关必须拿到 `.om`"**当天就被验证了**：从华为官方公开的
HiAI Foundation codelab（gitee）取到一份**真正的厂商 `.om`**（2,496,459 B，`IMOD` 魔数，
Caffe/SqueezeNet ImageNet 分类），喂进**同一条** NNRt 入口：

| | `.om`（厂商产出） | 我们的 `.ms` ×3 |
|---|---|---|
| `hiai_compat_code` | **0 = COMPATIBLE** | 1 = INCOMPATIBLE |
| `build_rc` | **0 = SUCCESS** | 1 = FAILED |
| `executor` | ok（IO `[1x3x227x227]` → `[1x1000x1x1]`） | — |
| `run_rc` | **0（推理成功）** | — |
| `run_ms_each` | **2.186 \| 0.944 \| 0.918**（稳态 ~0.92 ms） | — |
| `out_sum` | **1.0005**（1000 维和为 1 → 合法 softmax，真跑完前向） | — |

**所以答案是：有办法，而且已经跑通。** 门槛**只是格式**，不是权限、不是能力。
唯一缺口变成一句话：**我们缺把 ONNX 转成 `.om` 的 OMG**（华为账号 + 64 位 Linux + 与设备
CANN `108.631.120.010` 对齐）。细节与边界见 `_evidence/A11-cann-om-end-to-end-20260918.md`
与 `docs/adr/ADR-013-cann-path-proven-with-vendor-om.md`。
