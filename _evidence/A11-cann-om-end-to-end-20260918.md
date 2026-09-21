# A11 · CANN 端到端跑通：用厂商 `.om` 在麒麟 NPU 上真跑一次推理

- **日期**：2026-09-18 12:40–13:10
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **前置**：A10（CANN/NNRt 探针：库可用、设备可枚举、`.ms` 被拒）
- **证据**：`_evidence/cann_om_run_20260918.log`（全量 hilog）、`_evidence/cann_om_20260918.log`（首次编译验证）
- **代码**：`cpp/nnrt_probe.{h,cpp}`（本轮扩到"编译成功→建执行器→读 IO→真推理"）
- **一句话结论**：**CANN 这条路不是"没戏"，是"我们缺一个转换器"** ——
  同一次运行、同一台设备，`.om` 全绿跑通并测出稳态 **0.92 ms**，我们自己的 `.ms` 被拒。

---

## 1. 关键转折：找到了一份厂商产出的真 `.om`

来自华为官方公开的 HiAI Foundation codelab（gitee，`HarmonyOS_Codelabs` 组织）：

```
仓库 : harmonyos_codelabs/hiaifoundationkit-codelab-clientdemo-cpp
路径 : entry/src/main/resources/rawfile/hiai.om   → 落盘为 models/probe_hiai_imagenet.om
大小 : 2,496,459 B
魔数 : IMOD          ← HiAI 离线模型格式（对照：`IMOD` 是官方 .om 的头部标识）
配套 : labels_caffe.txt（999 行 ImageNet 类别）→ 判定这是 Caffe/SqueezeNet 227×227 图像分类
```

它存在的唯一目的是**证伪"通路不行"这个假说**：如果它被拒，说明设备侧根本没打开；
如果它通过，说明**门槛在格式而不在能力**。

## 2. 同一次运行的两组对照（决定性）

探针按顺序把 4 份模型交给同一条 NNRt 离线入口、同一张设备（`picked_device=8987859593747354028` = `HIAI_F`）：

| 模型 | 大小 | `hiai_compat_code` | `build_rc` | 结果 |
|---|---|---|---|---|
| **`probe_hiai_imagenet.om`** | 2,496,459 | **0 = COMPATIBLE** | **0 = SUCCESS** | ✅ **编译通过并跑出结果** |
| `y5fu_320x_head_fp32.ms` | 1,847,336 | 1 = INCOMPATIBLE | 1 = `OH_NN_FAILED` | ❌ |
| `rpv3_mdict_160_r3.ms` | 4,480,304 | 1 = INCOMPATIBLE | 1 = `OH_NN_FAILED` | ❌ |
| `litemodel_cls_96x_r1.ms` | 817,432 | 1 = INCOMPATIBLE | 1 = `OH_NN_FAILED` | ❌ |

**同一探针、同一设备、同一次运行** —— 差别只有文件格式。这就是"权限没问题、能力没问题、
只有格式不对"的干净对照。

## 3. `.om` 上的完整链路（编译 → IO → 真推理 → 计时）

```
hiai_compat_code = 0                                  HiAI 判为兼容
construct        = ok                                 NNRt 构造编译对象
set_device_rc    = 0                                  显式绑定设备成功
build_rc         = 0                                  ✅ 设备端编译成功
executor         = ok                                 执行器创建成功
in_count/out_count = 1 / 1
in0  = name=data,            dtype=FLOAT32, shape=[1x3x227x227]
out0 = name=output_0_prob_0, dtype=FLOAT32, shape=[1x1000x1x1]
in_bytes=618348  out_bytes=4000
run_rc           = 0                                  ✅ 推理执行成功
run_ms_each      = 2.186 | 0.944 | 0.918              ← 三次 RunSync 的墙钟毫秒
out_sum          = 1.0005                             ← 1000 个输出之和 ≈ 1.0
out_argmax       = 111  (val=0.1509)
executor_destroyed = 1 ; destroyed = 1                无泄漏
```

### 三个必须一起说的读法

1. **`out_sum = 1.0005` 是完整性校验，不是巧合。** 1000 维输出求和 ≈ 1.0，说明这是一条
   **合法的 softmax 概率分布** —— 即 NPU 真的跑完了完整前向，而不是返回未初始化缓冲。
   （对照：若只是"返回成功码"，`out_sum` 会是 0 或垃圾值。）
2. **计时口径**：`2.186 → 0.944 → 0.918 ms`。首轮高出的 ~1.27 ms 是**首次调用开销**
   （图落地/缓冲分配），与本项目在其它后端上反复观察到的"首轮含冷启动"一致。
   **稳态 ~0.92 ms** 是这台设备 CANN 侧的第一手数字。
3. **`argmax=111` 无意义 —— 输入是合成的。** 按项目纪律（ADR-004 §3.4）用确定性填充
   `x[j] = ((j×2654435761) mod 1000)/1000`，没有喂真实图片，所以类别不可解释。
   本轮的结论是**机制与时延**，不是精度。探针输出里也把这句话写进了 `out_note`。

## 4. 这条证据改变了什么

| 之前（A10 之后） | 现在（A11 之后） |
|---|---|
| 「CANN 通路**理论上**可用，但 `.ms` 被拒，终点未验证」 | 「CANN 通路**端到端实测可用**：编译 → IO → 推理 → 计时全绿」 |
| 缺口 = 「不知道设备收不收厂商模型」 | 缺口 = **只剩「我们怎么产出 `.om`」** |
| 无从谈性能 | 有了 NPU/CANN 侧第一个实测时延（稳态 0.92 ms） |

顺带确认了两件工程事实：
- **`OH_NNCompilation_SetDevice` 能显式绑定设备**（`set_device_rc=0`），设备 id 由
  `OH_NNDevice_GetAllDevicesID` 给出 → 这是比"解析日志推断落点"硬一档的落点证据手段。
- **IO 规格可以从运行时读出**（名字/类型/形状）—— 不需要看模型文件就能对接前后处理。

## 5. 仍未闭合的部分（别读过头）

1. **这份 `.om` 是"客人"，不是本项目的模型。** 它是 ImageNet 分类网络，与车牌识别流水线无关；
   它的角色是**机制证明**，不进任何生产路径、不出现在任何性能表里。
2. **我们的三个模型仍然上不了 CANN**，因为它们是 `.ms`。要变成 `.om` 需要 **OMG**
   （华为开发者账号 + 64 位 Linux + 与设备 CANN `108.631.120.010` 版本对齐）——手上没有。
3. **只有一份 `.om`**，无法据此外推"所有 `.om` 都能跑"（`.om` 是版本敏感的）。
4. 只有 3 次重复、无 thermal 标注，**不构成基准**，只是"能跑 + 量级"。
5. 未试在线构图（`OH_NNModel_*`）；未试 CANN Kit 的 `SetModelDeviceOrder` / 单算子通路。

## 6. 对生产路径的影响：零

| 检查项 | 结果 |
|---|---|
| 端到端 | ✅ `苏ED5172`，`cropSum=2773473`（**第六次连续同值**） |
| 三个阶段链（矩阵 / ncnn 槽位 / CANN） | ✅ 全部 `END` |
| watchdog / 崩溃 | **0 条** |
| 进程 | ✅ 存活 pid 51960 |

探针只 `dlopen` + 枚举 + 编译 + 3 次推理 + 逐级销毁，**不注册会话、不改引擎状态**，
跑在既有推理线程上（遵守 A7 的线程纪律）。HAP 因多带一份 2.5 MB 的 `.om` 从 63.7 → 66.2 MB。

## 7. 下一步（若要把这条线变成可用的第四个后端）

1. **取 OMG**（WSL2 + 开发者账号）→ 转我们自己的模型 → 本探针**改一行**即可验证
   （`compat` / `build_rc` / `run_rc` 三关全绿即算通过）。
2. 或者走 **NNRt 在线构图**（`OH_NNModel_AddTensor/AddOperation`）——不需要 `.om`，
   但要自研算子映射。规模：ONNX 节点 det 416 / rec 444 / cls 103。
3. 若真做出来，它才配被称为"第四后端"，届时需要与 `ms-nnrt` 做**同输入 L2 对照**
   （不能只比车牌串，要按 ADR-002 的保真方法学）。
