# A10 · CANN / HiAI / NNRt 真机探针：设备侧到底开着哪些通路

- **日期**：2026-09-18 12:10–12:35
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **动机**：回答「NPU/GPU 还有没有别的办法（尤其 CANN）」，把猜测换成设备自证
- **证据**：`_evidence/cann_probe_20260918.log`（全量 hilog）
- **代码**：`entry/src/main/cpp/nnrt_probe.{h,cpp}` + `Index.ets` 的 `cannProbe()`
  （App 内「探针控制台 → CANN/NNRt 探针」，或自检链末尾自动跑）

---

## 1. 设备上真实存在的 NDK 库（逐一 stat 过）

`/system/lib64` 目录不可列（permission denied），但**单文件可 stat**，于是逐个点名：

```
/system/lib64/ndk/libneural_network_runtime.so   1,518,880   ← NNRt 运行时
/system/lib64/ndk/libneural_network_core.so         95,936   ← NNRt 核心（OH_NNCompilation_*）
/system/lib64/ndk/libhiai_foundation.so            195,440   ← CANN Kit / HiAI Foundation
/system/lib64/ndk/libmindspore_lite_ndk.so         154,560
```

syscap 自证（`param get`）：

```
const.SystemCapability.AI.HiAIFoundation       = true
const.SystemCapability.AI.NeuralNetworkRuntime = true
```

**即：CANN Kit 与 NNRt 都在标准 NDK 面向上对普通应用开放，不需要合作伙伴白名单。**
（注意：这与 ADR-004 §4.2 记的「`kirin` 档缺 `libhiai_ir_infershape.so`」不是同一件事 ——
那条指的是 MindSpore Lite **内部**的私有档位，这里是 NDK 对外库。）

## 2. 探针结果一：枚举（库 + 版本 + 设备）

```
CANN ENUM ok=1;probe=nnrt+hiai;
  nnrt_runtime=soname; nnrt_core=soname; hiai_foundation=soname;
  cann_version=108.631.120.010;
  getAllDevices_rc=0; device_count=2;
  d0=id=8987859593747354028,  name=HIAI_F,                             type=ACCELERATOR
  d1=id=13306020061847651998, name=NPU_ohos.boot.hardware.kirin8020_v2_0, type=ACCELERATOR
```

- 三个库都能按 **soname** 直接 `dlopen`（说明在应用的默认搜索路径上，不用绝对路径兜底）。
- **CANN 版本号读到了**：`108.631.120.010`（这是判断 `.om` 兼容性的基准）。
- NNRt 上有 **2 张设备**，**两张都是 ACCELERATOR**：
  - `HIAI_F` —— HiAI Foundation 本体（设备 id 在枚举里排第一，`id[0]`）；
  - `NPU_ohos.boot.hardware.kirin8020_v2_0` —— **麒麟 8020 的 NPU 实名**。
- ⚠️ **没有任何 GPU 设备** —— NNRt 侧不存在 GPU 通路（与 ADR-004 的 OpenCL 判否互相印证）。

## 3. 探针结果二：把 `.ms` 喂给 offline model 入口

对三份生产模型各试一次（`deviceIndex=0` → `HIAI_F`）：

| 模型 | 字节数 | `hiai_compat_code` | `construct` | `set_device_rc` | `build_rc` |
|---|---|---|---|---|---|
| `y5fu_320x_head_fp32.ms`（检测） | 1,847,336 | **1 = INCOMPATIBLE** | ok | 0 | **1** |
| `rpv3_mdict_160_r3.ms`（识别） | 4,480,304 | **1 = INCOMPATIBLE** | ok | 0 | **1** |
| `litemodel_cls_96x_r1.ms`（分类） | 817,432 | **1 = INCOMPATIBLE** | ok | 0 | **1** |

读法（三行都同构）：

- `hiai_compat_code=1` —— **HiAI 自己判定这段字节不是它能吃的模型**；
- `construct=ok` —— NNRt 的 `ConstructWithOfflineModelBuffer` 构造时**不做格式校验**，所以构造成功不代表格式对；
- `set_device_rc=0` —— 显式选设备**成功**（`picked_device=8987859593747354028`，即 `HIAI_F`）；
- `build_rc=1` —— 编译失败（`OH_NN_ReturnCode` 的 1 = `OH_NN_FAILED`）。
- `destroyed=1` —— 编译对象已销毁，没漏。

**结论：NNRt/HiAI 的离线模型入口不认识 MindSpore Lite 的 `.ms`**，
与官方文档「offline model 由设备厂商提供的转换器生成」以及 HiAI 的兼容性检查**三方一致**。
要过这一关，必须拿到 **OMG** 转出的 `.om`（见 research 文档 §3：需华为账号 + 64 位 Linux + 版本对齐）。

### 一个自己踩到的坑（已修）

首版探针把 64 位 `deviceID` 用 `%d` 打印，输出成 `picked_device=1214632364`（截断值）。
**代码里传给 `SetDevice` 的始终是完整的 `size_t`**，只是打印错了 —— 但那会让证据不可信。
已加 `KvU64()` 用 `%llu`，重跑后 `picked_device=8987859593747354028` 与 `picked_device_all0` 一致。
> 教训：探针里凡是 `size_t` 句柄，落日志一律按 64 位打印。

## 4. 交叉验证：现有 NPU 通道 = 官方底座（本轮最有价值的一条）

同一次运行的日志里，两个独立来源给出**逐字相同**的设备名：

```
NNRt 枚举（本探针）      : NPU_ohos.boot.hardware.kirin8020_v2_0
MS Lite 落到 NPU 时打印  : model built on NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0
```

→ MindSpore Lite 的 NNRT delegate **走的就是 NNRt**，NNRt 在麒麟上落到 HiAI。
**"我们到底有没有真的用上 NPU"这个问题，从此不靠解析日志字符串，而有设备枚举实名互证。**

## 5. 对生产路径的影响：零

同一次运行、同一条自检链，末尾新增 CANN 探针后仍然全绿：

| 检查项 | 结果 |
|---|---|
| `E2EMATRIX END` / `NCNNSLOT END` / `CANN END` | ✅ 三个都到 |
| 端到端 | ✅ `苏ED5172`，`cropSum=2773473`（**第五次连续同值**） |
| watchdog / 崩溃 | **0 条** |
| 进程 | ✅ 存活 pid 37291（跑完 330 s 仍在） |

探针只做 `dlopen` + 枚举 + 一次 construct/build/destroy，**不注册任何会话、不改引擎状态**，
跑在既有推理线程上（遵守 A7 的线程纪律）。

## 6. 边界

- `build_rc=1` 只说明**这份字节**被拒；**没有**证明 `.om` 一定成功（手上没有 `.om` 可测）。
- `cann_version=108.631.120.010` 是设备侧版本；它与 DDK/OMG 版本的对应关系未验证。
- 只试了 `deviceIndex=0`（`HIAI_F`）；`d1`（麒麟 NPU）未单独试 —— 两者都是 ACCELERATOR，
  且 `HIAI_F` 是枚举首位，预期行为相同，但**未实测**。
- 没有做单算子通路（CANN Kit 的 `hiai_single_op.h`）的试验 —— 那是另一条可能绕开 `.om`
  的路线，尚未评估工作量。
