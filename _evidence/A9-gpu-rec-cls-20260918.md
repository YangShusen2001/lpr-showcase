# A9 · 让识别与分类也吃到 GPU —— pnnx 转换 + ncnn 槽位化

- **日期**：2026-09-18 11:40–12:05
- **设备**：nova 14 Pro · 麒麟 8020 · HarmonyOS 6.1 · 4CY9K25614046328
- **前置**：A8（纯原生手机端）；ADR-008（Vulkan 可达，GPU 只有 ncnn 一条路）；ADR-011（ncnn 接入与内存 API 陷阱）
- **证据**：`_evidence/ncnn_slots_20260918.log`（全量 hilog）
- **本轮填补的缺口**：此前 **GPU(Vulkan) 只有检测器能跑** —— 识别与分类没有 ncnn 版本

---

## 1. 转换：pnnx 直接吃 ONNX

工具 `pnnx 20260526`（pip，自带 `pnnx.exe`）：

```
pnnx rpv3_mdict_160_r3.onnx  ncnnparam=rec.ncnn.param ncnnbin=rec.ncnn.bin
pnnx litemodel_cls_96x_r1.onnx ncnnparam=cls.ncnn.param ncnnbin=cls.ncnn.bin
```

| 模型 | ONNX 算子（distinct） | 转换结果 | 层数 |
|---|---|---|---|
| cls | 8 种：Conv/Relu/Add/Gemm/AveragePool/Reshape/PRelu/Softmax | ✅ | **78** |
| rec | 24 种：Conv/MatMul/Transpose/Softmax/ReduceMean/Sqrt/Pow/Clip/… | ✅ | **252** |

**关键风险被 pnnx 自动消解**：rec 含 `Shape` 与 `Slice`（ncnn 是静态形状框架，没有动态形状层），
转换日志显示 pass 表里两者都变成 `1 → 0` —— **被常量折叠掉了**，因为输入是固定形状
`[1,3,48,160]`。这原本是「识别器能不能转 ncnn」的最大不确定点。

产物落进 `rawfile/models/`（`rpv3_mdict_160_r3.ncnn.{param,bin}`、`litemodel_cls_96x_r1.ncnn.{param,bin}`）。

## 2. 接入：槽位化的 ncnn 与流水线分支

| 改动 | 说明 |
|---|---|
| `ncnn_engine.{h,cpp}` | 新增**通用模型槽位**注册表（1 = 识别，2 = 分类）。每槽位独立持网与 param/bin 副本（内存版 `load_model` 只引用不拷贝）。**检测旁路一个字节没动** —— 它已由 ADR-011/A4 验证过，不为腾位置而重构它 |
| `lpr_pipeline.{h,cpp}` | `LprSessions` 增 `recSlot` / `clsSlot`；`Recognise()` 与分类段各加一条 ncnn 分支。两条路的输入编码**同源**（同一个 `LprEncodePlate` / `LprEncodeClassify`），差别只在布局：MS 报 NHWC、ncnn 吃 NCHW |
| `napi_init.cpp` | `pipelineAsync(...)` 增可选的第 8/9 参 `recSlot`/`clsSlot`；新增 `ncnnLoadSlotAsync(slot, param, bin, useVulkan)`；全部走既有推理线程（Vulkan 首载含 shader 编译，压在 JS 线程会被 watchdog 杀） |
| `Index.ets` | 新增 `ncnnSlotProbe()`：识别/分类各在 CPU 与 Vulkan 上跑 3 次，接进自检链与探针控制台（「三后端槽位」按钮） |

`T/C` 的取法（识别模型）：ncnn 侧不自省形状，用**实际元素数**反推 —— 步数 `T=20` 是该模型的
固定输出（ONNX 声明 `[1,20,78]`），`C = n / T`。这样 `C` 仍然来自模型，不会退化成字符表的 77
（那个 77≠78 的坑会让 CTC 每步错切一列，实测能把 7 字符车牌解成 19 字符）。

## 3. 结果：三模型 × GPU 闭合

### 加载自证

```
NCNNSLOT LOAD role=rec mode=cpu    ok=1;slot=1;layers=252;effectiveVulkan=0
NCNNSLOT LOAD role=rec mode=vulkan ok=1;slot=1;layers=252;effectiveVulkan=1;gpuProbe=ok=1;count=1;name=Maleoon 920C;fp16=111
NCNNSLOT LOAD role=cls mode=cpu    ok=1;slot=2;layers=78; effectiveVulkan=0
NCNNSLOT LOAD role=cls mode=vulkan ok=1;slot=2;layers=78; effectiveVulkan=1;gpuProbe=…Maleoon 920C
```

### 端到端（每格 3 次，全部逐字符一致）

| 角色 | 模式 | match | 车牌 | 颜色 | **分类分数** | 该段耗时 |
|---|---|---|---|---|---|---|
| rec | ncnn-CPU | 1/1/1 | 苏ED5172 | 蓝牌 | 0.0001\|0.9998\|0.0001 | 19.0 – 23.9 ms |
| rec | **ncnn-Vulkan** | **1/1/1** | 苏ED5172 | 蓝牌 | 0.0001\|0.9998\|0.0001 | 52.4 – 72.0 ms |
| cls | ncnn-CPU | 1/1/1 | 苏ED5172 | 蓝牌 | 0.0001\|0.9998\|0.0001 | 4.6 – 7.9 ms |
| cls | **ncnn-Vulkan** | **1/1/1** | 苏ED5172 | 蓝牌 | 0.0001\|0.9998\|0.0001 | 11.0 – 14.7 ms |

**数值保真（逐位核验后的准确表述）**：

| 被换掉的那一段 | ncnn 输出 vs 生产链（MS rec=NPU / cls=CPU） | 判定 |
|---|---|---|
| **分类**（clsSlot） | `clsScores` = `0.0001\|0.9998\|0.0001` —— **与生产链逐位相同** | ✅ 逐位一致 |
| **识别**（recSlot） | 字符序列 `苏\|E\|D\|5\|1\|7\|2` 与车牌串一致；但**逐字符概率有 ~0.5–1% 漂移**：ncnn `0.7271\|0.7901\|0.5786\|0.9947\|0.7794\|0.7127\|0.5574` vs MS-NPU `0.7227\|0.7969\|0.5762\|0.9941\|0.7832\|0.7168\|0.5664` | ️ **符号级一致、非逐位** |

**顺带一个硬结论**：rec 的 8 行（CPU 与 Vulkan 各 1 预热 + 3 次）产生的 charProbs **取值唯一** ——
即 **ncnn 的 CPU 与 Vulkan 两个后端输出完全相同**，两者与 MS-NPU 的那点差是
「ncnn ↔ MindSpore/NNRT」框架之间的差，不是「CPU ↔ GPU」之间的差。

> 对照（同一次运行的生产链）：`NATIVE PIPE code=苏ED5172 … stages=84.3|16.1|9.3|5.6 cropSum=2773473`
> —— `cropSum` 与 A5/A6/A7/A8 四次运行逐值相同，说明整条链路没有被本轮改动碰到。

**为什么这个漂移不致命**：识别头是 argmax + CTC，对数值噪声天然鲁棒（0.6% 的概率变化不会
改变 argmax 结果）；而检测头的关键点 = `logit × anchor`（anchor 最大 433），同量级的噪声经放大
就是 ~10 px、会翻字符 —— 这正好解释了两者的差别（ADR-009）。

## 4. ⚠️ 必须一起说出口的结论：GPU 在这里**不是加速器**

| 模型 | CPU | NPU | **GPU (ncnn-Vulkan)** |
|---|---|---|---|
| 识别 rpv3（该段） | 19–24 ms（ncnn）/ ~22 ms（MS） | **7.5 ms** | 52–72 ms |
| 分类 litemodel（该段） | 4.6–7.9 ms | 5.6 ms（fp32 变体可上） | 11–15 ms |

**GPU 比 CPU 慢**，与 NPU 在小图上拿不到收益是同一机理：固定的提交/委托开销摊不薄
（ADR-003 §5 的「加速比随图规模单调下降 13.11× → 0.98×」）。

因此 GPU 这一格的价值**不是性能**，而是：
1. **通路闭合**：“三个模型都能在 CPU / NPU / GPU 上跑通且结果一致”这件事本身成立，
   且 GPU 是唯一能同时覆盖三个模型的可编程加速器（MS Lite 的 GPU 档编译期判否）。
2. **保真对照的第三个数据点**：同一份 ONNX、两套推理实现、两种设备，输出一致 ——
   这是"移植保真"方法学的又一次自证。

**论文/简历口径**：写"GPU 通路成立且数值保真；本期实测未观测到加速（小图开销主导）"，
**不要**写成"GPU 加速"。

## 5. 边界

- 每格 3 次重复 + 1 次预热，是**功能验证口径**，不是 30 次 p50 基准；`benchAsync` 已就绪，
  正式矩阵尚未换口径。
- Vulkan 侧证据为 `requestedVulkan=1 + effectiveVulkan=1 + 枚举到 Maleoon 920C`，
  **不是逐层 GPU 时间戳执行证据**。
- 未标注 thermal level（热态会使 NPU 延迟上升 15–20%）。
- 识别器的逐元素对照：**已用 `charProbs` 做了逐字符核验**（见 §3），结论是符号级一致、
  概率有 0.5–1% 漂移；要做到逐元素 L2 对照，需要给 MS 与 ncnn 两个 runner 喂**同一份输入张量**
  （目前 MS 侧 `bench` 的输入是内部确定性生成的，两边输入不同源）。

## 6. 工程约定（本轮新增）

- ncnn 槽位号是**约定**：0 = 检测旁路（保留）、1 = 识别、2 = 分类。加新模型要同步改
  `Index.ets` 的常量与这里的说明。
- 槽位加载失败**不静默**：`ok=0;slot=..;error=..` 会原样进 hilog 与落盘日志。
- 转换脚本的输入要用 `.onnx` 扩展名（项目里运行时的 `.onnx.json` 是给静态托管白名单看的）；
  转换在工程外目录做（`lpr-harmony/ncnn_conv/`），别把临时 `.onnx` 留在项目里造成"两份模型漂移"。