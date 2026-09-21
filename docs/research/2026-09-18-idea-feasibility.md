# IDEA.md 三条目标的可行性调研 · 2026-09-18

**调研对象**：`IDEA.md` 的三条目标（①毕设+简历+论文+演示页 ②数据可捏造/模型可乞讨 ③麒麟 CPU/NPU/GPU 对比 + PC 调用鸿蒙 NPU 实时回传 + 智能轮询摄像头）
**调研方式**：外部资料检索（华为官方文档/开发者博客 + 开源仓库 + 同类研究）+ 本项目已落地证据（ADR-001~011、`_evidence/`、`tools/harmony/`）
**结论一句话**：**方向不假、大部分已经做通；真正"第一步就错了"的不是能不能做，而是把三件"炫技件"（NPU 卖点、PC 调 NPU 实时链路、多摄轮询）当成了项目主张，而它们要么与实测结论打脸，要么没有学术/工程意义，要么硬件直接不支持。**

---

## 一、逐条判定

| # | 目标 | 合理性 | 能否做到 | 判定 |
|---|---|---|---|---|
| ① | 毕设 + 简历 + 论文 + 可演示网页，四合一 | 合理，是标准配置 | 已完成大半（IEEE 稿有形、展示页端到端跑通、简历条目已写） | ✅ 保留 |
| ② | 数据可捏造、模型可乞讨 | 模型用开源 = 完全合规且是刻意取舍；**"数据捏造"是红线** | — | ⚠️ 措辞必须改 |
| ③a | 麒麟 8020 CPU/NPU/GPU 对比 | 合理，且是项目最硬的差异化 | **基本已做通**（见下） | ✅ 保留，需补方法学声明 |
| ③b | PC 调用鸿蒙 NPU，结果实时回传 PC | 技术上可行，**但无现成方案，且意义存疑** | 传输链路已通（走的是 WASM 不是 NPU），真·NPU 通路还需 1–2 周 | ⚠️ 降级为演示 |
| ③c | 智能轮询摄像头，谁拍到优先谁 | **与平台能力冲突** | 多个后置摄像头并发：官方未开放 | ❌ 必须改设计 |

### ③a 已有证据（不是"计划"，是"已发生"）
- NPU 路径：MindSpore Lite 2.6.0 → NNRT，`NPU_ohos.boot.hardware.kirin8020_v2_0`，日志 `HCL+hiaiserver` 实锤委托生效（ADR-009），**而这条路是 HarmonyOS 官方 Kit，不需要企业白名单**。
- GPU 路径：`dlopen libvulkan.so` 拿到 **Maleoon 920C**（0x19e5，Vulkan 1.3.275，fp16 在位），ncnn-vulkan 与 ncnn-cpu 已能同模型对照（ADR-008 / ADR-011）。
- 结论级发现：NPU 加速比随图规模单调下降 13.11× → 0.98×（小图无收益），且 **0.08% 数值漂移经 anchor 放大≈10px → 翻字符**，故生产链是 `det=CPU / rec=NPU / cls=CPU`。

### ③a 必须补的一条方法学声明（否则会被答辩问穿）
三个后端**跑在不同的框架栈上**：NPU 走 MindSpore Lite + NNRT，CPU 有 ms-cpu 与 ncnn-cpu 两套，GPU 只能走 ncnn-Vulkan（MS Lite 无 GPU 后端、ORT-Web 已死）。

→ 因此 **"CPU 78.9 ms vs NPU 6.02 ms"跨栈比较，只能当量级参考**。可比的是**同栈内比值**：`ncnn-cpu ↔ ncnn-vulkan`、`ms-cpu ↔ ms-nnrt`。表格里必须加一列 `框架`，否则这是全项目最容易被攻的一个点。

---

## 二、我认为"第一步就做错"的三个具体点

### A. 把"跑在麒麟 NPU 上"写成了项目标题（症状最重）
现状：`index.html` 的 `<title>`/`<h1>` 是"基于 YOLOv5 与 CRNN-CTC 的**在麒麟 8020 NPU 上**端到端车牌识别系统"，而论文与简历是"**基于 HyperLPR3 的端到端车牌识别与跨语言移植保真验证**"。

两个问题：
1. **口径不一致**：同一份材料给答辩老师和面试官的是两个不同的项目（一个是"算法选型 + 硬件部署"，一个是"移植保真方法学"）。
2. **卖点与结论打脸**：你实测的结论恰恰是"检测器四条补救路径全部回落 CPU""NPU 会翻字符"——标题写"在麒麟 NPU 上"，等于给评委递刀。

**改法**：标题只保留能兑现的主张 —— `基于 HyperLPR3 的端到端车牌识别：跨语言移植保真与麒麟 8020 异构后端实测`。NPU 是章节，不是标题。

### B. "PC 调用鸿蒙 NPU 实时回传" 被设计成了主功能
- **现成方案：没有。** 检索范围内不存在"鸿蒙手机当推理节点供 PC 调用"的开源项目。
- **你已经走通了一半**：`tools/harmony/pc_drive_phone.mjs` 用 `hdc fport tcp:9222 localabstract:webview_devtools_remote_<pid>` + CDP 驱动手机页面 —— 手法正确、思路漂亮。**但它驱动的是 ArkWeb，算力是 WASM/CPU，压根没碰 NPU。** 要真上 NPU，得让推理走 NAPI→NNRT，并自定义协议（socket 或 hdc fport 端口）回传。
- **意义问题（面试官的必问题）**：一台普通 PC 跑同一模型比手机 NPU 快一个数量级；图像经 USB 传输 + 编码的开销（数十 ms）远大于推理本身（NPU 上 5 ms）。**"能这么干"不是"该这么干"的理由。**

**改法**：定位改为 **"手机端推理节点 + PC 端编排/可视化"的协同链路演示**，明确写"这是联调能力展示，不是性能优化"，且**不要写进论文创新点**。论文里保留端侧三后端实测即可。

### C. "轮询摄像头，谁先拍到优先谁" 撞平台墙
官方与社区口径：
- 多摄同开**自 API 18 起支持，且官方开放的是"前置/后置相机同时开启预览与录像"**；前置/后置同时拍照"待开放"。
- 官方明确警告两类失败：**没查并发能力集就开相机**、**多摄同开超出基础功能范围**。
- 社区实测结论：**不是所有设备都支持同时打开两个摄像头**，硬件 ISP 通道有限；能并发的是系统给出的**组合与 Profile**，不是应用随便挑。

→ "多个后置摄像头并发 + 谁命中用谁"在手机上**做不出来**。

**改法（三选一，按推荐序）**：
1. **降级为顺序调度**：`switch` 逻辑摄像头 + 分段抓拍，做"多路轮询调度 + 命中锁定"的策略与延迟测量（可做、可测、有话讲）；
2. 换成**多路 CSI 的开发板**（RK3568/DAYU200 一类）做真正的多摄并发，但那已经不是"麒麟 8020"的故事了；
3. 删掉，把精力还给 ③a/③b。

---

## 三、②"数据可以捏造"的风险与合规化写法

| 做法 | 判定 |
|---|---|
| 用 HyperLPR3 开源模型 | ✅ 合规，且 ADR-001 已把它论证为**刻意取舍**（附 YOLOv9-t 否证记录），这是加分项 |
| 合成车牌数据（`gen_plate_dataset.py`） | ✅ 合规，但**必须标注"合成"**，写清生成方式与用途（只用于链路/回归，不用于报准确率） |
| 自采/公开实拍图 | ✅ 合规 |
| 编造性能/精度数字 | ❌ **学术不端**，且你现有的数据都是真实测量（协议、原始日志齐全）—— 没有任何理由冒这个险 |

**替代原则**：捏造"数据"→ 换成"合成数据 + 来源声明"；捏造"结果"→ 绝对不碰。你项目里"刻意不报准确率"的处理是对的，继续保持。

---

## 四、现成案例清单（可直接引用/借用）

### 4.1 鸿蒙端推理（与 ③a/③b 同构）
| 项目 | 价值 | 链接 |
|---|---|---|
| cmdbug/ncnn-harmony | 鸿蒙 NEXT API12 + ncnn，跑 NanoDet/YOLOv4-tiny，**含相机实时识别页 + benchncnn 基准页** —— 第③点最接近的现成骨架 | https://github.com/cmdbug/ncnn-harmony |
| OpenHarmony TPC · ncnn | ncnn 官方进 OpenHarmony 生态，交叉编译路径成熟 | https://gitcode.com/openharmony-tpc/ncnn |
| ncnn on HarmonyOS Next 开 Vulkan 链接失败（glslang 符号） | 你 ADR-008/011 的坑，别人也踩过 | https://blog.gitcode.com/7eb935b3473eba6ac91943586af39322.html |
| OpenHarmony 上 Vulkan 推理价值研究（此芯 P1 + RX580，benchncnn CPU/GPU 对照） | 与"GPU 到底有没有价值"完全同构，可作 Related Work 对标 | https://laval.csdn.net/68930a4fa6db534ba2bf4af9.html · https://ost.51cto.com/posts/36262 |
| OpenHarmony for RKNN（湖南大学 ESNL） | OpenHarmony 上接 Rockchip NPU 做推理的完整先例 | https://esnl.hnu.edu.cn/info/1002/2793.htm |
| AI Benchmark（Ignatov et al., 2018） | 手机 SoC NPU 评测的学术先例，§V 应引 | https://arxiv.org/abs/1810.01109 |

### 4.2 车牌识别移植（说明"移植本身不是创新点"）
| 项目 | 价值 | 链接 |
|---|---|---|
| HyperLPR3 本体 | 模型来源，官方支持 Linux/ARM/MacOS/Android，**无 HarmonyOS** | https://github.com/szad670401/HyperLPR |
| PlateRecognition / 其 ncnn 移植版 | 同类"高性能 LPR + ncnn 移植"已有公开实现 | https://github.com/pcb9382/PlateRecognition · https://github.com/zhahoi/PlateRecognition_ncnn |
| hyperlpr3-android-sdk | 官方 Android SDK 化先例 | https://github.com/HyperInspire/hyperlpr3-android-sdk |

→ **启示**：把 LPR 搬到某平台 = 常规操作。你的差异化只能也必须落在 **"跨语言数值保真方法学" + "端侧运行时契约缺陷定位"** 上，这一点项目已经走对了。

### 4.3 平台能力边界（决定 ③b/③c 的天花板）
| 议题 | 事实 | 链接 |
|---|---|---|
| 多摄同开 | API 18+，官方开放前/后摄并发；须先查并发能力集；超范围失败 | https://developer.huawei.com/consumer/cn/blog/topic/03209039596692077 |
| 双摄并发限制 | 非所有设备支持，ISP 通道有限；能并发的是系统给定的组合 | https://segmentfault.com/a/1190000047888064 |
| MindSpore Lite Kit | 官方内置 NNRT 使能 AI 芯片加速，**无需企业白名单** | https://developer.huawei.com/consumer/cn/sdk/mindspore-lite-kit |
| Vulkan 支持 | HarmonyOS SDK 支持 Vulkan 1.4.309，具体取决于 GPU 驱动 | https://developer.huawei.com/consumer/cn/doc/harmonyos-references/vulkan |
| PC↔手机通路 | `hdc fport`（含 localabstract）+ agent 进程做 RPC，有成熟先例 | https://harmonyosdev.csdn.net/69e6dd3b54b52172bc6b21fc.html |
| PC 侧直接说 hdc 协议 | `hdctool`（PyPI），Python 直连 hdc 守护进程，省掉 shell 拼接 | https://pypi.org/project/hdctool/ |

---

## 五、建议的收口顺序

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | 统一口径：网页 `<title>/<h1>` ↔ 论文标题 ↔ 简历标签 三处对齐；端侧表格补 `框架` 列 | 成本半天，消除全项目最容易被问穿的两处 |
| **P0** | 论文加"数据来源与合成声明"；确认不报准确率的边界写法 | 学术诚信，成本极低 |
| **P1** | 三模型 `.ms` 端到端（A4 已有 det 四后端对照）+ ≥5 轮 thermal 采集 | 补齐 ADR-003 §7 的欠账，这是"已完成"与"方法学已通"的分界 |
| **P1** | ncnn-Vulkan 30 次正式计时 + 逐层执行证据 | 让 GPU 那一列从"能跑"变成"有数据" |
| **P2** | PC↔手机协同：把推理从 ArkWeb 挪到 NAPI/NNRT，CDP 只当控制面 | 复用现有 `pc_drive_phone.mjs`，是最划算的"惊艳演示" |
| **P3** | 多摄：改顺序调度 + 命中锁定；或明确写"平台不支持并发"作为边界项 | 别为一个硬件不支持的诉求再投时间 |

### 附：可复现性风险（答辩/面试现场）
端侧全部数字绑定在**一台 nova 14 Pro（4CY9K25614046328）+ 特定 SDK/系统版本**上，设备故障或系统升级会让数字不可复现。
→ 主证据用 **PC 侧可复现的保真结果（8/8 逐通道一致）**，端侧数字配 `_evidence/` 原始日志与脚本，声明"单机单版本实测"。
