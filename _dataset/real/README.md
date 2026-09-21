# `_dataset/real/` —— 真实第三方车牌数据集（带标签）

## 内容

| 路径 | 内容 | 数量 |
|---|---|---|
| `crops/` | 真实车辆车牌**裁剪图**，**文件名即真值**（如 `京PL3N67.jpg`） | **1000 张**（**2.52 MB 实际内容**；`du` 显示 4.2 MB 是 4K 块对齐开销，勿引用） |
| `stretch_control.json` | 拉伸/压缩对照实验原始结果（`tools/stretch_control.py`） | n=200 |
| `colour_audit2.json` | 修正后的牌面颜色（`tools/plate_face_colour.py`）—— **用这个** | n=1000 |
| `glyph_neutrality.json` | 白平衡对照：牌面漆 vs 车牌白字（`tools/plate_face_colour.py` 逻辑） | 若干样本 |
| `golden_colour_corrected.json` | 黄金牌修正后牌色量测 | 1 |
| `colour_analysis.json` | ⚠️ **分类器有 bug，按色细分数字作废**（`tools/plate_colour_analysis.py`） | n=1000 |
| `glyph_geometry.json` | 宽高比 ↔ 字符数标定（`tools/glyph_count_geometry.py`） | n=1000 |

## 出处与许可

- **仓库**：`sirius-ai/LPRNet_Pytorch`
- **路径**：`data/test/*.jpg`
- **下载方式**：`https://codeload.github.com/sirius-ai/LPRNet_Pytorch/tar.gz/refs/heads/master`
  （18 MB tarball 一次性拉取；GitHub API 逐文件下载会触发限速，已弃用）
- **性质**：LPRNet 公开参考实现**随仓库发布的测试集**，真实拍摄车牌裁剪图，非合成
- **标签**：文件名即车牌串，**人工标注**
- **用途**：仅用于本项目学术精度评测（毕设 / IEEE 论文），不作再分发

**为什么选它**：这是 **LPRNet 自己的官方测试集**。用"对手的主场"做对比，结论对
生产识别器 rpv3 更保守——rpv3 从未在该分布上训练过。

## ⚠️ 已知局限（引用数据时必读）

1. **1000 张全部是 7 字符牌，新能源（8 字符）绿牌 0 张。**
   长度分布 `{7: 1000}`。**不能**用本数据集评估新能源牌场景。
   （已用牌色独立复核：1000 张里没有一张是真绿牌，与"真值全 7 字符"互洽。）
2. 全为**已裁剪**的车牌小图，**不含**检测环节，因此只能评识别器，不能评端到端。
3. 数据高度集中在**安徽省（皖）**，省份分布不均衡（rpv3 的主要错误正是省份混淆）。

## 结论摘要

见 `_evidence/A16-real-dataset-accuracy-20260918.md`：

- rpv3 **906/1000 = 90.6%** vs LPRNet **888/1000 = 88.8%**，McNemar **p=0.179 不显著**
- 错误画像：rpv3 **84/94 是同长度替换**，长度错误 **10/1000 = 1.0%**；
  对照组 LPRNet 长度错误 **70/1000 = 7.0%**（**是 rpv3 的 7 倍**）
- → **LPRNet 不能替换 rpv3**

## 复现

```bash
cd C:/Users/26671/Desktop/车牌识别
./.venv/Scripts/python.exe tools/lprnet_real_accuracy.py   # 全量精度
./.venv/Scripts/python.exe tools/stretch_control.py        # 拉伸/压缩对照
```
