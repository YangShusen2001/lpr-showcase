# ADR-010：端侧演示 300ms 的归因 —— 多线程没问题，瓶颈是 letterbox 预处理

- **状态**：已接受（真机实测）
- **日期**：2026-09-17
- **前置**：ADR-004 §5.4（Web 多线程）、ADR-006 §7（"检测耗时大头疑似预处理"）
- **设备**：HUAWEI nova 14 Pro（MIA-AL00）· 麒麟 8020 · HarmonyOS 6.1.0.135
- **证据**：`_evidence/web_diag_phone_20260917.txt`

---

## 1. 现象与疑问

「手机端侧演示还是 300 多毫秒」——而 ADR-004 记录的是 6 线程 85 ms。
差 3.5 倍，必须先分清是**多线程没生效**还是**场景不同**。

## 2. 实测：多线程是好的

在真机上跑 `demo.html?bench=30&threads=6`，直读 `window.__BENCH`：

```
{"done":true,"threads":6,"isolated":true,"sab":true,
 "hardwareConcurrency":12,"n":30,"p50":88.3,"mean":92.6,"min":86.9,"max":142.4}
```

- `threads=6`、`isolated=true`、`sab=true` —— COOP/COEP 生效、SharedArrayBuffer 在位、
  线程数钉住了。**Web 路径本身 88.3 ms，与 ADR-004 的 85 ms 一致。**
- 所以 300 ms **不是**多线程问题，也不是线程回落到 1。

## 3. 归因：输入分辨率 → letterbox

两条独立路径指向同一结论：

| 路径 | 输入 | 端到端 | 其中推理 |
|---|---|---|---|
| Web（bench 小图） | 小尺寸样本 | **88.3 ms** | — |
| Web（相机/整车图） | 1080p 帧 | **~300 ms**（用户观察） | 88 ms |
| 原生（hlpr-test 1080p） | 1920×1080 | 115.9 ms | det 79 ms 里推理仅 10–18 ms |

**推理不是瓶颈**：原生 det 阶段 79 ms，`DET BENCH` 的 CPU 推理 p50 只有 9.9–18.8 ms。
剩下 60+ ms 是 letterbox（1920×1080 → 320×320 双线性）+ 打包 + NMS/decode。

**ADR-006 §7 的怀疑至此被交叉证实**：检测阶段的大头是预处理，不是模型。

## 4. 结论与下一步（并入 C 阶段）

1. **相机取流降分辨率**（首选，收益最大）：预览流取 640×480 而非 1080p，
   letterbox 的输入面积降到 1/7，需同步验证小目标车牌仍可检出（量化后再定）。
2. **分段计时**：把 det 拆成 letterbox / 打包 / 推理 / decode 四段打到日志，
   用数字坐实"预处理占 X%"，再决定是否值得做 NEON/多线程预处理。
3. Web 端同理：相机流先降到 640×480 再喂 pipeline。
4. 备注：原生端到端 102→116 ms 的波动（det 68→79 ms）来自热降频/调度抖动，
   **单次测量不足以下结论** —— 后续的对比一律 ≥5 轮 + thermal 标注。

## 5. 工程提示

- `window.__BENCH` 是真机 Web 路径唯一的自证入口（demo.js），
  `threads/isolated/sab/hardwareConcurrency/p50` 一次拿全，比猜参数快。
- App 自检里取它：切到 `demo.html?bench=N&threads=M` → `waitBench()` 轮询 → hilog。
