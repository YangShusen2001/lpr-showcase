# assets/ —— 目录约定与「不能改回去」的文件名

> 这份 README 存在的唯一目的：**防止后来的人（或未来的我）把文件名改回 `.onnx` / `.mjs`，
> 从而悄悄弄坏实时 Demo。**

## 1. 目录布局

| 路径 | 内容 | 浏览器是否加载 |
|---|---|---|
| `assets/css/site.css` | 全站样式（介绍页 + 演示页共用） | 是 |
| `assets/js/demo.js` | 演示控制器：加载 ORT、跑流水线、渲染中间量 | 是 |
| `assets/js/pipeline.js` | 四级流水线的纯算法层，零依赖，只吃 `{data,width,height}` 缓冲区 | 是 |
| `assets/models/*.onnx.json` | 三个运行时 ONNX 模型（检测 / 识别 / 分类） | 是 |
| `assets/models/*.onnx` | **已于 2026-09-18 归档** → `_archive/2026-09-18/dead-assets/models/`（`y5fu_640x_sim`、三个 YOLOv9，见 ADR-001 证伪记录） | 否 |
| `assets/ort/ort.min.js` | onnxruntime-web 入口 | 是 |
| `assets/ort/ort-wasm-simd-threaded.js` | ORT 的 WASM 加载器（worker glue） | 是 |
| `assets/ort/ort-wasm-simd-threaded.wasm` | ORT 的 WASM 内核，13.3 MB | 是 |
| `assets/ort/ort-wasm-simd-threaded.jsep.*` | WebGPU/JSEP 变体，本项目**不用**（走 wasm provider）；**已于 2026-09-18 归档** → `_archive/2026-09-18/dead-assets/ort/` | 否 |
| `assets/samples/upstream/` | 上游原图副本；**已于 2026-09-18 归档**（4 张已提升为 `assets/samples/*.jpg`） | 否 |
| `assets/samples/*.jpg` | 演示用的真实车牌照片 | 是 |

## 2. 为什么模型叫 `.onnx.json`、ORT 入口叫 `.js`

本地预览服务（编辑器内置的静态托管，形如
`http://127.0.0.1:<port>/static-html/<hash>/`）**按扩展名白名单放行**，白名单大致是
`html / css / js / json / wasm / jpg / png / webp`。

实测（HEAD 请求）：

```
assets/ort/ort.min.js                          200  application/javascript
assets/ort/ort-wasm-simd-threaded.js           200  application/javascript
assets/ort/ort-wasm-simd-threaded.wasm         200  application/wasm      （13.3 MB 也放行）
assets/models/y5fu_320x_sim.onnx.json          200  application/json
assets/samples/scene-2.jpg                     200  image/jpeg

assets/ort/ort.min.mjs                         403  ← 白名单外
assets/models/y5fu_320x_sim.onnx               403  ← 白名单外
```

两个关键事实：

1. **被拦是因为扩展名，不是因为体积。** 13.3 MB 的 `.wasm` 返回 200，而 0.35 MB 的 `.mjs`
   返回 403。所以「文件太大」不是解释，改扩展名才是解法。
2. **浏览器只认字节，不认扩展名。** `.mjs` 与 `.js` 都是 `text/javascript`，`.onnx` 与
   `.onnx.json` 都是「一段 ONNX protobuf 字节」。改扩展名不改变任何语义。

因此：

- `ort.min.mjs` → **`ort.min.js`**
- `ort-wasm-simd-threaded.mjs` → **`ort-wasm-simd-threaded.js`**
- 三个运行时模型 `.onnx` → **`.onnx.json`**（保留 `.onnx` 字样是为了自解释：
  「这是一份 ONNX 文件，只是加了个白名单后缀」）

## 3. 配套的代码约束（改文件名时必须同步改这三处）

**① `assets/js/demo.js` —— ORT 入口与 `wasmPaths` 必须用「对象形式」**

```js
const ort = await import('../ort/ort.min.js');
// 不能写成 wasmPaths = '../ort/'（目录字符串）：ORT 会自行拼出 .mjs 文件名，
// 于是又踩回 403。必须显式给出改名后的 worker 与 wasm 两个 URL。
ort.env.wasm.wasmPaths = {
  mjs: new URL('../ort/ort-wasm-simd-threaded.js', import.meta.url).href,
  wasm: new URL('../ort/ort-wasm-simd-threaded.wasm', import.meta.url).href,
};
```

**② `assets/js/demo.js` —— 模型 URL 加 `.json`，并显式钉住格式**

```js
const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', format: 'onnx' };
const base = new URL('../models/', import.meta.url).href;
const det = await ort.InferenceSession.create(base + 'y5fu_320x_sim.onnx.json', opt);
```

`format: 'onnx'` 不能省：`.onnx.json` 会让 ORT 的格式猜测器把 `.json` 当成
「可能是 ORT 自己的 JSON 格式」，显式钉住才能消除这个歧义。

**③ `tools/hlpr_reference.py` —— Python 参考实现要能吃两种名字**

Python 侧（保真度验证的 ground truth）读的是 `assets/models/<name>.onnx`。
改名后由 `model_path()` 兜底：先找 `.onnx`，找不到再找 `.onnx.json`。
**不要为了省事去复制一份 `.onnx`** —— 两份 14 MB 的模型必然漂移，
而「Python 与浏览器跑的是同一份字节」正是本项目的核心主张。

## 4. 自检清单（改完必须跑）

```powershell
# 1) 内嵌 Demo 的首页：真实推理跑通 + 高度在预算内
node tools/web_check.mjs "/index.html?auto=1" "__PAGE" 240000

# 2) 独立演示页
node tools/web_check.mjs "/demo.html?auto=1" "__PAGE" 240000

# 3) Python 参考实现仍能加载模型
.\.venv\Scripts\python.exe tools\hlpr_reference.py
```

验收标准：两次 `web_check` 的 `errors` 与 `consoleErrors` 均为空、
`plate` 是真实车牌字符串；`hlpr_reference.py` 不抛 `FileNotFoundError`。
