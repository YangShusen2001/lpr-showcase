/**
 * 原生 LPR 应用 —— 三个标签页
 *
 *   原生演示 → 纯 ArkTS/NAPI，不经过 Web。三模型分别指定后端，C++ 跑完整流水线
 *   端侧演示 → mobile.html?embed=1   竖屏 Web 演示页（ORT WASM，只吃 CPU）
 *   探针     → webprobe.html + 加速器矩阵 + WASM 多线程扫描
 *
 * 为什么必须有「原生演示」这一页：
 *   麒麟 8020 的 NPU **只能**从原生 MindSpore Lite 触达。onnxruntime-web 既没有
 *   WebGL EP（上游 v1.29.0 已删除）也拿不到 WebGPU adapter，所以 Web 路径 100%
 *   是 CPU（ADR-004 §6.1）。要让识别模型吃到 NPU 的 2.2–3.0×，流水线必须落在
 *   C++ 里（lpr_pipeline.cpp），而 NAPI 的 loadModel 第三参数正好可以逐模型挑后端。
 *
 * 三个模型的后端选择不是随手写的（ADR-004 §3 实测矩阵）：
 *   y5fu 检测  → CPU。NPU 拒收：图里有 12 个 rank-5 常量（anchor-grid decode 烘进
 *                计算图），HiAI 的常量转换读不出 shape，CreateConstOp 失败后整图回落。
 *                四条图手术补救路径全部实测失败，唯一出路是模型侧导出裸 head。
 *   rpv3 识别  → NNRT/NPU。2.2–3.0×，这是本项目唯一真正吃到 NPU 的模型。
 *   cls 分类   → CPU。NPU 反而更慢（1.09 ms vs 0.57 ms），委托开销在小图上摊不薄。
 *
 * 注意：ArkTS 禁用 any/unknown，`JSON.parse` 无法通过 linter，
 * 所以原生侧一律返回 `key=value;` 扁平串（见 napi_init.cpp 的 KvSanitize）。
 */
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { camera } from '@kit.CameraKit';
import { image } from '@kit.ImageKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { webview } from '@kit.ArkWeb';
import lpr from 'libentry.so';

const TAG = 'LprMatrix';
const WEB_TAG = 'LprWeb';
const NATIVE_TAG = 'LprNativeUI';
const DOMAIN = 0xD001;

/**
 * 演示页 URL。`threads=6` 是钉死的：ADR-004 §5.4 两轮扫描都显示 6 线程是这台
 * 麒麟 8020 的最优点（≈85 ms），12 线程反而更慢（占满异构核 + 抢 ArkWeb 渲染核 +
 * 热降频）。
 */
const DEMO_URL: string = 'https://lpr.local/mobile.html?embed=1&threads=6';
const PROBE_URL: string = 'https://lpr.local/webprobe.html';

// ---------------------------------------------------------------- 原生演示配置

/**
 * 三个模型各自的请求后端（见文件头注释的实测依据）。
 *
 * det 用**裸 head** 但落 CPU，是 2026-09-17 实测后的取舍（ADR-009）：
 *   - 裸 head 上 NPU 是通的：NNRT 落点 OK、6.63ms 均值（vs 设备 CPU p50 14.3ms）；
 *   - 但同输入同布局下 NPU 与设备 CPU 仍有差：L2 1043.47 vs 1042.66（0.08%）、
 *     maxAbs 18.000 vs 18.0232。关键点 = logit×anchor（anchor 最大 433），
 *     0.023 的 logit 差经放大就是约 10px 偏移 —— 实测把 苏ED5172 认成 苏E05172
 *     （det 落 NPU 的结果），落 CPU 后逐字符恢复正确。
 *   - 消费检测关键点的下游（单应矫正）对 1 像素都敏感，所以 det 不吃 NPU；
 *     识别模型走 NPU 没问题（argmax+CTC 对数值噪声鲁棒，2.2–3.0× 收益照拿）。
 * 裸 head 保留（rank-4 能上 NPU 是 ADR-006 的结论），decode 在 C++ 里做，
 * 已由 verify_head_decode.py + verify_bare_head_pipeline.py 双重证明与原模型一致。
 * 裸 head 仍保留（比原模型少 103 个 decode 节点，decode 改在 C++ 里做，
 * 已由 verify_head_decode.py 证明与原模型逐值一致）。
 */
const BACKEND_DET: string = 'cpu';
const BACKEND_REC: string = 'nnrt';
const BACKEND_CLS: string = 'cpu';

const MODEL_DET: string = 'models/y5fu_320x_head_fp32.ms';
const MODEL_REC: string = 'models/rpv3_mdict_160_r3.ms';
const MODEL_CLS: string = 'models/litemodel_cls_96x_r1.ms';

/** 可识别的样本（rawfile 路径）。前两张是整车图，后几张是切好的车牌。 */
const NATIVE_SAMPLES: string[] = [
  'assets/samples/hlpr-test.jpg',
  'assets/samples/scene-2.jpg',
  'assets/samples/crop-0-津B6H920.jpg'
];

interface Row {
  model: string;
  backend: string;
}

/** 三个正式模型 + 图手术变体（变体只跑 nnrt / gpu，用来回答 fp32 能否上 NPU）。 */
const FULL_BACKENDS: string[] = ['nnrt', 'gpu', 'kirin', 'cpu'];
const MODELS_FULL: string[] = [
  'models/rpv3_mdict_160_r3.ms',
  'models/litemodel_cls_96x_r1.ms',
  'models/y5fu_320x_sim_s1r4_fp16.ms'
];
const MODELS_NNRT_GPU: string[] = [
  'models/litemodel_cls_96x_r1_fp32.ms',
  'models/y5fu_320x_sim_s1_fp32.ms',
  'models/y5fu_320x_sim_s1_fp16.ms',
  'models/y5fu_320x_sim.ms'
];

/**
 * WASM 多线程扫描档位。末尾故意重复一次 1 线程：整轮扫描要跑几分钟，SoC 会热降频，
 * 首尾两个 1 线程数据点之差就是这一轮的热漂移量。
 */
const MT_COUNTS: number[] = [1, 2, 4, 6, 8, 12, 1];
const MT_ITERS: number = 30;
const MT_TIMEOUT_MS: number = 120000;
/** 矩阵结果已固化在 ADR-004，日常不再重复 20 组组合；要复测把它改回 true。 */
const RUN_MATRIX: boolean = false;

/**
 * 启动自检：装机后自动「加载模型 → 枚举镜头 → 跑一次样本」，全部写 hilog。
 *
 * 存在的理由：原生 ArkUI 的按钮点不了（CDP 只能驱动 Web 内容，管不到原生组件），
 * 命令行验证需要一条不需要人手指的路径。演示前把它改成 false，
 * 否则应用一打开就自己在跑。
 */
const AUTO_SELFTEST: boolean = true;
/** 自检用的样本（整车图，能同时验证检测 + 矫正 + 识别 + 分类）。 */
const SELFTEST_SAMPLE: string = 'assets/samples/hlpr-test.jpg';

function parseKv(s: string): Map<string, string> {
  const m: Map<string, string> = new Map<string, string>();
  const parts: string[] = s.split(';');
  for (let i = 0; i < parts.length; i++) {
    const p: string = parts[i];
    const eq: number = p.indexOf('=');
    if (eq > 0) {
      m.set(p.substring(0, eq), p.substring(eq + 1));
    }
  }
  return m;
}

function num(m: Map<string, string>, k: string): number {
  const v: string | undefined = m.get(k);
  if (v === undefined || v.length === 0) {
    return 0;
  }
  const f: number = Number.parseFloat(v);
  return Number.isNaN(f) ? 0 : f;
}

function str(m: Map<string, string>, k: string): string {
  return m.get(k) ?? '';
}

/** CameraType 枚举值 -> 人类可读的镜头类型（Camera Kit 的常量顺序）。 */
function lensName(t: number): string {
  if (t === 1) { return '广角'; }
  if (t === 2) { return '超广角'; }
  if (t === 3) { return '长焦'; }
  if (t === 4) { return 'TOF'; }
  if (t === 5) { return '真深度'; }
  return '默认';
}

/** CameraPosition: 0=未指定 1=后置 2=前置 */
function posName(p: number): string {
  if (p === 1) { return '后置'; }
  if (p === 2) { return '前置'; }
  return '未知';
}

/** 分类输出顺序是 [yellow, blue, green]（见 pipeline.js 的 PLATE_COLOURS）。 */
function colourOf(cls: string[]): string {
  let best: number = 0;
  let bv: number = -Infinity;
  for (let i = 0; i < cls.length; i++) {
    const v: number = Number.parseFloat(cls[i]);
    if (!Number.isNaN(v) && v > bv) {
      bv = v;
      best = i;
    }
  }
  if (best === 0) { return '黄牌'; }
  if (best === 1) { return '蓝牌'; }
  return '绿牌';
}

/** 解码后的 RGBA 帧：pipeline() 只吃这个。 */
interface RgbaFrame {
  data: ArrayBuffer;
  width: number;
  height: number;
}

@Entry
@Component
struct Index {
  @State lines: string[] = [];
  @State progress: string = '';
  @State busy: boolean = false;
  /** 0 = 原生演示，1 = 端侧 Web 演示，2 = 探针控制台。 */
  @State tab: number = 0;

  // ---- 原生演示页状态
  @State nStatus: string = '未加载模型';
  @State nBackends: string = '';
  @State nPlate: string = '';
  @State nDetail: string = '';
  @State nTiming: string = '';
  @State nLenses: string[] = [];

  private webCtrl: webview.WebviewController = new webview.WebviewController();
  private ctx: common.UIAbilityContext | undefined = undefined;
  private webReady: boolean = false;
  private loadedTab: number = 0;
  private probeStarted: boolean = false;

  /** 原生会话 id（loadModel 返回），-1 表示尚未加载。 */
  private detId: number = -1;
  private recId: number = -1;
  private clsId: number = -1;

  private append(s: string): void {
    this.lines = this.lines.concat([s]);
    hilog.info(DOMAIN, TAG, '%{public}s', s);
  }

  private appendWeb(s: string): void {
    this.lines = this.lines.concat([s]);
    hilog.info(DOMAIN, WEB_TAG, '%{public}s', s);
  }

  private appendNative(s: string): void {
    this.lines = this.lines.concat([s]);
    hilog.info(DOMAIN, NATIVE_TAG, '%{public}s', s);
  }

  private async tick(): Promise<void> {
    await new Promise<void>((resolve: () => void) => {
      setTimeout(() => resolve(), 0);
    });
  }

  private toArrayBuffer(u: Uint8Array): ArrayBuffer {
    return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
  }

  private mimeOf(path: string): string {
    if (path.endsWith('.html')) { return 'text/html'; }
    if (path.endsWith('.js') || path.endsWith('.mjs')) { return 'application/javascript'; }
    if (path.endsWith('.wasm')) { return 'application/wasm'; }
    if (path.endsWith('.css')) { return 'text/css'; }
    if (path.endsWith('.json')) { return 'application/json'; }
    if (path.endsWith('.png')) { return 'image/png'; }
    if (path.endsWith('.jpg') || path.endsWith('.jpeg')) { return 'image/jpeg'; }
    return 'application/octet-stream';
  }

  // ------------------------------------------------------------ 原生演示

  /**
   * 加载三个模型，逐个指定后端。
   *
   * 关键点：loadModel 的第三个参数决定这个会话落到哪个后端，三个会话互相独立 ——
   * 所以「检测 CPU + 识别 NPU + 分类 CPU」这种混合配置是表达得出来的。
   * 返回串里的 `backend=` 是**实际落点**（可能是回退后的 CPU），不是请求值；
   * 想确认是否真的上了 NPU，只能看这个字段。
   */
  private async loadNative(): Promise<void> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    this.nStatus = '加载中…';
    try {
      const t0: number = Date.now();

      const detBytes = c.resourceManager.getRawFileContentSync(MODEL_DET);
      const d: Map<string, string> = parseKv(
        lpr.loadModel('det', this.toArrayBuffer(detBytes), BACKEND_DET));

      const recBytes = c.resourceManager.getRawFileContentSync(MODEL_REC);
      const r: Map<string, string> = parseKv(
        lpr.loadModel('rec', this.toArrayBuffer(recBytes), BACKEND_REC));

      const clsBytes = c.resourceManager.getRawFileContentSync(MODEL_CLS);
      const k: Map<string, string> = parseKv(
        lpr.loadModel('cls', this.toArrayBuffer(clsBytes), BACKEND_CLS));

      if (d.get('ok') !== '1' || r.get('ok') !== '1' || k.get('ok') !== '1') {
        this.nStatus = '加载失败';
        this.appendNative('NATIVE LOAD FAIL det=' + str(d, 'error') +
          ' rec=' + str(r, 'error') + ' cls=' + str(k, 'error'));
        return;
      }

      this.detId = num(d, 'id');
      this.recId = num(r, 'id');
      this.clsId = num(k, 'id');

      this.nBackends = 'det=' + str(d, 'backend') + '  rec=' + str(r, 'backend') +
        '  cls=' + str(k, 'backend');
      this.nStatus = '就绪（' + ((Date.now() - t0) / 1000).toFixed(1) + ' s）';
      this.appendNative('NATIVE LOAD ok ' + this.nBackends);
      this.appendNative('NATIVE TRAIL det[' + str(d, 'attemptLog') + '] rec[' +
        str(r, 'attemptLog') + '] cls[' + str(k, 'attemptLog') + ']');
    } catch (e) {
      const err = e as Error;
      this.nStatus = '异常: ' + err.message;
      this.appendNative('NATIVE LOAD EXCEPTION ' + err.message);
    }
  }

  /**
   * 枚举这台机器上所有物理镜头。
   *
   * 这是 Web 路径做不到的事：ArkWeb 的 getUserMedia 只能拿到系统给的那一路默认流，
   * 无法按镜头类型挑选；原生 Camera Kit 才能把广角 / 超广角 / 长焦逐个列出来并按
   * cameraId 单独打开。
   */
  private listLenses(): void {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    try {
      const mgr = camera.getCameraManager(c);
      const cams: camera.CameraDevice[] = mgr.getSupportedCameras();
      const out: string[] = [];
      for (let i = 0; i < cams.length; i++) {
        const cam: camera.CameraDevice = cams[i];
        const line: string = '#' + cam.cameraId + ' ' + posName(cam.cameraPosition) +
          ' · ' + lensName(cam.cameraType);
        out.push(line);
        this.appendNative('NATIVE LENS id=' + cam.cameraId +
          ' pos=' + cam.cameraPosition + ' type=' + cam.cameraType + ' -> ' + line);
      }
      this.nLenses = out;
      this.appendNative('NATIVE LENS count=' + cams.length);
    } catch (e) {
      const err = e as Error;
      this.appendNative('NATIVE LENS EXCEPTION ' + err.message);
    }
  }

  /**
   * GPU 专项：三个模型逐个请求 `gpu` 后端。
   *
   * 只看 `backend=` 这个**实际落点**字段，不看请求值 —— MS Lite 的后端拒绝是
   * 编译期判否（`IsValid# GPU is not supported`）后静默回退 CPU，
   * 所以「请求了 gpu」和「跑在 gpu 上」是两件事。attemptLog 保留完整尝试链。
   */
  private async gpuProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.appendNative('NATIVE GPU PROBE BEGIN');
    const models: string[] = [MODEL_DET, MODEL_REC, MODEL_CLS];
    for (let i = 0; i < models.length; i++) {
      await this.probeOne(ctx, models[i], 'gpu');
      await this.tick();
    }
    this.appendNative('NATIVE GPU PROBE END');
  }

  /**
   * 裸 head 的 NPU 验收（ADR-004 §4.1 的待办项）。
   *
   * 唯一要回答的问题：把 12 个 rank-5 常量从图里切掉之后（tools/export_bare_head.py），
   * NPU 还拒不拒收？只看 `backend=` 这个**实际落点**。
   *
   * 注意：这不是端到端测试。decode 还在 C++ 里没实现，所以即便落点变成 NPU，
   * 也跑不出车牌 —— 那是下一步。
   */
  private async headProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.appendNative('NATIVE HEAD PROBE BEGIN');
    const models: string[] = [
      'models/lprnet_npufix_fp16.ms',
      'models/lprnet_npufix_fp32.ms'
    ];
    for (let i = 0; i < models.length; i++) {
      await this.probeOne(ctx, models[i], 'nnrt');
      await this.tick();
    }
    this.appendNative('NATIVE HEAD PROBE END');
  }

  /** 把 rawfile 里的 JPEG 解成 RGBA —— pipeline() 只吃 RGBA 缓冲。 */
  private async decodeRgba(file: string): Promise<RgbaFrame | null> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return null;
    }
    const bytes: Uint8Array = c.resourceManager.getRawFileContentSync(file);
    const src: image.ImageSource = image.createImageSource(this.toArrayBuffer(bytes));
    const pm: image.PixelMap = await src.createPixelMap();
    const info: image.ImageInfo = await pm.getImageInfo();
    const w: number = info.size.width;
    const h: number = info.size.height;
    const rgba: ArrayBuffer = new ArrayBuffer(w * h * 4);
    await pm.readPixelsToBuffer(rgba);
    await pm.release();
    await src.release();
    this.appendNative('NATIVE DECODE ' + file + ' ' + w + 'x' + h);
    return { data: rgba, width: w, height: h };
  }

  /**
   * 跑一次原生完整流水线。
   *
   * 全链路都在 C++：letterbox → 检测 → NMS → 单应矫正（双三次）→ 识别 → CTC → 分类。
   * JS 侧只递一张 RGBA 图和三个会话 id，拿回几十字节的结果串。
   */
  private async runNative(file: string): Promise<void> {
    if (this.busy) {
      return;
    }
    if (this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      await this.loadNative();
      if (this.detId < 0) {
        return;
      }
    }
    this.busy = true;
    try {
      const frame: RgbaFrame | null = await this.decodeRgba(file);
      if (frame === null) {
        this.busy = false;
        return;
      }
      const raw: string = lpr.pipeline(frame.data, frame.width, frame.height,
        this.detId, this.recId, this.clsId);
      const kv: Map<string, string> = parseKv(raw);

      if (kv.get('ok') !== '1') {
        this.nStatus = '推理失败';
        this.nPlate = '—';
        this.appendNative('NATIVE PIPE FAIL ' + str(kv, 'error'));
        this.busy = false;
        return;
      }

      const count: number = num(kv, 'count');
      this.nStatus = count > 0 ? '识别成功' : '未检测到车牌';
      if (count === 0) {
        this.nPlate = '—';
        this.nDetail = '';
        this.nTiming = 'total ' + str(kv, 'totalMs') + ' ms';
        this.appendNative('NATIVE PIPE ok=0-plates total=' + str(kv, 'totalMs'));
        this.busy = false;
        return;
      }

      const p0: string = str(kv, 'p0');
      const f: string[] = p0.split(',');
      // 0=code 1=det 2=rec 3=layer 4=rect 5=crop 6=cls 7=chars 8=probs 9=times 10=sum
      const clsArr: string[] = f[6].split('|');
      const colour: string = colourOf(clsArr);
      this.nPlate = f[0] + '  ' + colour;
      this.nDetail = '检测 ' + f[1] + ' · 识别 ' + f[2] + ' · 层数 ' +
        (f[3] === '1' ? '双层' : '单层') + ' · 裁剪 ' + f[5].replace('|', 'x') +
        ' · 字符 ' + f[7];
      this.nTiming = 'total ' + str(kv, 'totalMs') + ' ms   [' + f[9].replace(/\|/g, ' / ') + ']';
      this.appendNative('NATIVE PIPE code=' + f[0] + ' colour=' + colour +
        ' count=' + count + ' total=' + str(kv, 'totalMs') + ' stages=' + f[9] +
        ' cropSum=' + f[10]);
    } catch (e) {
      const err = e as Error;
      this.nStatus = '异常: ' + err.message;
      this.appendNative('NATIVE PIPE EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  /**
   * 同进程内加载两份裸 head（nnrt / cpu），跑同一 CPU-diff 协议，比 checksum。
   *
   * 回答一个必须回答的问题：nnrt 档到底是「NPU 真在算」还是「静默回落 CPU」？
   * 判据很硬 —— 如果静默回落，两次输出必须**逐位相同**（同一份 CPU 代码路径）；
   * 若 checksum 不同，则 NPU 确实参与计算。
   */
  /**
   * Web 路径诊断：真机上跑一次 demo.html 的 bench，把 window.__BENCH 抓回来。
   *
   * 存在的理由：Web 演示（端侧演示页）报 300+ms，而 ADR-004 记的是 6 线程 85ms。
   * 两者差太多，必须搞清楚是「多线程没生效」还是「输入分辨率不同」——
   * __BENCH 里带 threads / isolated / hardwareConcurrency，一眼可辨。
   */
  /**
   * ncnn 路径自证（A 阶段第二步）：加载 ncnn 裸 head → 同一张样本图跑 → 打印三输出
   * L2/maxAbs 与耗时。这些值与 PC 侧 ONNX 同图参考对照，差值即 ncnn 移植保真度。
   * 预处理复用 lpr_pipeline 的 letterbox/NCHW，保证与 PC 参考同源可比。
   */
  private async ncnnProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    try {
      const param: Uint8Array =
        ctx.resourceManager.getRawFileContentSync('models/y5fu_320x_head.ncnn.param');
      const bin: Uint8Array =
        ctx.resourceManager.getRawFileContentSync('models/y5fu_320x_head.ncnn.bin');
      const l: Map<string, string> = parseKv(
        lpr.ncnnLoad(this.toArrayBuffer(param), this.toArrayBuffer(bin), true));
      if (l.get('ok') !== '1') {
        this.appendNative('NCNN LOAD FAIL ' + str(l, 'error'));
        return;
      }
      this.appendNative('NCNN LOAD ok layers=' + str(l, 'layers') +
        ' in=' + str(l, 'inputs') + ' out=' + str(l, 'outputs'));
      const frame: RgbaFrame | null = await this.decodeRgba(SELFTEST_SAMPLE);
      if (frame === null) {
        this.appendNative('NCNN RUN SKIP (no frame)');
        return;
      }
      this.appendNative('NCNN RUN ' + lpr.ncnnRun(frame.data, frame.width, frame.height, 5));
    } catch (e) {
      const err = e as Error;
      this.appendNative('NCNN EXCEPTION ' + err.message);
    }
  }

  private async webDiag(): Promise<void> {
    this.appendNative('WEB DIAG BEGIN demo.html?bench=30&threads=6');
    this.webCtrl.loadUrl('https://lpr.local/demo.html?bench=30&threads=6');
    const res: string = await this.waitBench(90000);
    this.appendNative('WEB DIAG ' + res);
  }

  private async detDriftProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const file: string = 'models/y5fu_320x_head_fp32.ms';
    const backends: string[] = ['nnrt', 'cpu'];
    for (let i = 0; i < backends.length; i++) {
      const be: string = backends[i];
      try {
        const bytes: Uint8Array = ctx.resourceManager.getRawFileContentSync(file);
        const loadRaw: string = lpr.loadModel('drift_' + be, this.toArrayBuffer(bytes), be);
        const li: Map<string, string> = parseKv(loadRaw);
        if (li.get('ok') !== '1') {
          this.appendNative('DET DRIFT ' + be + ' LOAD-FAIL ' + str(li, 'error'));
          continue;
        }
        const id: number = num(li, 'id');
        const b: Map<string, string> = parseKv(lpr.bench(id, 3, 10));
        this.appendNative('DET DRIFT ' + be + ' landed=' + str(li, 'backend') +
          ' L2=' + str(b, 'checksum') + ' maxAbs=' + str(b, 'maxAbs') +
          ' p50=' + str(b, 'p50Ms') + ' mean=' + str(b, 'meanMs') +
          ' dtype=' + str(b, 'outputDtype') + ' elems=' + str(b, 'outputElems'));
      } catch (e) {
        const err = e as Error;
        this.appendNative('DET DRIFT ' + be + ' EXCEPTION ' + err.message);
      }
    }
  }

  /**
   * GPU 终审探针（ADR-008）：调原生 vulkanProbe，把 JSON 切块打进 hilog。
   * hilog 单条会截断，所以按 ~400 字符切块；UI 侧也同步 append。
   * 前置探针（hdc shell）已经确认 const.SystemCapability.Graphic.Vulkan=true、
   * /vendor/etc/vulkan/ 存在 —— 这一步回答的是「App 到底能不能 dlopen loader、
   * 枚不枚得出带 COMPUTE 队列的 GPU」。
   */
  private vulkanProbeRun(): void {
    try {
      const raw: string = lpr.vulkanProbe();
      this.appendNative('VULKAN PROBE len=' + raw.length.toString());
      const chunk: number = 400;
      for (let i = 0; i < raw.length; i += chunk) {
        const end: number = Math.min(i + chunk, raw.length);
        this.appendNative('VULKAN PROBE [' + Math.floor(i / chunk).toString() + '] ' +
          raw.substring(i, end));
      }
    } catch (e) {
      const err = e as Error;
      this.appendNative('VULKAN PROBE EXCEPTION ' + err.message);
    }
  }

  // ------------------------------------------------------------ Web 探针（原逻辑）

  private makeResponse(file: string): WebResourceResponse | null {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return null;
    }
    let data: Uint8Array;
    try {
      data = c.resourceManager.getRawFileContentSync(file);
    } catch (e) {
      this.appendWeb('WEBPROBE MISSING-RAWFILE ' + file);
      return null;
    }
    const resp = new WebResourceResponse();
    const heads: Header[] = [
      { headerKey: 'Cross-Origin-Opener-Policy', headerValue: 'same-origin' },
      { headerKey: 'Cross-Origin-Embedder-Policy', headerValue: 'require-corp' },
      { headerKey: 'Cross-Origin-Resource-Policy', headerValue: 'same-origin' },
      { headerKey: 'Cache-Control', headerValue: 'no-cache' }
    ];
    resp.setResponseHeader(heads);
    resp.setResponseData(this.toArrayBuffer(data));
    resp.setResponseEncoding('utf-8');
    resp.setResponseMimeType(this.mimeOf(file));
    resp.setResponseCode(200);
    resp.setReasonMessage('OK');
    return resp;
  }

  private collectWeb(): void {
    this.webCtrl.runJavaScript('document.getElementById("log").innerText')
      .then((r: string) => {
        const arr: string[] = r.split('\n');
        for (let i = 0; i < arr.length; i++) {
          const ln: string = arr[i].trim();
          if (ln.length > 0) {
            this.appendWeb(ln);
          }
        }
        this.appendWeb('WEBPROBE END');
      })
      .catch((e: Error) => {
        this.appendWeb('WEBPROBE EXCEPTION ' + e.message);
      });
  }

  private async probeOne(ctx: common.UIAbilityContext, file: string, backend: string): Promise<void> {
    const short: string = file.replace('models/', '');
    const rm = ctx.resourceManager;

    let bytes: Uint8Array;
    try {
      bytes = rm.getRawFileContentSync(file);
    } catch (e) {
      this.append('MATRIX ' + short + ' ' + backend + ' MISSING-RAWFILE');
      return;
    }

    const loadRaw: string = lpr.loadModel(short + '|' + backend, this.toArrayBuffer(bytes), backend);
    const li: Map<string, string> = parseKv(loadRaw);

    if (li.get('ok') !== '1') {
      this.append('MATRIX ' + short + ' ' + backend + ' BUILD-FAIL err=' +
        str(li, 'error') + ' trail=' + str(li, 'attemptLog'));
      return;
    }

    const id: number = num(li, 'id');
    const landed: string = str(li, 'backend');

    const bRaw: string = lpr.bench(id, 5, 30);
    const b: Map<string, string> = parseKv(bRaw);

    if (b.get('ok') !== '1') {
      this.append('MATRIX ' + short + ' ' + backend + ' LANDED=' + landed +
        ' BENCH-FAIL err=' + str(b, 'error'));
      return;
    }

    const cks: number = num(b, 'checksum');
    const ck16: number = num(b, 'checksumAsFp16');
    let verdict: string = 'PASS';
    if (landed.indexOf('CPU') < 0 && cks > 1e6) {
      verdict = 'SUSPECT-FP16-MISLABEL';
    }

    this.append('MATRIX ' + short + ' ' + backend + ' LANDED=' + landed +
      ' in=' + str(li, 'inputShape') + '/' + str(li, 'inputFormat') +
      ' p50=' + str(b, 'p50Ms') + 'ms mean=' + str(b, 'meanMs') + 'ms' +
      ' L2=' + cks.toFixed(4) + ' L2asFp16=' + ck16.toFixed(4) +
      ' maxAbs=' + str(b, 'maxAbs') +
      ' dtype=' + str(b, 'outputDtype') +
      ' verdict=' + verdict);
  }

  private async probeAll(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.lines = [];

    try {
      const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
      this.append('MATRIX NNRT-DEVICES ' + lpr.listNnrtDevices());

      const plan: Row[] = [];
      for (let i = 0; i < MODELS_FULL.length; i++) {
        for (let j = 0; j < FULL_BACKENDS.length; j++) {
          plan.push({ model: MODELS_FULL[i], backend: FULL_BACKENDS[j] });
        }
      }
      for (let i = 0; i < MODELS_NNRT_GPU.length; i++) {
        plan.push({ model: MODELS_NNRT_GPU[i], backend: 'nnrt' });
        plan.push({ model: MODELS_NNRT_GPU[i], backend: 'gpu' });
      }

      for (let i = 0; i < plan.length; i++) {
        this.progress = (i + 1).toString() + '/' + plan.length.toString();
        await this.probeOne(ctx, plan[i].model, plan[i].backend);
        await this.tick();
      }
      this.progress = 'done';
      this.append('MATRIX END');
    } catch (e) {
      const err = e as Error;
      this.append('MATRIX EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve: () => void) => {
      setTimeout(() => resolve(), ms);
    });
  }

  private async waitBench(timeoutMs: number): Promise<string> {
    const t0: number = Date.now();
    let retries: number = 0;
    let lastErr: string = '';
    while (Date.now() - t0 < timeoutMs) {
      await this.sleep(700);
      try {
        const r: string = await this.webCtrl.runJavaScript(
          '(window.__BENCH && window.__BENCH.done) ? JSON.stringify(window.__BENCH) : ""');
        if (r !== undefined && r !== null && r.length > 2) {
          return r;
        }
      } catch (e) {
        const err = e as Error;
        retries = retries + 1;
        lastErr = err.message;
      }
    }
    return 'TIMEOUT retries=' + retries.toString() + ' lastErr=' + lastErr;
  }

  private async mtSweep(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.appendWeb('MT SWEEP BEGIN counts=' + MT_COUNTS.join(',') + ' iters=' + MT_ITERS.toString());
    try {
      for (let i = 0; i < MT_COUNTS.length; i++) {
        const t: number = MT_COUNTS[i];
        this.progress = 'MT ' + (i + 1).toString() + '/' + MT_COUNTS.length.toString() + ' t=' + t.toString();
        this.webCtrl.loadUrl('https://lpr.local/demo.html?bench=' + MT_ITERS.toString() +
          '&threads=' + t.toString());
        const res: string = await this.waitBench(MT_TIMEOUT_MS);
        this.appendWeb('MT threads=' + t.toString() + ' ' + res);
      }
      this.appendWeb('MT SWEEP END');
    } catch (e) {
      const err = e as Error;
      this.appendWeb('MT EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  private async runAll(): Promise<void> {
    if (RUN_MATRIX) {
      await this.probeAll();
    }
    await this.mtSweep();
  }

  private switchTab(t: number): void {
    if (this.busy) {
      return;
    }
    if (this.loadedTab === t && this.webReady) {
      this.tab = t;
      return;
    }
    this.tab = t;
    if (!this.webReady) {
      return;
    }
    this.loadedTab = t;
    // 原生页（0）不需要 Web 承载，只把 Web 压成 1vp 即可。
    if (t === 0) {
      return;
    }
    try {
      this.webCtrl.loadUrl(t === 1 ? DEMO_URL : PROBE_URL);
    } catch (e) {
      const err = e as Error;
      this.appendWeb('TAB SWITCH FAILED ' + err.message);
    }
  }

  private onPageEnd(): void {
    this.webReady = true;
    if (this.loadedTab === 2 && !this.probeStarted) {
      this.probeStarted = true;
      this.collectWeb();
      this.sleep(3000).then(() => {
        this.runAll();
      });
    }
  }

  private async requestCamera(): Promise<void> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    const perms: Permissions[] = ['ohos.permission.CAMERA'];
    try {
      const res = await abilityAccessCtrl.createAtManager().requestPermissionsFromUser(c, perms);
      this.appendWeb('CAMERA PERM auth=' + res.authResults.join(','));
    } catch (e) {
      const err = e as Error;
      this.appendWeb('CAMERA PERM FAIL ' + err.message);
    }
  }

  aboutToAppear(): void {
    this.ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    // 开 Web 调试开关：ArkWeb 会起一个 domain socket，配合
    //   hdc fport tcp:9222 localabstract:webview_devtools_remote_<pid>
    // 电脑侧就能用 CDP 直接驱动这个页面（Runtime.evaluate）。
    webview.WebviewController.setWebDebuggingAccess(true);
    this.requestCamera();
    // 自检：等 UI 起来（Web 组件绑定 + 权限弹窗不阻塞）再跑，全部结果进 hilog。
    if (AUTO_SELFTEST) {
      this.sleep(1500).then(() => {
        this.vulkanProbeRun();
        this.listLenses();
        this.loadNative().then(() => {
          this.runNative(SELFTEST_SAMPLE).then(() => {
            // det 的 CPU-diff 量化：确定性输入下的 L2/maxAbs，对照
            // tools/det_head_ref_checksum.py 的 PC 参考值 → 量化 NNRT 漂移。
            this.appendNative('DET BENCH ' + lpr.bench(this.detId, 3, 10));
            this.detDriftProbe();
            this.headProbe().then(() => {
              this.ncnnProbe().then(() => {
                this.webDiag();
              });
            });
          });
        });
      });
    }
  }

  @Builder
  tabButton(label: string, idx: number) {
    Text(label)
      .fontSize(14)
      .fontWeight(this.tab === idx ? FontWeight.Medium : FontWeight.Normal)
      .fontColor(this.tab === idx ? '#ffffff' : '#4b5563')
      .backgroundColor(this.tab === idx ? '#0a0c10' : '#f1f3f6')
      .padding({ left: 14, right: 14, top: 7, bottom: 7 })
      .borderRadius(8)
      .onClick(() => {
        this.switchTab(idx);
      })
  }

  @Builder
  nativeHeader() {
    Column({ space: 8 }) {
      Text('原生端到端 · 麒麟 8020')
        .fontSize(17)
        .fontWeight(FontWeight.Bold)
        .width('100%')
      Text('C++ 完整流水线（检测→NMS→单应矫正→识别→CTC→分类）· 三模型独立后端')
        .fontSize(11)
        .fontColor('#8a8f98')
        .width('100%')

      Row({ space: 8 }) {
        Button('加载模型')
          .fontSize(13)
          .onClick(() => {
            this.loadNative();
          })
        Button('枚举镜头')
          .fontSize(13)
          .backgroundColor('#2b5f3a')
          .onClick(() => {
            this.listLenses();
          })
      }
      .width('100%')

      Text('后端 ' + (this.nBackends.length > 0 ? this.nBackends : '—'))
        .fontSize(11)
        .fontColor('#8a8f98')
        .width('100%')

      Text(this.nStatus)
        .fontSize(12)
        .fontColor('#4b5563')
        .width('100%')

      Row({ space: 8 }) {
        ForEach(NATIVE_SAMPLES, (s: string) => {
          Button(s.replace('assets/samples/', '').replace('.jpg', ''))
            .fontSize(12)
            .backgroundColor('#3a3f47')
            .onClick(() => {
              this.runNative(s);
            })
        }, (s: string) => s)
      }
      .width('100%')

      if (this.nPlate.length > 0) {
        Column({ space: 4 }) {
          Text(this.nPlate)
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .fontColor('#0a0c10')
            .width('100%')
          Text(this.nDetail)
            .fontSize(11)
            .fontColor('#4b5563')
            .width('100%')
          Text(this.nTiming)
            .fontSize(11)
            .fontColor('#8a8f98')
            .width('100%')
        }
        .width('100%')
        .padding(10)
        .backgroundColor('#f1f3f6')
        .borderRadius(8)
        .alignItems(HorizontalAlign.Start)
      }

      if (this.nLenses.length > 0) {
        Column({ space: 2 }) {
          Text('物理镜头 ' + this.nLenses.length.toString())
            .fontSize(11)
            .fontColor('#8a8f98')
            .width('100%')
          ForEach(this.nLenses, (l: string) => {
            Text(l)
              .fontSize(11)
              .fontColor('#4b5563')
              .width('100%')
          }, (l: string) => l)
        }
        .width('100%')
        .padding(8)
        .backgroundColor('#f8f9fb')
        .borderRadius(8)
        .alignItems(HorizontalAlign.Start)
      }
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
  }

  @Builder
  probeHeader() {
    Column({ space: 8 }) {
      Text('加速器矩阵探针 · 麒麟 8020')
        .fontSize(17)
        .fontWeight(FontWeight.Bold)
        .width('100%')
      Text('MindSpore Lite NDK · NNRT / GPU / KIRIN_NPU / CPU 逐项实测')
        .fontSize(11)
        .fontColor('#8a8f98')
        .width('100%')
      Row({ space: 8 }) {
        Button(this.busy ? ('探测中 ' + this.progress) : '重新探测')
          .fontSize(13)
          .onClick(() => {
            this.probeAll();
          })
        Button('清空')
          .fontSize(13)
          .backgroundColor('#3a3f47')
          .onClick(() => {
            this.lines = [];
          })
        Button('Web 探测')
          .fontSize(13)
          .backgroundColor('#2b5f3a')
          .onClick(() => {
            this.collectWeb();
          })
        Button('WASM 多线程')
          .fontSize(13)
          .backgroundColor('#5a3b8f')
          .onClick(() => {
            this.mtSweep();
          })
      }
      .width('100%')
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
  }

  @Builder
  probeLog() {
    Scroll() {
      Column({ space: 2 }) {
        ForEach(this.lines, (item: string) => {
          Text(item)
            .fontSize(10)
            .fontFamily('monospace')
            .fontColor('#d7dce3')
            .width('100%')
        }, (item: string, idx: number) => idx.toString() + '_' + item)
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start)
    }
    .layoutWeight(1)
    .width('100%')
    .padding(8)
    .backgroundColor('#14161a')
    .borderRadius(8)
  }

  build() {
    Column({ space: 8 }) {
      Row({ space: 6 }) {
        this.tabButton('原生演示', 0)
        this.tabButton('端侧演示', 1)
        this.tabButton('探针控制台', 2)
      }
      .width('100%')
      .padding({ left: 12, right: 12, top: 12, bottom: 4 })

      if (this.tab === 0) {
        Column() {
          this.nativeHeader()
        }
        .width('100%')
        .padding({ left: 12, right: 12 })
      }

      if (this.tab === 2) {
        Column() {
          this.probeHeader()
        }
        .width('100%')
        .padding({ left: 12, right: 12 })
      }

      // 单例 Web：src 固定为演示页，切页靠 loadUrl。原生页（tab 0）不需要画面，
      // 压成 1vp 且透明 —— 它仍是一台正常的引擎，只是不给它显示空间。
      Web({ src: DEMO_URL, controller: this.webCtrl })
        .onInterceptRequest((event) => {
          if (event === undefined || event === null) {
            return null;
          }
          const url: string = event.request.getRequestUrl();
          const prefix: string = 'https://lpr.local/';
          if (url.indexOf(prefix) !== 0) {
            return null;
          }
          const q: number = url.indexOf('?');
          const raw: string = q < 0 ? url.substring(prefix.length) : url.substring(prefix.length, q);
          // 路径里的中文文件名会被浏览器百分号编码（crop-0-%E6%B4%A5B6H920.jpg），
          // 而 rawfile 里存的是原字面名 —— 不解码就会一路 MISSING-RAWFILE。
          let file: string = raw;
          try {
            file = decodeURIComponent(raw);
          } catch (e) {
            // 非法转义序列：原样使用，交给 getRawFileContentSync 报 MISSING
          }
          return this.makeResponse(file);
        })
        .onPageEnd(() => {
          this.onPageEnd();
        })
        .onPermissionRequest((event) => {
          if (event === undefined || event === null) {
            return;
          }
          // ProtectedResourceType 是 ArkUI 侧的全局 ambient 枚举（web.d.ts），
          // 既不在 webview 命名空间下，也不由 @kit.ArkWeb 再导出 —— 直接裸用。
          const res: string[] = event.request.getAccessibleResource();
          if (res.indexOf(ProtectedResourceType.VIDEO_CAPTURE) >= 0) {
            event.request.grant(res);
            this.appendWeb('CAMERA GRANT VIDEO_CAPTURE');
            return;
          }
          event.request.deny();
          this.appendWeb('CAMERA DENY ' + res.join(','));
        })
        .width('100%')
        .layoutWeight(this.tab === 1 ? 1 : 0)
        .height(this.tab === 1 ? '100%' : 1)
        .opacity(this.tab === 1 ? 1 : 0)

      if (this.tab === 2) {
        Column() {
          this.probeLog()
        }
        .layoutWeight(1)
        .width('100%')
        .padding({ left: 12, right: 12 })
      }
    }
    .width('100%')
    .height('100%')
    .padding({ bottom: 12 })
    .backgroundColor('#ffffff')
  }
}
