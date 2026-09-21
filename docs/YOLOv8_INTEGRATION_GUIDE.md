# YOLOv8 集成指南

## 📋 已完成的工作

- ✅ YOLOv8n.pt 已转换为 ONNX 格式 (`assets/models/yolov8n.onnx`, 12.3 MB)
- ✅ YOLOv8 检测器代码已创建 (`yolov8_detect.cpp` / `yolov8_detect.h`)
- ✅ LprSessions 结构体已添加 `useYoloV8` 和 `detSize` 字段

## 🔧 下一步操作

### 1. 修改 lpr_pipeline.cpp 以支持 YOLOv8

在 `LprPipeline` 函数中（约第 805 行），添加 YOLOv8 分支：

```cpp
// Before: if (s.detNcnn) { ... } else if (MsRunMulti(s.det, ...)

// Add after the existing detector code:
if (s.useYoloV8) {
  // YOLOv8 specific processing
  const bool detNhwc = true;  // YOLOv8 uses NHWC format
  std::vector<float> detIn = detNhwc ? LprToNhwc(lb.img, true) : LprToNchw(lb.img, true);
  
  MsRunMulti(s.det, detIn.data(), detOuts, err);
  if (detOuts.empty()) {
    err = "YOLOv8 detector returned no tensors";
    return false;
  }
  
  // Decode YOLOv8 output
  auto yolo_output = DetectYoloV8(lb, detOuts[0], s.confThresh, s.iouThresh);
  
  // Convert to standard format
  for (const auto& box : yolo_output) {
    PlateResult item;
    item.rect[0] = static_cast<int>(box[0]);
    item.rect[1] = static_cast<int>(box[1]);
    item.rect[2] = static_cast<int>(box[2]);
    item.rect[3] = static_cast<int>(box[3]);
    item.detScore = box[4];
    out.push_back(item);
  }
  
  continue;  // Skip the rest of the pipeline for this detection
}
```

### 2. 更新 CMakeLists.txt 或构建配置

确保 `yolov8_detect.cpp` 被编译到项目中。在鸿蒙工程中，这通常通过 `build-profile.json5` 或 `.ohos` 配置文件管理。

### 3. 更新模型加载逻辑

在 `ms_engine.cpp` 或相关文件中，添加 YOLOv8 模型的加载逻辑：

```cpp
// Load yolov8n.onnx instead of y5fu_320x_sim.onnx
std::vector<char> model_bytes = ReadRawFile("models/yolov8n.onnx");
MsSession* det_session = MsLoad(model_bytes, "cpu", err);
```

### 4. 测试与对比

编译并安装到手机后，进行性能对比测试：

```bash
# Enable YOLOv8 in the app settings or via ADB
hdc shell setprop persist.sys.yolov8.enabled true

# Run self-test and compare results
hilog | grep "E2E preprocess"
```

## 📊 预期性能提升

基于公开数据和经验：
- **YOLOv5**: ~25ms (CPU, 320x320)
- **YOLOv8**: ~18ms (CPU, 640x640) - **预计提升 28%**

实际测试结果可能因硬件和实现细节而异。

## ⚠️ 注意事项

1. **输入尺寸差异**: YOLOv5 使用 320x320，YOLOv8 使用 640x640
2. **输出格式差异**: YOLOv8 使用 anchor-free 检测头，解码逻辑不同
3. **内存占用**: YOLOv8 模型更大（12.3MB vs 2.2MB）
4. **精度**: YOLOv8 通常具有更高的 mAP，但需要验证在车牌检测任务上的表现

## 🔄 回滚方案

如果 YOLOv8 效果不佳，可以通过设置 `useYoloV8 = false` 回退到 YOLOv5。

---

**最后更新**: 2026-09-19
