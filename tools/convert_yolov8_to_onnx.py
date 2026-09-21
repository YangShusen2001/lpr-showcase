#!/usr/bin/env python3
"""
Convert YOLOv8n.pt to ONNX format for deployment on HarmonyOS NPU/GPU/CPU.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Configuration
MODEL_PT = Path(r"C:\Users\26671\Downloads\yolov8n.pt")
OUTPUT_DIR = Path(__file__).parent / ".." / "assets" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ONNX = OUTPUT_DIR / "yolov8n.onnx"

def convert_to_onnx():
    """Convert YOLOv8 model to ONNX format."""
    if not MODEL_PT.exists():
        print(f"Error: Model file not found: {MODEL_PT}")
        return False
    
    print(f"Loading model: {MODEL_PT}")
    model = YOLO(str(MODEL_PT))
    
    print("Exporting to ONNX format...")
    # Export with imgsz=640 (standard detection size)
    success = model.export(
        format='onnx',
        imgsz=640,
        simplify=True,  # Simplify the model for better performance
        dynamic=False,  # Fixed input size for better optimization
        half=False,     # Use FP32 for maximum compatibility
        device='cpu',   # Use CPU
        batch=1         # Batch size 1
    )
    
    if success and OUTPUT_ONNX.exists():
        file_size = OUTPUT_ONNX.stat().st_size
        print(f"✓ Successfully converted: {OUTPUT_ONNX} ({file_size / 1024 / 1024:.2f} MB)")
        return True
    else:
        print("✗ Conversion failed")
        return False

if __name__ == "__main__":
    success = convert_to_onnx()
    if success:
        print("\nModel ready for MindSpore Lite conversion!")
        print("Next step: Run 'python convert_onnx_ms.py assets/models/yolov8n.onnx yolov8n'")
    else:
        print("\nFailed to convert model.")
