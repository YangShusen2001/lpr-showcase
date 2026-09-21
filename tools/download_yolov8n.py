#!/usr/bin/env python3
"""
Download YOLOv8-N ONNX model from Ultralytics GitHub releases.
YOLOv8 is faster and more efficient than YOLOv5, with better NPU support.
"""

import os
import urllib.request
from pathlib import Path

# Model configuration
MODEL_NAME = "yolov8n"
OUTPUT_DIR = Path(__file__).parent / ".." / "assets" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download URL for YOLOv8-N ONNX model (GitHub releases)
DOWNLOAD_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
    "https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.onnx",
]

def download_model(model_name, output_dir):
    """Download the ONNX model from multiple sources."""
    output_file = output_dir / f"{model_name}.onnx"
    
    for i, url in enumerate(DOWNLOAD_URLS, 1):
        print(f"Trying source {i}/{len(DOWNLOAD_URLS)}: {url}...")
        try:
            urllib.request.urlretrieve(url, output_file)
            file_size = output_file.stat().st_size
            print(f"Successfully downloaded from source {i}: {output_file} ({file_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    # Clean up partial download
    if output_file.exists():
        output_file.unlink()
    
    print("All download sources failed.")
    return False

if __name__ == "__main__":
    success = download_model(MODEL_NAME, OUTPUT_DIR)
    if success:
        print("\nModel ready for conversion to MindSpore Lite format.")
        print("Next step: Run 'python convert_onnx_ms.py assets/models/yolov8n.onnx yolov8n'")
    else:
        print("\nFailed to download model.")
