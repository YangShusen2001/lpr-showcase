#!/usr/bin/env python3
"""
Download PP-YOLOE-S ONNX model from PaddleDetection official repository.
This script downloads the pre-converted ONNX model directly.
"""

import os
import urllib.request
from pathlib import Path

# Model configuration
MODEL_NAME = "pp_yoloe_s"
OUTPUT_DIR = Path(__file__).parent / ".." / "assets" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download URL for PP-YOLOE-S ONNX model (multiple sources)
DOWNLOAD_URLS = {
    "pp_yoloe_s": [
        "https://github.com/PaddlePaddle/PaddleDetection/releases/download/v2.7.0/pp_yoloe_s.onnx",
        "https://paddledet.bj.bcebos.com/pp_yoloe_s.onnx",
        "https://modelscope.cn/api/v1/modelscope/repo/file?repo_type=model&repo_id=PaddleDetection/paddlex_ppyoloe_s&file=pp_yoloe_s.onnx",
    ],
}

def download_model(model_name, output_dir):
    """Download the ONNX model from multiple sources."""
    if model_name not in DOWNLOAD_URLS:
        print(f"Unknown model: {model_name}")
        return False
    
    urls = DOWNLOAD_URLS[model_name]
    output_file = output_dir / f"{model_name}.onnx"
    
    for i, url in enumerate(urls, 1):
        print(f"Trying source {i}/{len(urls)}: {url}...")
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
        print("Next step: Run 'python convert_onnx_ms.py assets/models/pp_yoloe_s.onnx pp_yoloe_s'")
    else:
        print("\nFailed to download model. Please check your network connection.")
