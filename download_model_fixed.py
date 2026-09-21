#!/usr/bin/env python3
"""Download YOLOv8-N ONNX model with proper SSL handling."""

import ssl
import urllib.request
from pathlib import Path

# Create SSL context that doesn't verify certificates (for testing)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Model configuration
MODEL_NAME = "yolov8n"
OUTPUT_DIR = Path(__file__).parent / "assets" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download URL for YOLOv8-N ONNX model (try multiple sources)
DOWNLOAD_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
    "https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.onnx",
    "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.onnx",
]

def download_model():
    """Download the ONNX model from multiple sources."""
    output_file = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
    
    for i, url in enumerate(DOWNLOAD_URLS, 1):
        print(f"Trying source {i}/{len(DOWNLOAD_URLS)}: {url}...")
        try:
            # Create request with proper headers
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            # Download with SSL context
            response = urllib.request.urlopen(req, context=ctx, timeout=60)
            data = response.read()
            
            # Save to file
            with open(output_file, 'wb') as f:
                f.write(data)
            
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
    success = download_model()
    if success:
        print("\nModel ready for conversion!")
        print("Next step: Run 'python convert_onnx_ms.py assets/models/yolov8n.onnx yolov8n'")
    else:
        print("\nFailed to download model.")
