"""Run the approved ncnn Vulkan build and save UTF-8 evidence."""
from pathlib import Path
import subprocess
import os
import json

ROOT = Path(r"C:\Users\26671\lpr-harmony")
OUT = Path(r"C:\Users\26671\Desktop\车牌识别\_evidence")
SDK = Path(r"D:\IDE\DevEco_Studio\sdk\default\openharmony\native")
CMAKE = SDK / "build-tools/cmake/bin/cmake.exe"
BUILD = ROOT / "third_party/ncnn_build_ohos_vk"
env = os.environ.copy()
env.pop("NODE_OPTIONS", None)
env["PATH"] = str(CMAKE.parent) + os.pathsep + env.get("PATH", "")
results = []

def run(label, args, cwd=None):
    with (OUT / (label + ".log")).open("wb") as f:
        p = subprocess.run([str(a) for a in args], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    results.append({"step": label, "returncode": p.returncode})
    (OUT / "a3_build_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if p.returncode:
        raise SystemExit(p.returncode)

run("a3_vk_configure", [CMAKE, "-S", ROOT / "third_party/ncnn", "-B", BUILD,
    "-DNCNN_VULKAN=ON", "-DNCNN_BENCHMARK=ON", "-DNCNN_STDIO=ON", "-DNCNN_STRING=ON"])
run("a3_vk_build", [CMAKE, "--build", BUILD, "-j", "4"])
lib = BUILD / "src/libncnn.so"
results.append({"library": str(lib), "exists": lib.is_file(), "bytes": lib.stat().st_size if lib.is_file() else 0})
(OUT / "a3_build_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
