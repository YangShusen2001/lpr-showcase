# -*- coding: utf-8 -*-
"""
把三个车牌 ONNX 转成 MindSpore Lite 的 .ms。

要点（来自技能 mindspore-lite-windows-converter-fix）：
- converter 的 DLL 依赖需从 2.3.0 包补齐（本机已补齐）
- 动态形状必须用 --inputShape 固定（本机三模型本来就是静态，仍显式传以防万一）
- 转换出的 .ms 带只读属性，要清掉，否则 hvigor CompileResource 删不掉 intermediates
  副本报 Error Code: 11204003
"""
import json
import os
import shutil
import subprocess
import sys

MS = r"D:\Tools\mindspore-lite\mindspore-lite-2.6.0-win-x64"
CONVERTER_DIR = os.path.join(MS, "tools", "converter", "converter")
CONVERTER = os.path.join(CONVERTER_DIR, "converter_lite.exe")
LIB_DIR = os.path.join(MS, "tools", "converter", "lib")

SRC_MODELS = r"C:\Users\26671\Desktop\车牌识别\assets\models"
WORK = r"C:\Users\26671\lpr-harmony\models_ms"
OUT_JSON = os.path.join(WORK, "_convert_result.json")

# (name, input_name, input_shape)
MODELS = [
    ("y5fu_320x_sim", "input", "1,3,320,320"),
    ("rpv3_mdict_160_r3", "data", "1,3,48,160"),
    ("litemodel_cls_96x_r1", "data", "1,3,96,96"),
]


def find_src(name):
    for ext in (".onnx", ".onnx.json"):
        p = os.path.join(SRC_MODELS, name + ext)
        if os.path.isfile(p):
            return p
    return None


def clear_readonly(path):
    if not os.path.isfile(path):
        return False
    try:
        os.chmod(path, 0o666)
        return not (os.stat(path).st_mode & 0o200 == 0)
    except OSError:
        return False


def main():
    os.makedirs(WORK, exist_ok=True)

    env = dict(os.environ)
    # converter 需要自己的 lib 目录在 PATH 最前
    env["PATH"] = LIB_DIR + os.pathsep + CONVERTER_DIR + os.pathsep + env.get("PATH", "")

    report = {"converter": CONVERTER, "converter_exists": os.path.isfile(CONVERTER), "models": {}}

    if not report["converter_exists"]:
        report["fatal"] = "converter_lite.exe not found"
    else:
        for name, in_name, in_shape in MODELS:
            e = {"input_name": in_name, "input_shape": in_shape}
            src = find_src(name)
            if not src:
                e["status"] = "SKIP"
                e["error"] = "source model not found"
                report["models"][name] = e
                continue

            # 拷成 .onnx 名，避开任何扩展名相关的解析差异
            local_onnx = os.path.join(WORK, name + ".onnx")
            shutil.copyfile(src, local_onnx)
            e["src"] = src
            e["src_size"] = os.path.getsize(local_onnx)

            out_prefix = os.path.join(WORK, name)
            ms_file = out_prefix + ".ms"
            if os.path.isfile(ms_file):
                clear_readonly(ms_file)
                os.remove(ms_file)

            cmd = [
                CONVERTER,
                "--fmk=ONNX",
                "--modelFile=" + local_onnx,
                "--outputFile=" + out_prefix,
                "--fp16=on",
                "--inputShape=%s:%s" % (in_name, in_shape),
            ]
            e["cmd"] = " ".join(cmd)
            try:
                p = subprocess.run(cmd, capture_output=True, cwd=CONVERTER_DIR, env=env, timeout=900)
                e["rc"] = p.returncode
                e["stdout"] = p.stdout.decode("utf-8", "replace")[-4000:]
                e["stderr"] = p.stderr.decode("utf-8", "replace")[-4000:]
            except subprocess.TimeoutExpired:
                e["status"] = "TIMEOUT"
                report["models"][name] = e
                continue

            e["ms_exists"] = os.path.isfile(ms_file)
            if e["ms_exists"]:
                e["ms_size"] = os.path.getsize(ms_file)
                clear_readonly(ms_file)
                e["ms_readonly_cleared"] = not os.stat(ms_file).st_mode & 0o200 == 0
                e["status"] = "OK"
            else:
                e["status"] = "FAIL"

            report["models"][name] = e

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()
