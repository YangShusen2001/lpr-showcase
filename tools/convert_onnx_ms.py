"""Generic ONNX -> MindSpore Lite .ms converter (fp16 + fp32).

Same wrapper as tools/convert_bare_head.py, parameterised by a job table so new
candidate models can be added without editing code.
"""
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

MS = r"D:\Tools\mindspore-lite\mindspore-lite-2.6.0-win-x64"
CONVERTER_DIR = os.path.join(MS, "tools", "converter", "converter")
CONVERTER = os.path.join(CONVERTER_DIR, "converter_lite.exe")
LIB_DIR = os.path.join(MS, "tools", "converter", "lib")

# (src onnx, output prefix, input tensor name, input shape)
JOBS = [
    (r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet_npufix.onnx",
     r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet_npufix",
     "input", "1,3,24,94"),
]


def clear_readonly(p):
    if os.path.isfile(p):
        try:
            os.chmod(p, 0o666)
        except OSError:
            pass


def main():
    env = dict(os.environ)
    env["PATH"] = LIB_DIR + os.pathsep + CONVERTER_DIR + os.pathsep + env.get("PATH", "")
    report = {}

    for src, prefix, in_name, in_shape in JOBS:
        if not os.path.isfile(src):
            print("missing", src)
            continue
        entry = {}
        for tag, fp16 in [("fp16", "on"), ("fp32", "off")]:
            out_prefix = "%s_%s" % (prefix, tag)
            ms = out_prefix + ".ms"
            clear_readonly(ms)
            if os.path.isfile(ms):
                os.remove(ms)
            cmd = [
                CONVERTER, "--fmk=ONNX", "--modelFile=" + src,
                "--outputFile=" + out_prefix, "--fp16=" + fp16,
                "--inputShape=%s:%s" % (in_name, in_shape),
            ]
            e = {"cmd": " ".join(cmd)}
            try:
                p = subprocess.run(cmd, capture_output=True, cwd=CONVERTER_DIR,
                                   env=env, timeout=1800)
                e["rc"] = p.returncode
                e["stdout"] = p.stdout.decode("utf-8", "replace")[-1500:]
                e["stderr"] = p.stderr.decode("utf-8", "replace")[-1500:]
            except subprocess.TimeoutExpired:
                e["status"] = "TIMEOUT"
                entry[tag] = e
                continue
            e["status"] = "OK" if os.path.isfile(ms) else "FAIL"
            if os.path.isfile(ms):
                e["ms"] = ms
                e["ms_size"] = os.path.getsize(ms)
            entry[tag] = e
            print(os.path.basename(prefix), tag, e["status"], e.get("ms_size"), e.get("rc"))
        report[os.path.basename(prefix)] = entry

    out_json = os.path.join(os.path.dirname(JOBS[0][0]), "_convert.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("written", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
