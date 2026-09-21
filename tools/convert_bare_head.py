"""Convert the bare-head y5fu to .ms, fp16 and fp32, for the NPU acceptance test.

The question this answers is narrow and binary: once the 12 rank-5 constants are gone
(see tools/export_bare_head.py), does the NPU stop rejecting the model?
It is NOT a "does it work end to end" test — the decode is not implemented yet.
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

WORK = r"C:\Users\26671\lpr-harmony\models_ms\head"
SRC = os.path.join(WORK, "y5fu_320x_dec.onnx")
OUT_JSON = os.path.join(WORK, "_convert_dec.json")

INPUT_NAME = "input"
INPUT_SHAPE = "1,3,320,320"


def clear_readonly(p):
    if os.path.isfile(p):
        try:
            os.chmod(p, 0o666)
        except OSError:
            pass


def main():
    if not os.path.isfile(SRC):
        print("missing", SRC)
        return 1

    env = dict(os.environ)
    env["PATH"] = LIB_DIR + os.pathsep + CONVERTER_DIR + os.pathsep + env.get("PATH", "")
    rep = {}

    for tag, fp16 in [("fp16", "on"), ("fp32", "off")]:
        out_prefix = os.path.join(WORK, "y5fu_320x_dec_" + tag)
        ms = out_prefix + ".ms"
        clear_readonly(ms)
        if os.path.isfile(ms):
            os.remove(ms)
        cmd = [
            CONVERTER, "--fmk=ONNX", "--modelFile=" + SRC,
            "--outputFile=" + out_prefix, "--fp16=" + fp16,
            "--inputShape=%s:%s" % (INPUT_NAME, INPUT_SHAPE),
        ]
        e = {"cmd": " ".join(cmd)}
        try:
            p = subprocess.run(cmd, capture_output=True, cwd=CONVERTER_DIR,
                               env=env, timeout=1200)
            e["rc"] = p.returncode
            e["stdout"] = p.stdout.decode("utf-8", "replace")[-1200:]
            e["stderr"] = p.stderr.decode("utf-8", "replace")[-1200:]
        except subprocess.TimeoutExpired:
            e["status"] = "TIMEOUT"
            rep[tag] = e
            continue
        e["status"] = "OK" if os.path.isfile(ms) else "FAIL"
        if os.path.isfile(ms):
            e["ms"] = ms
            e["ms_size"] = os.path.getsize(ms)
        rep[tag] = e
        print(tag, e["status"], e.get("ms_size"), e.get("rc"))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("written", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
