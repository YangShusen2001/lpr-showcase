"""Build App, install only a fresh successful artifact, capture startup logs."""
import os
import subprocess
from pathlib import Path
import json

ROOT = Path(r"C:\Users\26671\lpr-harmony\LprDemo")
OUT = Path(r"C:\Users\26671\Desktop\车牌识别\_evidence")
HDC = r"D:\Tools\Huawei\HarmonyOSSDK\hmscore\3.1.0\toolchains\hdc.exe"
env = os.environ.copy()
env.pop("NODE_OPTIONS", None)
env.update(DEVECO_HOME="D:/IDE/DevEco_Studio", DEVECO_SDK_HOME="D:/IDE/DevEco_Studio/sdk", NODE_HOME="D:/IDE/DevEco_Studio/tools/node", JAVA_HOME="D:/IDE/DevEco_Studio/jbr")
env["PATH"] = "D:/IDE/DevEco_Studio/jbr/bin;" + env.get("PATH", "")
status = []

def run(label, args, timeout=180):
    p = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    text = (p.stdout + p.stderr).decode("utf-8", "replace")
    (OUT / (label + ".log")).write_text(text, encoding="utf-8")
    status.append({"step": label, "returncode": p.returncode})
    (OUT / "a3_app_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if p.returncode:
        raise SystemExit(p.returncode)
    return text

text = run("a3_hap_build", ["D:/IDE/DevEco_Studio/tools/node/node.exe", "D:/IDE/DevEco_Studio/tools/hvigor/bin/hvigorw.js", "assembleHap", "--mode", "module", "-p", "product=default", "-p", "buildMode=debug", "--no-daemon"], 480)
if "BUILD SUCCESSFUL" not in text:
    raise SystemExit("No build success marker; refusing stale install")
text = run("a3_install", [HDC, "-t", "4CY9K25614046328", "install", "-r", str(ROOT / "entry/build/default/outputs/default/entry-default-signed.hap")])
if "success" not in text.lower():
    raise SystemExit("No installation success marker")
run("a3_stop", [HDC, "-t", "4CY9K25614046328", "shell", "aa force-stop com.shusen.lprdemo"])
run("a3_start", [HDC, "-t", "4CY9K25614046328", "shell", "aa start -a EntryAbility -b com.shusen.lprdemo"])
# Capture a bounded live log window, not a sleep/poll loop.
p = subprocess.Popen([HDC, "-t", "4CY9K25614046328", "shell", "hilog"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
try:
    data, _ = p.communicate(timeout=55)
except subprocess.TimeoutExpired:
    p.kill()
    data, _ = p.communicate()
text = data.decode("utf-8", "replace")
(OUT / "a3_device_full.log").write_text(text, encoding="utf-8")
(OUT / "a3_device_summary.txt").write_text("\n".join(x for x in text.splitlines() if any(k in x for k in ("NCNN", "ncnn", "Maleoon", "NATIVE PIPE", "LprNative"))), encoding="utf-8")
