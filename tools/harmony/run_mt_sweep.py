"""WASM 多线程扫描的采集脚本：装机 → 冷启 → 轮询抓 LprWeb 日志。

与 run_matrix_web.py 的两点区别：
1. 轮询抓取而不是等满再一次性 dump —— 扫描要跑 2~3 分钟，hilog 是环形缓冲，
   一次性 dump 时前面的 MT 行可能已经被系统日志挤掉。每 20 s 抓一次就不会丢。
2. 抓到的行按「去掉 hilog 前缀后的正文」去重，重复轮询不会把同一条记多遍。
"""
import subprocess
import time
import re

HDC = r"D:\IDE\DevEco_Studio\sdk\default\openharmony\toolchains\hdc.exe"
HAP = r"C:\Users\26671\lpr-harmony\LprDemo\entry\build\default\outputs\default\entry-default-signed.hap"
OUT = r"C:\Users\26671\lpr-harmony\mt_sweep.txt"
MAX_S = 420
POLL_S = 20

PAT = re.compile(r"LprMatrix|LprWeb")
BODY = re.compile(r"Lpr(?:Web|Matrix):\s*(.*)$")


def sh(args, timeout=300):
    r = subprocess.run([HDC, "shell"] + args, capture_output=True, timeout=timeout)
    return r.stdout.decode("utf-8", "replace")


def flush(install, start, elapsed, done, lines):
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("=== INSTALL ===\n%s\n=== START ===\n%s\n" % (install, start))
        f.write("=== POLLS ===\nelapsed=%.0fs done=%s lines=%d\n=== LOG ===\n"
                % (elapsed, done, len(lines)))
        f.write("\n".join(lines))


r = subprocess.run([HDC, "install", "-r", HAP], capture_output=True, timeout=900)
install = (r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")).strip()
print("install:", install)

sh(["hilog", "-r"])
sh(["aa", "force-stop", "com.shusen.lprdemo"])
time.sleep(2)
start = sh(["aa", "start", "-a", "EntryAbility", "-b", "com.shusen.lprdemo"]).strip()
print("start:", start)

seen, keys, done = [], set(), False
t0 = time.time()
while time.time() - t0 < MAX_S:
    time.sleep(POLL_S)
    try:
        txt = sh(["hilog", "-x"], timeout=300)
    except subprocess.TimeoutExpired:
        continue
    for ln in txt.splitlines():
        if not PAT.search(ln):
            continue
        m = BODY.search(ln)
        key = m.group(1).strip() if m else ln.strip()
        if key in keys:
            continue
        keys.add(key)
        seen.append(ln)
        if "MT SWEEP END" in key:
            done = True
    flush(install, start, time.time() - t0, done, seen)
    print("poll elapsed=%.0fs lines=%d done=%s" % (time.time() - t0, len(seen), done))
    if done:
        break

flush(install, start, time.time() - t0, done, seen)
print("FINAL lines=%d done=%s -> %s" % (len(seen), done, OUT))
