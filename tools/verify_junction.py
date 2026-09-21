# -*- coding: utf-8 -*-
"""用 Python 复核 junction（Python 源码默认 UTF-8，中文路径不会乱码）。"""
import json
import os
import subprocess

SRC = r"C:\Users\26671\Desktop\车牌识别"
DST = r"C:\Users\26671\Desktop\lpr-showcase"
OUT = r"C:\Users\26671\Desktop\_junction_result.json"

r = {}
r["src_exists"] = os.path.isdir(SRC)
r["dst_exists"] = os.path.isdir(DST)
try:
    r["dst_entries"] = len(os.listdir(DST))
except OSError as e:
    r["dst_entries"] = "ERR: %s" % e

# 通过联接读到真实内容（抽查几个关键文件）
probes = [
    "index.html",
    "demo.html",
    os.path.join("assets", "models", "y5fu_320x_sim.onnx.json"),
    os.path.join("assets", "ort", "ort-wasm-simd-threaded.wasm"),
    os.path.join("tools", "sync_rawfile.py"),
]
r["probes"] = {}
for p in probes:
    fp = os.path.join(DST, p)
    r["probes"][p] = {
        "exists": os.path.isfile(fp),
        "size": os.path.getsize(fp) if os.path.isfile(fp) else None,
    }

# 关键：写穿透测试的"只读版"——比较联接路径与真实路径的 stat，证明是同一份
r["same_file_same_inode"] = {}
for p in ["index.html", os.path.join("assets", "js", "demo.js")]:
    a = os.stat(os.path.join(DST, p))
    b = os.stat(os.path.join(SRC, p))
    r["same_file_same_inode"][p] = (a.st_ino == b.st_ino and a.st_size == b.st_size)

# realpath 会不会把 ASCII 路径解回中文？（DevEco/Node 行为预警）
r["realpath_of_junction"] = os.path.realpath(DST)

# DevEco 工程路径 ASCII 检查
dp = r"C:\Users\26671\lpr-harmony\LprDemo"
r["deveco_project"] = dp
r["deveco_exists"] = os.path.isdir(dp)
r["deveco_is_ascii"] = dp.isascii()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)
print("ok")
