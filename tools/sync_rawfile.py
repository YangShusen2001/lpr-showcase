#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把网页版车牌识别交付物同步进鸿蒙工程 rawfile/。

设计要点
--------
1. 只同步运行期真正需要的文件（见 MANIFEST），不整目录搬运 —— 网页版 assets/ 共 89MB，
   其中 ort-wasm-simd-threaded.jsep.wasm(27MB) 与 yolo-v9-*.onnx(23MB) 是实验产物，
   浏览器 demo 从不用到；搬进去只会把 HAP 撑到 90MB+。
2. 对 HTML 做一次"仅限副本"的绝对地址归一化：源 index.html 被外部编辑器写回了
   http://127.0.0.1:10504/static-html/<hash>/ 这类本地预览地址，手机上必然 404。
   源文件是用户在实时编辑的，绝不动；只改 rawfile 里的副本。
3. 幂等：重复执行会先清空 rawfile/ 再重建，并打印逐文件校验（大小）。

用法
----
python sync_rawfile.py [--src 项目根] [--dst rawfile 目录] [--check]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

# 项目根从脚本自身位置推导（<root>/tools/sync_rawfile.py），不写死绝对路径 ——
# 这样项目文件夹改名/搬家都不会失效。
DEFAULT_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DST = r"C:\Users\26671\lpr-harmony\LprDemo\entry\src\main\resources\rawfile"

# 运行期必需清单（相对项目根）。
#
# 2026-09-18 起手机端是**纯原生**：不再打包任何网页、样式、JS 与 ORT 运行时，
# rawfile 里只留「原生推理真正读的字节」—— 样本图，以及 lpr-harmony 侧投放的
# .ms 模型与 ncnn param/bin（见 KEEP）。网页版交付物（index.html / demo.html /
# assets/ort/**）仍留在项目仓库里给 PC 浏览器用，只是不再进 HAP。
MANIFEST = [
    # 原生演示的三个样本（Index.ets 的 NATIVE_SAMPLES 与自检样本都在内）
    "assets/samples/hlpr-test.jpg",
    "assets/samples/scene-2.jpg",
    "assets/samples/crop-0-津B6H920.jpg",
]

# 壳专属文件：网页探针页（webprobe.html）随 ArkWeb 一起拆掉了，现在为空。
# 保留这个列表，是为了让「先 rmtree 再重建」的幂等实现继续可用。
# 元素 = (相对项目根的源路径, rawfile 内的目标相对路径)
EXTRA = []

# 由 lpr-harmony 侧的转换脚本（convert_to_ms.py / exp_convert.py / 图手术脚本）单独投放的
# rawfile 内容 —— 本脚本既不产出也不理解它们，但重建时绝不能顺手删掉。
# 路径相对 rawfile 根（文件或目录均可）。
KEEP = [
    "models",   # 7 个 .ms 原生模型，约 12 MB，重新转换要几分钟
]

# 只对 HTML 副本生效的改写：把残留的本地预览绝对地址压回相对路径
LOCAL_ABS = re.compile(r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?/static-html/[0-9a-f]+/")


def rewrite_html(text: str):
    """返回 (新文本, 替换次数)。"""
    new, n = LOCAL_ABS.subn("", text)
    return new, n


def sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--check", action="store_true", help="只校验，不写盘")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if not os.path.isdir(src):
        print(f"[x] 源目录不存在: {src}")
        return 1

    # 清单 + 壳专属文件，统一成 (源相对路径, 目标相对路径) 二元组
    entries = [(rel, rel) for rel in MANIFEST] + list(EXTRA)

    missing = [s_rel for (s_rel, _) in entries if not os.path.isfile(os.path.join(src, s_rel))]
    if missing:
        print("[x] 清单里有文件缺失：")
        for m in missing:
            print("    -", m)
        return 1

    rows, total = [], 0
    for s_rel, d_rel in entries:
        s = os.path.join(src, s_rel)
        size = os.path.getsize(s)
        total += size
        rows.append({"rel": s_rel, "dst": d_rel, "size": size, "sha1": sha1(s), "rewritten": 0})

    if args.check:
        ok = True
        for r in rows:
            d = os.path.join(dst, r["dst"])
            if not os.path.isfile(d):
                print(f"[!] 缺失 {r['dst']}")
                ok = False
            elif os.path.getsize(d) != r["size"] and not r["dst"].endswith(".html"):
                print(f"[!] 大小不符 {r['dst']}")
                ok = False
        for rel in KEEP:
            if not os.path.exists(os.path.join(dst, rel)):
                print(f"[!] 缺失（保留项） {rel}")
                ok = False
        print(f"[check] 共 {len(rows)} 项，{'一致' if ok else '有不一致'}")
        return 0 if ok else 1

    # 幂等重建：先给 KEEP 里的内容找个临时落脚点，重建完再搬回原位。
    # 直接 rmtree 会把 .ms 原生模型一起清掉（约 12 MB，重转一次要几分钟）。
    stash = None
    kept = []
    if os.path.isdir(dst):
        stash = tempfile.mkdtemp(prefix="rawfile-keep-")
        for rel in KEEP:
            s = os.path.join(dst, rel)
            if os.path.exists(s):
                shutil.move(s, os.path.join(stash, rel))
                kept.append(rel)
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    for r in rows:
        s = os.path.join(src, r["rel"])
        d = os.path.join(dst, r["dst"].replace("/", os.sep))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if r["dst"].endswith(".html"):
            raw = open(s, "rb").read().decode("utf-8")
            new, n = rewrite_html(raw)
            open(d, "wb").write(new.encode("utf-8"))
            r["rewritten"] = n
        else:
            shutil.copyfile(s, d)

    if stash is not None:
        for rel in kept:
            shutil.move(os.path.join(stash, rel), os.path.join(dst, rel))
        shutil.rmtree(stash, ignore_errors=True)

    # 清单必须落在 resources/ 之外：hvigor 的 CompileResource 会扫描该目录下的所有条目，
    # 出现一个非目录文件就会以 11211101 "not a directory" 直接中断构建。
    # dst = <proj>/entry/src/main/resources/rawfile -> 上跳 5 级 = <proj> 的父目录
    probe = dst.rstrip("\\/")
    for _ in range(6):
        probe = os.path.dirname(probe)
    manifest_out = os.path.join(probe, "rawfile-manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump({"totalBytes": total, "files": rows}, f, ensure_ascii=False, indent=2)

    print(f"[ok] 同步 {len(rows)} 个文件 -> {dst}")
    print(f"[ok] 合计 {total / 1048576:.2f} MiB")
    if kept:
        print(f"[ok] 原样保留（非本脚本产出）: {', '.join(kept)}")
    for r in rows:
        note = f"  (改写本地绝对地址 {r['rewritten']} 处)" if r["rewritten"] else ""
        where = r["rel"] if r["dst"] == r["rel"] else f"{r['rel']} -> {r['dst']}"
        print(f"     {r['size']:>10,}  {where}{note}")
    print(f"[ok] 清单: {manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
