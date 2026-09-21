# -*- coding: utf-8 -*-
"""
在 Desktop 上创建一个纯 ASCII 的目录联接（junction），指向中文名的项目目录。

为什么用 junction 而不是 rename：
  Windows 禁止重命名任何"正被进程当作当前工作目录(cwd)"的目录。
  本机有 88 个进程（含 WorkBuddy.exe / sandbox-cli.exe 自身）把
  C:\\Users\\26671\\Desktop\\车牌识别 当作 cwd，所以只要 WorkBuddy 还开着就
  一定重命名失败。junction 只是在别处新建一个"快捷方式式"的目录项，
  不触碰源目录本身，因此不受该限制，DevEco Studio 也能正常跟随它。

用法：
  python make_ascii_junction.py            # 创建 junction
  python make_ascii_junction.py --check    # 只检查，不创建
"""
import json
import os
import subprocess
import sys

SRC = r"C:\Users\26671\Desktop\车牌识别"
DST = r"C:\Users\26671\Desktop\lpr-showcase"
RESULT = r"C:\Users\26671\Desktop\_junction_result.json"


def is_junction(path):
    """用 os.path.islink / reparse 属性判断，避免把真目录误判为 junction。"""
    if not os.path.exists(path):
        return False
    try:
        # Python 3.8+ 在 Windows 上对 junction 也返回 True
        return os.path.islink(path)
    except OSError:
        return False


def main():
    check_only = "--check" in sys.argv
    r = {
        "src": SRC,
        "dst": DST,
        "src_exists": os.path.isdir(SRC),
        "dst_exists": os.path.exists(DST),
        "dst_is_junction": is_junction(DST),
        "check_only": check_only,
    }

    if not r["src_exists"]:
        r["action"] = "SKIP: source not found"
    elif r["dst_exists"] and not r["dst_is_junction"]:
        r["action"] = "SKIP: dst exists and is a real dir/file, refuse to touch"
    elif r["dst_is_junction"]:
        r["action"] = "SKIP: junction already present"
    elif check_only:
        r["action"] = "CHECK: would create junction"
    else:
        ps = (
            "New-Item -ItemType Junction -Path '{dst}' -Target '{src}' "
            "| Select-Object -ExpandProperty FullName"
        ).format(dst=DST, src=SRC)
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
        )
        r["rc"] = p.returncode
        r["stdout"] = p.stdout.decode("utf-8", "replace").strip()
        r["stderr"] = p.stderr.decode("utf-8", "replace").strip()
        r["action"] = "CREATE: attempted"
        r["dst_exists_after"] = os.path.exists(DST)
        r["dst_is_junction_after"] = is_junction(DST)

    # 通过 junction 读一个真实文件，证明它能透明转发
    probe = os.path.join(DST, "index.html")
    r["probe_index_html_via_junction"] = os.path.isfile(probe)
    r["probe_size"] = os.path.getsize(probe) if os.path.isfile(probe) else None

    # DevEco 真正需要的是 ASCII 路径下的工程：确认鸿蒙工程本身已是 ASCII
    r["deveco_project"] = r"C:\Users\26671\lpr-harmony\LprDemo"
    r["deveco_project_exists"] = os.path.isdir(r["deveco_project"])

    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    print("done ->", RESULT)


if __name__ == "__main__":
    main()
