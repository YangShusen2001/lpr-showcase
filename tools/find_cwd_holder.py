#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列出所有进程的「当前工作目录」，找出谁占着某个目录（Windows 下被占为 cwd 的目录无法改名）。

原理：ReadProcessMemory 读目标进程 PEB -> ProcessParameters -> CurrentDirectory.DosPath。
仅支持 64 位 Windows（PEB 偏移 0x20 / RTL_USER_PROCESS_PARAMETERS 的 CurrentDirectory 偏移 0x38）。

用法：python find_cwd_holder.py <目录绝对路径>
"""
import ctypes
import ctypes.wintypes as wintypes
import os
import sys

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

PROCESS_BASIC_INFORMATION = None


class UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("Length", ctypes.c_ushort),
        ("MaximumLength", ctypes.c_ushort),
        ("Buffer", ctypes.c_void_p),
    ]


class CURDIR(ctypes.Structure):
    _fields_ = [("DosPath", UNICODE_STRING), ("Handle", ctypes.c_void_p)]


def read_mem(handle, addr, size):
    buf = ctypes.create_string_buffer(size)
    read = ctypes.c_size_t(0)
    ok = k32.ReadProcessMemory(handle, ctypes.c_void_p(addr), buf, size, ctypes.byref(read))
    if not ok or read.value != size:
        return None
    return buf.raw


def cwd_of(pid):
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        # PEB 地址在 PBI 的第 2 个指针
        pbi = ctypes.create_string_buffer(48)
        ret = ctypes.c_ulong(0)
        status = ntdll.NtQueryInformationProcess(h, 0, pbi, 48, ctypes.byref(ret))
        if status != 0:
            return None
        peb = ctypes.c_void_p.from_buffer(pbi, 8).value
        if not peb:
            return None
        # PEB.ProcessParameters 在 x64 偏移 0x20
        raw = read_mem(h, peb + 0x20, 8)
        if raw is None:
            return None
        params = ctypes.c_void_p.from_buffer_copy(raw).value
        if not params:
            return None
        # RTL_USER_PROCESS_PARAMETERS.CurrentDirectory 在 x64 偏移 0x38
        raw = read_mem(h, params + 0x38, ctypes.sizeof(CURDIR))
        if raw is None:
            return None
        cur = CURDIR.from_buffer_copy(raw)
        if not cur.DosPath.Buffer or cur.DosPath.Length == 0:
            return ""
        data = read_mem(h, cur.DosPath.Buffer, cur.DosPath.Length)
        if data is None:
            return None
        return data.decode("utf-16-le", "replace")
    finally:
        k32.CloseHandle(h)


def proc_name(pid):
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return "?"
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(1024)
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return "?"
    finally:
        k32.CloseHandle(h)


def main():
    target = os.path.normcase(os.path.normpath(sys.argv[1])).rstrip("\\")
    hits = []
    for pid in range(4, 65536):
        cwd = cwd_of(pid)
        if not cwd:
            continue
        norm = os.path.normcase(os.path.normpath(cwd)).rstrip("\\")
        if norm == target:
            hits.append((pid, proc_name(pid), cwd))
    for pid, name, cwd in hits:
        print(f"PID {pid:>6}  {name:<24} cwd = {cwd}")
    print(f"命中 {len(hits)} 个进程把 {sys.argv[1]} 当作当前工作目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
