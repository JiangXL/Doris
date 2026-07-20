#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 03[F]_GM 下指定文件夹（SYN, Quality below 60, NOTE, NN, Mutiple, MISC, MCP, DphIDxxx）
的目录结构，记录每个文件所在的相对目录。
生成结构映射文件 structure.json（供 move_files.py 使用）。
"""

import json
import os
import re
import glob
from collections import defaultdict
from pathlib import Path


SRC_DIR = Path("03[F]_GM")
OUTPUT_JSON = Path("structure.json")
OUTPUT_TXT = Path("structure.txt")

# 仅扫描这些文件夹：精确名称 + DphIDxxx（DphID 后跟 3 位数字）
ALLOWED_NAMES = {"SYN", "Quality below 60", "NOTE", "NN", "Mutiple", "MISC", "MCP"}
DPHID_RE = re.compile(r"^DphID\d{3}$")


def is_allowed_dir(name):
    return name in ALLOWED_NAMES or DPHID_RE.match(name) is not None

def scan_dph_folder():
    if not SRC_DIR.is_dir():
        print(f"错误：源目录不存在: {SRC_DIR}")
        return 1

    # dictionary: filename -> list of relative_dirs （保留同一文件出现的所有位置）
    mapping = defaultdict(list)

    # 仅遍历允许的顶层文件夹
    subdirs = sorted(
        d for d in SRC_DIR.iterdir() if d.is_dir() and is_allowed_dir(d.name)
    )
    for sub in subdirs:
        for root, dirs, files in os.walk(sub):
            dirs.sort()
            files.sort()
            for name in files:
                # 仅扫描 JPG 文件（不区分大小写）
                if Path(name).suffix.lower() != ".jpg":
                    continue
                full_path = Path(root) / name
                rel_dir = full_path.parent.relative_to(SRC_DIR).as_posix()
                rel_dir = "." if rel_dir == "" else rel_dir
                mapping[name].append(rel_dir)

    # 扫描顶层文件里的 JPG 图片（最佳背鳍）
    top_jpg_paths = sorted(glob.glob( os.path.join(glob.escape(SRC_DIR), "*.JPG")))
    for top_jpg_name in top_jpg_paths:
        mapping[os.path.basename(top_jpg_name)].append(".")

    # 转换为普通 dict 以便序列化
    mapping = {name: dirs for name, dirs in mapping.items()}

    # 写入 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)

    # 写入文本清单（保留所有位置）
    total_entries = sum(len(dirs) for dirs in mapping.values())
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Source: {SRC_DIR}\n")
        f.write(f"# Unique files: {len(mapping)}\n")
        f.write(f"# Total entries (including duplicates): {total_entries}\n")

    print(f"扫描完成：{len(mapping)} 个唯一文件，{total_entries} 个总条目")
    print(f"JSON 映射: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    scan_dph_folder()
