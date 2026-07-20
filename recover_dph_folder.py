#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 structure.json 的映射，将文件从源文件夹复制到目标目录结构中。

structure.json 格式（由 scan_dph_folder.py 生成）：
    {"ZRQ06280.JPG": ["Mutiple/DphID001", "NN/DphID001_DphID003"], ...}

用法：
    1. python copy_files.py SOURCE_DIR DEST_DIR [--json structure.json] 
    2. python copy_files.py  

同一文件若对应多个目录，会复制到每一个目录。
"""

import os
import json
import shutil
import sys
from pathlib import Path

def select_folder(title="Select Folder") -> str | None:
    """弹出系统文件对话框，让用户选择一个文件夹。

    Args:
        title: 对话框标题

    Returns:
        选中的文件夹绝对路径；如果用户取消则返回 None
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes("-topmost", True)  # 确保对话框置顶
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()

    if folder_path and os.path.isdir(folder_path):
        return os.path.abspath(folder_path)
    return None

def delete_previous_folder(dest_dir):
    print("delete")
    '''
    FOLDER_NAMES = {"SYN", "Quality below 60", "NOTE", "NN", "Mutiple", "MISC", "MCP"}
    DPHID_RE = re.compile(r"^DphID\d{3}$")
    for folder in FOLDER_NAMES:
        if Path(folder).exists():
            shutil.rmtree(folder)
    '''

def recovery_dph_folder(json_file, source_dir, dest_dir):
    with open(json_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    copied = 0       # 实际复制的条目数
    skipped = 0      # 目标已存在而跳过的条目数
    missing = []     # 源文件夹中找不到的文件
    errors = []      # 复制失败的条目

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    for name in sorted(mapping):
        src_file = source_dir /  name
        if not src_file.is_file():
            missing.append(name)
            continue

        for rel_dir in mapping[name]:
            file_dest_dir = dest_dir if rel_dir == "." else dest_dir /  rel_dir
            dest_file = file_dest_dir / name

            if dest_file.exists():
                skipped += 1
                continue

            try:
                file_dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
            except OSError as e:
                errors.append(f"{name} -> {rel_dir}: {e}")
                continue
            copied += 1

    total = sum(len(dirs) for dirs in mapping.values())
    print(f"  JSON 条目总数:   {total}")
    print(f"  已复制:          {copied}")
    print(f"  已存在跳过:      {skipped}")
    print(f"  源文件缺失:      {len(missing)}")
    print(f"  复制失败:        {len(errors)}")

    if missing:
        print("\n源文件夹中未找到的文件（前 20 个）：")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... 以及另外 {len(missing) - 20} 个")

    if errors:
        print("\n复制失败的条目：")
        for err in errors:
            print(f"  {err}")

    return 1 if errors else 0

if __name__ == "__main__":
    if len(sys.argv) == 2:
        source_dir = sys.argv[1]
    elif len(sys.argv) == 3:
        source_dir = sys.argv[2]
        dest_dir = sys.argv[3]
    elif len(sys.argv) == 4:
        source_dir = sys.argv[1]
        dest_dir = sys.argv[2]
        json_file = sys.argv[3]
    else:
        source_dir = select_folder(title="选中原始相片目录")
        if source_dir is None:
            print("No folder selected. Exiting.")
            sys.exit(0)
        dest_dir = "./"
        json_file = "structure.json"
        print(f"Selected folder: {source_dir}")
    recovery_dph_folder(json_file, source_dir, dest_dir)
