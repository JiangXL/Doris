#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从选中的源目录复制图片，以还原 structure.json 中文件结构
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


STRUCTURE_FILE = Path("structure.json")


def select_root_dir() -> Path:
    """交互式选择根目录：列出当前目录下的子文件夹供用户选择。"""
    dirs = sorted(
        d for d in Path(".").iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    print("请选择源根目录：")
    for i, d in enumerate(dirs, 1):
        print(f"  {i}. {d.name}")
    while True:
        choice = input("输入编号（或直接输入路径）: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(dirs):
            return dirs[int(choice) - 1]
        p = Path(choice)
        if p.is_dir():
            return p
        print("无效选择，请重试。")

def main():
    parser = argparse.ArgumentParser(
        description="按 structure.json 将所选根目录中的文件复制到对应的所有目录位置。"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="源根目录（不提供时交互式选择）。",
    )

    args = parser.parse_args()

    if not STRUCTURE_FILE.is_file():
        print(f"错误：结构文件不存在: {STRUCTURE_FILE}，请先运行 scan_dph_folder.py")
        return 1

    src_dir = args.src_dir if args.src_dir is not None else select_root_dir()
    if not src_dir.is_dir():
        print(f"错误：源目录不存在: {src_dir}")
        return 1
    print(f"源根目录: {src_dir}")

    with open(STRUCTURE_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for name, rel_dirs in sorted(mapping.items()):
        if name not in src_files:
            not_found.append(name)
            continue

        src_path = src_files[name]


        for rel_dir in rel_dirs:
            target_dir = src_dir / rel_dir
            target_path = target_dir / name

        copied += copied_for_file
        skipped_same += same_for_file
        skipped_in_place += in_place_for_file

        # 只有该文件确实需要保留在非根目录的副本时，
        # 才删除顶层源文件；纯根目录文件保留不动。
        if args.remove_source and not args.dry_run:
            non_root_copies = copied_for_file + same_for_file
            if non_root_copies > 0:
                src_path.unlink()
                removed_source.append(str(src_path))

    if not_found:
        print(f"\n未在 {src_dir} 中找到的文件 ({len(not_found)} 个):")
        for name in not_found[:20]:
            print(f"  - {name}")
        if len(not_found) > 20:
            print(f"  ... 还有 {len(not_found) - 20} 个")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
