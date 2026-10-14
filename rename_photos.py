#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rename_photos — 按 EXIF 时间排序后重命名原始 JPG

PhotoRenamer 从根目录的路径解析出位置、摄影师、相机、组号,
从 EXIF 读取拍摄日期, 然后按拍摄时间顺序给每张 JPG 分配一个
4 位序号, 生成新文件名:

    {Index}_{YYYYMMDD}{Position}{GroupID}{Photographer}{Camera}{OriginalIndex}.JPG

示例:
    ZRQ04753.JPG -> 0001_20260324JM02ZRA14753.JPG
"""

import os
import re
from datetime import datetime


class PhotoRenamer:
    """按 EXIF 日期排序并重命名原始 JPG。

    用法:
        renamer = PhotoRenamer(root_dir)
        mapping = renamer.scan().preview()   # 查看新旧文件名映射
        renamer.rename()                     # 实际执行重命名

    元数据优先从路径自动解析 (如 .../20260324-A1-ZR-JM-SC-ZLF/PHOTO/02[F]_GM);
    解析失败或需覆盖时, 可通过构造参数传入 date/position/group_id/
    photographer/camera。
    """

    def __init__(self, root_dir, date=None, position=None, group_id=None,
                 photographer=None, camera=None):
        self.root_dir = os.path.abspath(root_dir)
        self.date = date
        self.position = position
        self.group_id = group_id
        self.photographer = photographer
        self.camera = camera
        self.mapping = []  # [(old_path, new_path), ...]

    def scan(self):
        """读取图片元数据, 解析路径, 生成按时间排序的新旧文件名映射。
        返回 self, 方便链式调用。"""
        self._parse_path()
        records = []
        for name in sorted(os.listdir(self.root_dir)):
            if not name.upper().endswith(".JPG"):
                continue
            path = os.path.join(self.root_dir, name)
            if not os.path.isfile(path):
                continue
            dt = self._read_datetime(path)
            idx = self._extract_index(name)
            records.append((dt, idx, name, path))
        # 按 EXIF 时间升序; 时间相同按原文件名排序, 保证稳定
        records.sort(key=lambda x: (x[0], x[2]))
        self.mapping = []
        for i, (dt, idx, name, old_path) in enumerate(records, 1):
            date_str = dt.strftime("%Y%m%d") if dt else self.date or ""
            new_name = "%04d_%s%s%s%s%s%s.JPG" % (
                i, date_str, self.position or "",
                self.group_id or "", self.photographer or "",
                self.camera or "", idx or "")
            new_path = os.path.join(self.root_dir, new_name)
            # 万一冲突, 追加序号
            new_path = self._unique_path(new_path)
            self.mapping.append((old_path, new_path))
        return self

    def preview(self):
        """返回 [(old_name, new_name), ...] 列表。"""
        return [(os.path.basename(o), os.path.basename(n))
                for o, n in self.mapping]

    def rename(self, dry_run=False):
        """执行重命名。dry_run=True 时不真正移动文件, 只返回映射。"""
        if not self.mapping:
            self.scan()
        if dry_run:
            return self.preview()
        moved = []
        for old_path, new_path in self.mapping:
            os.rename(old_path, new_path)
            moved.append((old_path, new_path))
        # 映射中的 old_path 已失效, 重新扫描以便再次 rename
        self.mapping = []
        return [(os.path.basename(o), os.path.basename(n)) for o, n in moved]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _parse_path(self):
        """从目录路径解析日期、相机、摄影师、位置、组号。"""
        parts = self.root_dir.split(os.sep)
        # 找形如 20260324-A1-ZR-JM-SC-ZLF 的目录段
        for p in parts:
            m = re.match(r"^(\d{8})-(.+)$", p)
            if not m:
                continue
            if not self.date:
                self.date = m.group(1)
            tokens = m.group(2).split("-")
            if not self.camera and len(tokens) > 0:
                self.camera = tokens[0]
            if not self.photographer and len(tokens) > 1:
                self.photographer = tokens[1]
            if not self.position and len(tokens) > 2:
                self.position = tokens[2]
            break
        # 组号取根目录名开头的数字 (如 02[F]_GM -> 02)
        base = os.path.basename(self.root_dir)
        m = re.match(r"^(\d+)", base)
        if m and not self.group_id:
            self.group_id = m.group(1)

    @staticmethod
    def _read_datetime(path):
        """读取 EXIF DateTimeOriginal (含 SubSec), 返回 datetime; 失败返回 None。"""
        try:
            import pyexiv2
            with pyexiv2.Image(path) as img:
                meta = img.read_exif()
            dt = meta.get("Exif.Photo.DateTimeOriginal", "")
            sub = meta.get("Exif.Photo.SubSecTimeOriginal", "0") or "0"
            if not dt:
                return None
            # 补足 6 位微秒
            sub = (sub + "000000")[:6]
            return datetime.strptime("%s.%s" % (dt, sub),
                                     "%Y:%m:%d %H:%M:%S.%f")
        except Exception:
            return None

    @staticmethod
    def _extract_index(name):
        """从原文件名中提取数字部分并去前导零, 如 ZRQ04753 -> 4753。"""
        m = re.search(r"\d+", name)
        if not m:
            return ""
        return str(int(m.group()))

    @staticmethod
    def _unique_path(path):
        """若目标已存在, 在扩展名前追加 _2, _3..."""
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 2
        while True:
            cand = "%s_%d%s" % (base, i, ext)
            if not os.path.exists(cand):
                return cand
            i += 1


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    dry = "--dry-run" in sys.argv
    renamer = DolphinRenamer(root)
    mapping = renamer.scan().rename(dry_run=dry)
    for old_name, new_name in mapping:
        print("%s -> %s" % (old_name, new_name))
