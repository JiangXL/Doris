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
import sys
from datetime import datetime


def _require_pyexiv2():
    try:
        import pyexiv2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "本程序需要 pyexiv2 读取 EXIF, 请先安装: pip install pyexiv2"
        ) from exc


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
        self.records = []       # [(dt, idx, name, path, ctime), ...]
        self.mapping = []       # [(old_path, new_path), ...]
        self._read_exif = False  # records 中的 dt 是否来自 EXIF

    def scan(self, start_index=1, sort_by="name", progress_callback=None):
        """读取图片元数据, 解析路径, 生成按时间排序的新旧文件名映射。
        返回 self, 方便链式调用。"""
        self._parse_path()
        self.records = []
        read_exif = (sort_by == "exif")
        if read_exif:
            _require_pyexiv2()
        self._read_exif = read_exif

        candidates = []
        for name in sorted(os.listdir(self.root_dir)):
            if not name.upper().endswith(".JPG"):
                continue
            path = os.path.join(self.root_dir, name)
            if not os.path.isfile(path):
                continue
            candidates.append((name, path))

        total = len(candidates)
        for i, (name, path) in enumerate(candidates, 1):
            dt = self._read_datetime(path) if read_exif else None
            idx = self._extract_index(name)
            ctime = self._file_creation_time(path)
            self.records.append((dt, idx, name, path, ctime))
            if progress_callback:
                progress_callback(i, total)

        self._sort_records(sort_by)
        return self._build_mapping(start_index, progress_callback)

    def _sort_records(self, sort_by):
        """根据排序方式对 self.records 进行排序。"""
        if sort_by == "exif":
            self.records.sort(key=lambda x: (x[0] or datetime.max, x[2].lower()))
        elif sort_by == "ctime":
            self.records.sort(key=lambda x: (x[4], x[2].lower()))
        elif sort_by == "name":
            self.records.sort(key=lambda x: x[2].lower())
        else:
            raise ValueError("Unsupported sort_by: %r" % sort_by)

    def _build_mapping(self, start_index=1, progress_callback=None):
        """根据已读取并排序的 records 生成 mapping。"""
        self.mapping = []
        total = len(self.records)
        for i, (dt, idx, name, old_path, _ctime) in enumerate(
                self.records, start_index):
            date_str = dt.strftime("%Y%m%d") if dt else self.date or ""
            new_name = "%04d_%s%s%s%s%s%s.JPG" % (
                i, date_str, self.position or "",
                self.group_id or "", self.photographer or "",
                self.camera or "", idx or "")
            new_path = os.path.join(self.root_dir, new_name)
            # 万一冲突, 追加序号
            new_path = self._unique_path(new_path)
            self.mapping.append((old_path, new_path))
            if progress_callback:
                progress_callback(i - start_index + 1, total)
        return self

    def preview(self, start_index=1, sort_by="name", progress_callback=None):
        """返回 [(old_name, new_name), ...] 列表。"""
        need_exif = (sort_by == "exif" and not self._read_exif)
        if not self.records or need_exif:
            self.scan(start_index=start_index, sort_by=sort_by,
                      progress_callback=progress_callback)
        else:
            self._sort_records(sort_by)
            self._build_mapping(start_index=start_index,
                                progress_callback=progress_callback)
        return [(os.path.basename(o), os.path.basename(n))
                for o, n in self.mapping]

    def rename(self, start_index=1, sort_by="name", dry_run=False,
               progress_callback=None):
        """执行重命名。dry_run=True 时不真正移动文件, 只返回映射。"""
        if not self.mapping:
            self.scan(start_index=start_index, sort_by=sort_by,
                      progress_callback=progress_callback)
        if dry_run:
            return self.preview()
        total = len(self.mapping)
        moved = []
        for i, (old_path, new_path) in enumerate(self.mapping, 1):
            os.rename(old_path, new_path)
            moved.append((old_path, new_path))
            if progress_callback:
                progress_callback(i, total)
        # 映射中的 old_path 已失效, 重新扫描以便再次 rename
        self.mapping = []
        self.records = []
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
        """从原文件名中提取数字部分, 如 ZRQ04753 -> 4753。"""
        m = re.search(r"\d+", name)
        if not m:
            return ""
        return f'{int(m.group()):04d}'

    @staticmethod
    def _file_creation_time(path):
        """返回文件创建/修改时间; 优先使用 st_birthtime, 否则 st_ctime。"""
        try:
            st = os.stat(path)
            return getattr(st, "st_birthtime", st.st_ctime)
        except Exception:
            return 0

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


def collect_initial_metadata(root_dir):
    """根据路径返回初始元数据字典 (不读取 EXIF)。"""
    renamer = PhotoRenamer(root_dir)
    renamer._parse_path()
    return {
        "date": renamer.date or "",
        "position": renamer.position or "",
        "group_id": renamer.group_id or "",
        "photographer": renamer.photographer or "",
        "camera": renamer.camera or "",
    }


# ----------------------------------------------------------------------
# Tkinter GUI
# ----------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAS_TK = True
except ImportError:
    _HAS_TK = False


class PhotoRenamerGUI:
    def __init__(self, master, initial_dir=None):
        self.master = master
        master.title("Photo Renamer")
        master.geometry("900x720")
        master.minsize(700, 450)

        self.dir_var = tk.StringVar()
        self.field_vars = {}
        self.start_var = tk.IntVar(value=1)
        self.sort_var = tk.StringVar(value="Name")
        self.sort_by_map = {
            "Name": "name",
            "EXIF Date": "exif",
            "File Create Time": "ctime",
        }
        self.field_names = [
            ("date", "Date (YYYYMMDD)"),
            ("position", "Position"),
            ("group_id", "Group ID"),
            ("photographer", "Photographer"),
            ("camera", "Camera"),
        ]

        self._renamer = None
        self._updating_fields = False
        self._preview_after = None

        self._build_ui()
        self._attach_traces()

        if initial_dir:
            self.dir_var.set(initial_dir)
            self._load_metadata()

    def _build_ui(self):
        main = ttk.Frame(self.master, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 目录选择
        dir_frame = ttk.LabelFrame(main, text="Working Directory", padding=10)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(dir_frame, textvariable=self.dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse ...", command=self._browse).pack(
            side=tk.RIGHT)

        # 元数据
        meta_frame = ttk.LabelFrame(main, text="Metadata", padding=10)
        meta_frame.pack(fill=tk.X, pady=(0, 10))
        for i, (key, label) in enumerate(self.field_names):
            ttk.Label(meta_frame, text=label + ":").grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=3)
            var = tk.StringVar()
            self.field_vars[key] = var
            ttk.Entry(meta_frame, textvariable=var, width=30).grid(
                row=i, column=1, sticky=tk.W + tk.E, padx=5, pady=3)

        # 起始序号
        ttk.Label(meta_frame, text="Start Index:").grid(
            row=len(self.field_names), column=0, sticky=tk.W, padx=5, pady=3)
        ttk.Entry(meta_frame, textvariable=self.start_var, width=10).grid(
            row=len(self.field_names), column=1, sticky=tk.W, padx=5, pady=3)

        # 排序方式
        ttk.Label(meta_frame, text="Sort by:").grid(
            row=len(self.field_names) + 1, column=0, sticky=tk.W, padx=5, pady=3)
        sort_combo = ttk.Combobox(
            meta_frame, textvariable=self.sort_var,
            values=list(self.sort_by_map.keys()), state="readonly", width=18)
        sort_combo.grid(
            row=len(self.field_names) + 1, column=1, sticky=tk.W, padx=5, pady=3)
        meta_frame.columnconfigure(1, weight=1)

        # 操作按钮
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btn_frame, text="Rename", command=self._rename).pack(
            side=tk.LEFT)

        # 预览列表
        tree_frame = ttk.Frame(main)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        cols = ("old_name", "new_name")
        self.tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("old_name", text="Current Name")
        self.tree.heading("new_name", text="New Name")
        self.tree.column("old_name", width=300)
        self.tree.column("new_name", width=400)
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                            command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL,
                            command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # 状态栏 + 进度条
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        self.status_var = tk.StringVar(value="Select a directory to start.")
        ttk.Label(status_frame, textvariable=self.status_var).pack(
            side=tk.LEFT, fill=tk.Y)
        self.progress = ttk.Progressbar(
            status_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT)

    def _attach_traces(self):
        for var in self.field_vars.values():
            var.trace_add("write", self._on_field_change)
        self.start_var.trace_add("write", self._on_field_change)
        self.sort_var.trace_add("write", self._on_field_change)

    def _on_field_change(self, *args):
        if self._updating_fields:
            return
        self._schedule_preview()

    def _schedule_preview(self):
        if self._preview_after is not None:
            self.master.after_cancel(self._preview_after)
        self._preview_after = self.master.after(300, self._preview)

    def _browse(self):
        path = filedialog.askdirectory(title="Select photo folder")
        if path:
            self.dir_var.set(path)
            self._load_metadata()

    def _load_metadata(self):
        path = self.dir_var.get().strip()
        if not path or not os.path.isdir(path):
            for var in self.field_vars.values():
                var.set("")
            self.start_var.set(1)
            self.sort_var.set("Name")
            self.status_var.set("Invalid directory.")
            self._renamer = None
            return

        self._updating_fields = True
        meta = collect_initial_metadata(path)
        for key, var in self.field_vars.items():
            var.set(meta.get(key, ""))
        self.start_var.set(1)
        self.sort_var.set("Name")
        self._updating_fields = False

        self._renamer = None
        self._schedule_preview()

    def _get_metadata(self):
        return {key: var.get().strip()
                for key, var in self.field_vars.items()}

    def _get_start_index(self):
        try:
            value = self.start_var.get()
        except tk.TclError:
            return None
        return value

    def _get_sort_by(self):
        return self.sort_by_map.get(self.sort_var.get(), "name")

    def _validate(self):
        path = self.dir_var.get().strip()
        if not path:
            messagebox.showwarning("No directory", "Please select a directory.")
            return None
        if not os.path.isdir(path):
            messagebox.showerror("Invalid directory",
                                 "Directory does not exist: %s" % path)
            return None

        meta = self._get_metadata()
        date = meta.get("date", "")
        if date and not re.fullmatch(r"\d{8}", date):
            messagebox.showerror(
                "Invalid date",
                "Date must be 8 digits (YYYYMMDD) or empty.")
            return None

        start_index = self._get_start_index()
        if start_index is None or start_index < 1:
            messagebox.showerror(
                "Invalid start index",
                "Start index must be a positive integer.")
            return None

        sort_by = self._get_sort_by()
        if sort_by not in self.sort_by_map.values():
            messagebox.showerror("Invalid sort", "Unknown sort method.")
            return None

        return path, meta, start_index, sort_by

    def _make_progress_callback(self, phase):
        def callback(current, total):
            self.progress["maximum"] = max(total, 1)
            self.progress["value"] = current
            self.status_var.set("%s ... %d / %d" % (phase, current, total))
            self.master.update_idletasks()
        return callback

    def _preview(self):
        self._preview_after = None
        validated = self._validate()
        if validated is None:
            return
        path, meta, start_index, sort_by = validated
        abs_path = os.path.abspath(path)

        try:
            self.progress["value"] = 0
            self.progress["maximum"] = 1
            self.status_var.set("Reading photos ...")
            self.master.update_idletasks()

            need_scan = (self._renamer is None or
                         self._renamer.root_dir != abs_path or
                         not self._renamer.records)
            if need_scan:
                self._renamer = PhotoRenamer(path, **meta)
            else:
                for key, value in meta.items():
                    setattr(self._renamer, key, value)

            phase = "Reading photos" if need_scan else "Updating preview"
            mapping = self._renamer.preview(
                start_index=start_index, sort_by=sort_by,
                progress_callback=self._make_progress_callback(phase))
            self._show_mapping(mapping)
            self.status_var.set("%d file(s) previewed." % len(mapping))
            self.progress["value"] = self.progress["maximum"]
        except Exception as exc:
            self.status_var.set("Preview failed.")
            self.progress["value"] = 0
            messagebox.showerror("Error", str(exc))

    def _rename(self):
        if self._preview_after is not None:
            self.master.after_cancel(self._preview_after)
            self._preview_after = None
        self._preview()
        if self._renamer is None or not self._renamer.mapping:
            return

        mapping = self._renamer.mapping
        if not mapping:
            messagebox.showinfo("No files", "No JPG files to rename.")
            return

        if not messagebox.askyesno(
                "Confirm", "Rename %d file(s)?" % len(mapping)):
            return

        try:
            renamed = self._renamer.rename(
                progress_callback=self._make_progress_callback("Renaming"))
        except Exception as exc:
            self.status_var.set("Rename failed.")
            self.progress["value"] = 0
            messagebox.showerror("Error", str(exc))
            return

        self._show_mapping(renamed)
        self.status_var.set("Renamed %d file(s)." % len(renamed))
        self.progress["value"] = self.progress["maximum"]
        messagebox.showinfo(
            "Done", "Successfully renamed %d file(s)." % len(renamed))

    def _show_mapping(self, mapping):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for old_name, new_name in mapping:
            self.tree.insert("", tk.END, values=(old_name, new_name))


def run_gui(initial_dir=None):
    root = tk.Tk()
    PhotoRenamerGUI(root, initial_dir=initial_dir)
    root.mainloop()


def run_cli():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = args[0] if args else "."
    start_index = 1
    if len(args) == 2:
        start_index = int(args[1]) 
    dry = "--dry-run" in sys.argv
    renamer = PhotoRenamer(root)
    mapping = renamer.scan().rename(start_index= start_index, dry_run=dry)
    for old_name, new_name in mapping:
        print("%s -> %s" % (old_name, new_name))


def _gui_initial_dir():
    for a in sys.argv[1:]:
        if a != "--gui" and not a.startswith("-"):
            return a
    return None


if __name__ == "__main__":
    if "--gui" in sys.argv or (len(sys.argv) == 1 and _HAS_TK):
        if not _HAS_TK:
            sys.exit("tkinter is required for the GUI.")
        run_gui(initial_dir=_gui_initial_dir())
    else:
        run_cli()
