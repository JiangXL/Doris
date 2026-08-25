#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DorisGUI — 中华白海豚照片浏览与分组 GUI

6 个标签页:
  1. 单张全图浏览 (Full Images)
  2. 按连拍分组 (Continuous Shots, IMAGE_METAINFO.csv 的 shot_id)
  3. 按鳍部位分组 (Fin Aspect: DL=Left / DR=Right / Others=Tail·Head)
  4. 按模糊与否分组 (Blur / Clear, 依据 FIN/BLUR 与 FIN_MEGA/BLUR 目录)
  5. 按个体编号分组 (Fin ID, FIN/FinIDxxx 目录)
  6. 按社会结构分组 (Social Structure, 同一连拍中个体共现的连通分量)

用法:
  python DorisGUI.py [root_folder]
默认 root 为演示目录。
"""

import os
import sys
import csv
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QDialog,
    QScrollArea, QFileDialog, QToolBar, QAction, QMessageBox,
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QImageIOHandler
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

DEFAULT_ROOT = "/media/dolphin/2026PHOTO/20260324-A1-ZR-JM-SC-ZLF/PHOTO/02[F]_GM"
IMG_EXTS = (".jpg", ".jpeg", ".png")
THUMB_SIZE = 256

ASPECT_LABELS = {"DL": "Left (DL)", "DR": "Right (DR)", "Others": "Tail/Head (Others)"}


# ---------------------------------------------------------------------------
# 缩略图后台加载
# ---------------------------------------------------------------------------
class ThumbLoader(QThread):
    loaded = pyqtSignal(str, QImage)

    def __init__(self, paths):
        super().__init__()
        self._paths = list(paths)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        rot90 = (QImageIOHandler.TransformationRotate90,
                 QImageIOHandler.TransformationRotate270,
                 QImageIOHandler.TransformationFlipAndRotate90,
                 QImageIOHandler.TransformationMirrorAndRotate90)
        for p in self._paths:
            if self._stop:
                break
            reader = QImageReader(p)
            reader.setAutoTransform(True)  # 按 EXIF 方向旋转
            # setScaledSize 不保持宽高比, 需按原始尺寸(含 EXIF 旋转)手动算目标尺寸
            sz = QSize(reader.size())
            if reader.transformation() in rot90:
                sz.transpose()
            sz.scale(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio)
            reader.setScaledSize(sz)
            img = reader.read()
            if not img.isNull():
                self.loaded.emit(p, img)


class ImageGrid(QListWidget):
    """缩略图网格;双击弹窗查看大图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.setGridSize(QSize(THUMB_SIZE + 16, THUMB_SIZE + 64))
        self.setSpacing(8)
        self.setWordWrap(True)  # 长文件名换行显示
        self.setTextElideMode(Qt.ElideNone)  # 省略会抑制换行, 关闭
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Static)
        self._loader = None
        self._items = {}
        self.itemDoubleClicked.connect(self._open_full)

    def show_images(self, paths):
        if self._loader is not None:
            try:
                self._loader.loaded.disconnect(self._set_thumb)
            except TypeError:
                pass
            self._loader.stop()
            self._loader.wait()
        self.clear()
        self._items = {}
        for p in paths:
            name = os.path.basename(p)
            # 插入零宽空格使长文件名可在任意位置换行(默认只在空格处换行)
            display = "\u200b".join(name[i:i + 12] for i in range(0, len(name), 12))
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            # 必须在创建时给定尺寸, 否则视图按纯文字布局后图标加载不重排, 图像被裁
            # 宽度与 gridSize 一致, 文字区域占满图像宽度
            item.setSizeHint(QSize(THUMB_SIZE + 16, THUMB_SIZE + 48))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.addItem(item)
            self._items[p] = item
        self._loader = ThumbLoader(paths)
        self._loader.loaded.connect(self._set_thumb)
        self._loader.start()

    def _set_thumb(self, path, img):
        item = self._items.get(path)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _open_full(self, item):
        path = item.data(Qt.UserRole)
        dlg = QDialog(self)
        dlg.setWindowTitle(os.path.basename(path))
        dlg.resize(1000, 700)
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea()
        lbl = QLabel()
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        pm = QPixmap.fromImage(reader.read())
        if not pm.isNull():
            pm = pm.scaled(dlg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(pm)
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll)
        dlg.exec_()


# ---------------------------------------------------------------------------
# 通用 "侧边分组 + 网格" 页
# ---------------------------------------------------------------------------
def _wrap_text(s, width=34):
    """手动按宽度断行(优先在逗号/空格后断开), 避免依赖视图换行设置。"""
    lines, cur = [], ""
    for ch in s:
        cur += ch
        if (len(cur) >= width and ch in ", ]") or len(cur) >= width + 12:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return "\n".join(lines)


class GroupedTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.splitter = QSplitter(Qt.Horizontal)
        self.group_list = QListWidget()
        self.group_list.setMinimumWidth(240)   # 保证初始宽度, 否则 splitter 按空列表 sizeHint 给得很窄
        self.group_list.setMaximumWidth(420)
        self.group_list.setWordWrap(False)
        self.group_list.setTextElideMode(Qt.ElideNone)  # 不省略
        self.grid = ImageGrid()
        self.splitter.addWidget(self.group_list)
        self.splitter.addWidget(self.grid)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([280, 1000])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.splitter)
        self._groups = {}
        self.group_list.currentTextChanged.connect(self._on_group)

    def set_groups(self, groups, tooltips=None):
        """groups: dict[str, list[str]] — 组名 -> 图片路径列表 (有序)"""
        self._groups = groups
        self.group_list.clear()
        for name in groups:
            display = "%s  (%d)" % (name, len(groups[name]))
            item = QListWidgetItem(display, self.group_list)
            item.setData(Qt.UserRole, name)
            item.setToolTip((tooltips or {}).get(name, name))
        if groups:
            self.group_list.setCurrentRow(0)

    def _on_group(self, text):
        item = self.group_list.currentItem()
        if item is None:
            return
        self.grid.show_images(self._groups.get(item.data(Qt.UserRole), []))


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def list_images(folder, recursive=False):
    out = []
    if recursive:
        for dirpath, _, files in os.walk(folder):
            for fn in sorted(files):
                if fn.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(dirpath, fn))
    elif os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(folder, fn)):
                out.append(os.path.join(folder, fn))
    return out


class Dataset:
    """从 root 目录加载全部元数据。"""

    def __init__(self, root):
        self.root = root
        self.meta_dir = self._find_meta_dir()
        self.image_rows = self._load_meta("IMAGE_METAINFO.csv")
        self.fin_rows = self._load_meta("FIN_METAINFO.csv")

    def _find_meta_dir(self):
        for name in ("METAINFO_MEGA", "METAINFO"):
            d = os.path.join(self.root, name)
            if os.path.isdir(d):
                return d
        return self.root

    def _load_meta(self, fname):
        p = os.path.join(self.meta_dir, fname)
        return read_csv(p) if os.path.isfile(p) else []

    # 1. 全图
    def full_images(self):
        return list_images(self.root)

    # 2. 连拍分组
    def shots(self):
        groups = defaultdict(list)
        for row in self.image_rows:
            p = os.path.join(self.root, row["orig_img_name"])
            if os.path.isfile(p):
                groups.setdefault("shot_%s" % row.get("shot_id", "?"), []).append(p)
        if not groups:  # 无元数据时退化为单组
            groups["all"] = self.full_images()
        return dict(sorted(groups.items(),
                           key=lambda kv: int(kv[0].split("_")[1])
                           if kv[0].split("_")[1].isdigit() else 0))

    def _fin_path(self, row):
        rel = row.get("path", "")
        for base in (self.root, os.path.join(self.root, "FIN_MEGA"),
                     os.path.join(self.root, "FIN")):
            p = os.path.join(base, os.path.basename(rel)) if base.endswith(("FIN", "FIN_MEGA")) \
                else os.path.join(base, rel)
            if os.path.isfile(p):
                return p
        p = os.path.join(self.root, rel)
        return p if os.path.isfile(p) else None

    # 3. 部位分组
    def aspects(self):
        groups = defaultdict(list)
        for row in self.fin_rows:
            p = self._fin_path(row)
            if p:
                label = ASPECT_LABELS.get(row.get("class", "?"), row.get("class", "?"))
                groups[label].append(p)
        return dict(groups)

    # 4. 模糊分组
    def blur(self):
        blur_names = set()
        for sub in (os.path.join("FIN", "BLUR"), os.path.join("FIN_MEGA", "BLUR")):
            for p in list_images(os.path.join(self.root, sub)):
                blur_names.add(os.path.basename(p))
        groups = {"Blur": [], "Clear": []}
        rows = self.fin_rows or [{"path": os.path.relpath(p, self.root)}
                                 for p in list_images(os.path.join(self.root, "FIN_MEGA"))]
        seen = set()
        for row in rows:
            p = self._fin_path(row) if "path" in row else None
            if p and p not in seen:
                seen.add(p)
                groups["Blur" if os.path.basename(p) in blur_names else "Clear"].append(p)
        return groups

    # 5. 个体分组
    def fin_ids(self):
        groups = {}
        fin_dir = os.path.join(self.root, "FIN")
        if os.path.isdir(fin_dir):
            for name in sorted(os.listdir(fin_dir)):
                d = os.path.join(fin_dir, name)
                if name.startswith("FinID") and os.path.isdir(d):
                    imgs = list_images(d)
                    if imgs:
                        groups[name] = imgs
        return groups

    # 6. 社会结构: 同一连拍共现个体的连通分量
    def social(self):
        # fin 文件名 -> FinID
        fin_to_id = {}
        for fid, imgs in self.fin_ids().items():
            for p in imgs:
                fin_to_id[os.path.basename(p)] = fid
        if not fin_to_id:
            return {}
        # shot_id -> FinID 集合
        shot_to_ids = defaultdict(set)
        for row in self.fin_rows:
            fid = fin_to_id.get(os.path.basename(row.get("path", "")))
            if fid:
                shot_to_ids[row.get("shot_id", "?")].add(fid)
        # 并查集
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ids in shot_to_ids.values():
            ids = list(ids)
            for other in ids[1:]:
                union(ids[0], other)
        comps = defaultdict(list)
        for fid in fin_to_id.values():
            comps[find(fid)].append(fid)
        id_imgs = self.fin_ids()
        groups = {}
        self.social_tooltips = {}
        for i, (_, members) in enumerate(
                sorted(comps.items(), key=lambda kv: -len(kv[1])), 1):
            members = sorted(set(members))
            name = "Group%d" % i
            self.social_tooltips[name] = "成员: " + ",".join(members)
            imgs = []
            for m in members:
                imgs.extend(id_imgs.get(m, []))
            groups[name] = sorted(imgs)
        return groups


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.setWindowTitle("Doris v1.2 — %s" % root)
        self.resize(1280, 800)

        toolbar = QToolBar()
        act = QAction("Open folder…", self)
        act.triggered.connect(self._pick_folder)
        toolbar.addAction(act)
        self.addToolBar(toolbar)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.load(root)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择 PHOTO 根目录")
        if d:
            self.load(d)

    def load(self, root):
        self.root = root
        ds = Dataset(root)
        self.tabs.clear()

        # 1. 单张全图
        tab1 = QWidget()
        lay = QVBoxLayout(tab1)
        lay.setContentsMargins(0, 0, 0, 0)
        grid = ImageGrid()
        grid.show_images(ds.full_images())
        lay.addWidget(grid)
        self.tabs.addTab(tab1, "Full Images")

        # 2-6. 分组页
        for title, fn in [
            ("Continuous Shots", ds.shots),
            ("Fin Aspect", ds.aspects),
            ("Blur", ds.blur),
            ("Fin ID", ds.fin_ids),
            ("Social Structure", ds.social),
        ]:
            tab = GroupedTab()
            groups = fn()
            if groups:
                tooltips = getattr(ds, "social_tooltips", None) if title == "Social Structure" else None
                tab.set_groups(groups, tooltips)
            else:
                tab.grid.show_images([])
                QListWidgetItem("(no data)", tab.group_list)
            self.tabs.addTab(tab, title)


def main():
    app = QApplication(sys.argv)
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    if not os.path.isdir(root):
        QMessageBox.critical(None, "Doris", "目录不存在: %s" % root)
        sys.exit(1)
    win = MainWindow(root)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
