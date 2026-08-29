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
from collections import defaultdict

import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QDialog,
    QScrollArea, QFileDialog, QToolBar, QAction, QMessageBox,
    QGraphicsView, QGraphicsScene,
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QImageIOHandler, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

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
        dlg = ImageViewerDialog(path, self)
        dlg.exec_()


class ZoomableView(QGraphicsView):
    """滚轮缩放(以光标为中心), 左键拖拽平移, 双击适应窗口。"""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)               # 拖拽平移
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 以光标为中心缩放
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)


class ImageViewerDialog(QDialog):
    """大图查看窗口。"""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(path))
        self.resize(1200, 800)

        reader = QImageReader(path)
        reader.setAutoTransform(True)  # 按 EXIF 方向旋转
        pm = QPixmap.fromImage(reader.read())

        scene = QGraphicsScene(self)
        scene.addPixmap(pm)
        self.view = ZoomableView(scene)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

        # 打开时适应窗口
        if not pm.isNull():
            self.view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)


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
# 数据加载 (pandas)
# ---------------------------------------------------------------------------
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
        self.image_df = self._load_meta("IMAGE_METAINFO.csv")
        self.fin_df = self._load_meta("FIN_METAINFO.csv")
        if not self.fin_df.empty:
            self.fin_df["fullpath"] = self.fin_df["path"].map(self._fin_path)

    def _find_meta_dir(self):
        for name in ("METAINFO_MEGA", "METAINFO"):
            d = os.path.join(self.root, name)
            if os.path.isdir(d):
                return d
        return self.root

    def _load_meta(self, fname):
        p = os.path.join(self.meta_dir, fname)
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    # 1. 全图
    def full_images(self):
        return list_images(self.root)

    # 2. 连拍分组
    def shots(self):
        if self.image_df.empty:  # 无元数据时退化为单组
            return {"all": self.full_images()}
        df = self.image_df[self.image_df["orig_img_name"].map(
            lambda n: os.path.isfile(os.path.join(self.root, n)))]
        groups = {}
        for sid, g in df.groupby("shot_id", sort=True):
            groups["shot_%s" % sid] = [os.path.join(self.root, n)
                                       for n in g["orig_img_name"]]
        return groups

    def _fin_path(self, rel):
        rel = str(rel)
        for base in (self.root, os.path.join(self.root, "FIN_MEGA"),
                     os.path.join(self.root, "FIN")):
            p = os.path.join(base, os.path.basename(rel)) if base.endswith(("FIN", "FIN_MEGA")) \
                else os.path.join(base, rel)
            if os.path.isfile(p):
                return p
        return None

    # 3. 部位分组
    def aspects(self):
        df = self.fin_df.dropna(subset=["fullpath"])
        groups = {}
        for cls, g in df.groupby("class"):
            label = ASPECT_LABELS.get(cls, str(cls))
            groups[label] = list(g["fullpath"])
        return groups

    # 4. 模糊分组
    def blur(self):
        blur_names = {os.path.basename(p)
                      for p in list_images(os.path.join(self.root, "FIN", "BLUR"))}
        groups = {"Blur": [], "Clear": []}
        if self.fin_df.empty:
            paths = list_images(os.path.join(self.root, "FIN_MEGA"))
        else:
            paths = list(self.fin_df["fullpath"].dropna().unique())
        for p in paths:
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
        # shot_id -> FinID 集合 (pandas)
        df = self.fin_df.copy()
        df["fin_name"] = df["path"].map(lambda p: os.path.basename(str(p)))
        df["fin_id"] = df["fin_name"].map(fin_to_id)
        shot_to_ids = df.dropna(subset=["fin_id"]).groupby("shot_id")["fin_id"] \
            .agg(set).to_dict()
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        root = sys.argv[1] 
        if not os.path.isdir(root):
            QMessageBox.critical(None, "Doris", "目录不存在: %s" % root)
            sys.exit(1)
        win = MainWindow(root)
        win.show()
        sys.exit(app.exec_())
