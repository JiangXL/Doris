#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DorisGUI — 中华白海豚照片浏览与分组 GUI

6 个标签页:
  1. 单张全图浏览 (Full Images)
  2. 按连拍分组 (Continuous Shots, IMAGE_METAINFO.csv 的 shot_id)
  3. 按鳍部位分组 (Fin Aspect: DL=Left / DR=Right / Others=Tail, Head, lateral Fin / Wrong=0 )
  4. 按模糊与否分组 (Blur / Clear, 依据 FIN/BLUR 与 FIN_MEGA/BLUR 目录)
  5. 按个体编号分组 (Fin ID, FIN/FinIDxxx 目录)
  6. 按社会结构分组 (Social Structure, 同一连拍中个体共现的连通分量)

用法:
  python DorisGUI.py [root_folder]
默认 root 为演示目录。
"""

import os
import re
import sys
import shutil
from datetime import datetime
from collections import defaultdict

import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QDialog,
    QScrollArea, QFileDialog, QToolBar, QAction, QMessageBox,
    QGraphicsView, QGraphicsScene,
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QImageIOHandler, QPainter, QBrush, QColor, QDrag, QPen
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPoint, QRect

IMG_EXTS = (".jpg", ".jpeg", ".png")
THUMB_SIZE = 256
MIME_PATHS = "application/x-doris-paths"

ASPECT_LABELS = {"DL": "Left (DL)", "DR": "Right (DR)", "Others": "Tail/Head (Others)", "ND":"Not Dolphin"}

BOX_COLORS = {"DL": QColor(0, 200, 0), "DR": QColor(255, 140, 0),
              "Others": QColor(220, 0, 0), "ND": QColor(128, 128, 128)}

_ROT90 = (QImageIOHandler.TransformationRotate90,
          QImageIOHandler.TransformationRotate270,
          QImageIOHandler.TransformationFlipAndRotate90,
          QImageIOHandler.TransformationMirrorAndRotate90)


def _map_box(box, trans, w, h):
    """stored 像素坐标的框 -> EXIF 旋转后显示坐标的框 (w,h 为 stored 尺寸)。
    镜像类变换未处理(相机照片一般没有)。"""
    x0, y0, x1, y1 = box
    T = QImageIOHandler
    if trans == T.TransformationRotate90:
        return h - y1, x0, h - y0, x1
    if trans == T.TransformationRotate270:
        return y0, w - x1, y1, w - x0
    if trans == T.TransformationRotate180:
        return w - x1, h - y1, w - x0, h - y0
    return x0, y0, x1, y1


def draw_boxes(target, boxes, trans, stored_w, stored_h):
    """在已按 EXIF 方向显示的 QImage/QPixmap 上画鳍框。
    boxes: [(x0, y0, x1, y1, cls), ...] stored 像素坐标。"""
    if trans in _ROT90:
        disp_w, disp_h = stored_h, stored_w
    else:
        disp_w, disp_h = stored_w, stored_h
    sx = target.width() / disp_w
    sy = target.height() / disp_h
    pen_w = max(2, target.width() // 300)
    p = QPainter(target)
    f = p.font()
    f.setPixelSize(max(12, target.width() // 40))
    f.setBold(True)
    p.setFont(f)
    for box in boxes:
        x0, y0, x1, y1, cls = box
        x0, y0, x1, y1 = _map_box((x0, y0, x1, y1), trans, stored_w, stored_h)
        rect = QRect(int(x0 * sx), int(y0 * sy),
                     int((x1 - x0) * sx), int((y1 - y0) * sy))
        color = BOX_COLORS.get(cls, Qt.yellow)
        p.setPen(QPen(color, pen_w))
        p.drawRect(rect)
        p.drawText(rect.left() + 2, rect.top() - 4, cls)
    p.end()


# ---------------------------------------------------------------------------
# 缩略图后台加载
# ---------------------------------------------------------------------------
class ThumbLoader(QThread):
    loaded = pyqtSignal(str, QImage)

    def __init__(self, paths, boxes=None):
        super().__init__()
        self._paths = list(paths)
        self._boxes = boxes or {}
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
            trans = reader.transformation()
            stored_w, stored_h = reader.size().width(), reader.size().height()
            # setScaledSize 不保持宽高比, 需按原始尺寸(含 EXIF 旋转)手动算目标尺寸
            sz = QSize(reader.size())
            if trans in rot90:
                sz.transpose()
            sz.scale(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio)
            reader.setScaledSize(sz)
            img = reader.read()
            if not img.isNull():
                boxes = self._boxes.get(p)
                if boxes:
                    draw_boxes(img, boxes, trans, stored_w, stored_h)
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
        self.setSelectionMode(QListWidget.ExtendedSelection)  # 多选
        self.setDragEnabled(True)                             # 可拖出
        self.setDragDropMode(QListWidget.DragOnly)            # 不允许放回自身
        self._loader = None
        self._items = {}
        self._boxes = {}
        self.itemDoubleClicked.connect(self._open_full)

    def mimeData(self, items):
        """拖拽时携带选中项的文件路径。"""
        md = QMimeData()
        md.setData(MIME_PATHS, "\n".join(
            it.data(Qt.UserRole) for it in items).encode("utf-8"))
        return md

    def startDrag(self, actions):
        """自定义拖拽图像: 第一张缩略图 + 数量角标, MoveAction 光标。"""
        items = self.selectedItems()
        if not items:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData(items))
        pm = self._drag_pixmap(items)
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))  # 光标位于图像中心
        drag.exec_(Qt.MoveAction)

    @staticmethod
    def _drag_pixmap(items):
        base = items[0].icon().pixmap(96, 96)
        if base.isNull():
            # 缩略图尚未加载时直接从文件读一张小的
            reader = QImageReader(items[0].data(Qt.UserRole))
            reader.setAutoTransform(True)
            sz = QSize(reader.size())
            sz.scale(96, 96, Qt.KeepAspectRatio)  # setScaledSize 不保宽高比
            reader.setScaledSize(sz)
            img = reader.read()
            base = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        if base.isNull():
            base = QPixmap(96, 96)
            base.fill(Qt.lightGray)
        n = len(items)
        if n == 1:
            return base
        # 右上角画数量角标
        pm = base.copy()
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        badge = QRect(pm.width() - 34, 2, 32, 32)
        p.setBrush(QColor(30, 144, 255, 220))
        p.setPen(Qt.NoPen)
        p.drawEllipse(badge)
        p.setPen(Qt.white)
        f = p.font()
        f.setBold(True)
        f.setPixelSize(18)
        p.setFont(f)
        p.drawText(badge, Qt.AlignCenter, str(n))
        p.end()
        return pm

    def show_images(self, paths, boxes=None):
        if self._loader is not None:
            try:
                self._loader.loaded.disconnect(self._set_thumb)
            except TypeError:
                pass
            self._loader.stop()
            self._loader.wait()
        self._boxes = boxes or {}
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
        self._loader = ThumbLoader(paths, self._boxes)
        self._loader.loaded.connect(self._set_thumb)
        self._loader.start()

    def _set_thumb(self, path, img):
        item = self._items.get(path)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _open_full(self, item):
        paths = [self.item(i).data(Qt.UserRole) for i in range(self.count())]
        dlg = ImageViewerDialog(paths, self.row(item), self, boxes=self._boxes)
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

    def keyPressEvent(self, event):
        # ←/→ 交给对话框做翻页, 不做滚动
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)


class ImageViewerDialog(QDialog):
    """大图查看窗口。←/→ 切换组内上一张/下一张。"""

    def __init__(self, paths, index, parent=None, boxes=None):
        super().__init__(parent)
        self._paths = paths
        self._index = index
        self._boxes = boxes or {}
        self.resize(1200, 800)

        self._scene = QGraphicsScene(self)
        self._item = None
        self.view = ZoomableView(self._scene)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

        self._show(index)

    def _show(self, index):
        index = max(0, min(index, len(self._paths) - 1))
        self._index = index
        path = self._paths[index]
        reader = QImageReader(path)
        reader.setAutoTransform(True)  # 按 EXIF 方向旋转
        trans = reader.transformation()
        stored_w, stored_h = reader.size().width(), reader.size().height()
        pm = QPixmap.fromImage(reader.read())
        boxes = self._boxes.get(path)
        if boxes and not pm.isNull():
            draw_boxes(pm, boxes, trans, stored_w, stored_h)
        if self._item is None:
            self._item = self._scene.addPixmap(pm)
        else:
            self._item.setPixmap(pm)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self.setWindowTitle("%s  (%d/%d)" % (os.path.basename(path),
                                             index + 1, len(self._paths)))
        if not pm.isNull():
            self.view.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._show(self._index - 1)   # 上一张
        elif event.key() == Qt.Key_Right:
            self._show(self._index + 1)   # 下一张
        else:
            super().keyPressEvent(event)


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


class GroupListWidget(QListWidget):
    """侧边分组列表, 接受从网格拖来的图片路径。"""

    paths_dropped = pyqtSignal(list, str)  # ([path, ...], 目标组名)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)
        self._hover_item = None  # 拖拽悬停高亮的分组项

    def _set_hover(self, item):
        if self._hover_item is item:
            return
        if self._hover_item is not None:
            self._hover_item.setBackground(QBrush())  # 恢复默认
        self._hover_item = item
        if item is not None:
            item.setBackground(QColor(30, 144, 255, 80))  # 高亮目标分组

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_PATHS):
            event.setDropAction(Qt.MoveAction)  # 移动光标
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME_PATHS):
            self._set_hover(self.itemAt(event.pos()))  # 经过时高亮
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_hover(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        item = self.itemAt(event.pos())
        self._set_hover(None)
        if item is None or not event.mimeData().hasFormat(MIME_PATHS):
            event.ignore()
            return
        paths = bytes(event.mimeData().data(MIME_PATHS)).decode("utf-8").splitlines()
        paths = [p for p in paths if p]
        if paths:
            self.paths_dropped.emit(paths, item.data(Qt.UserRole))
            event.setDropAction(Qt.MoveAction)
            event.accept()


class GroupedTab(QWidget):
    def __init__(self, mover=None, parent=None):
        """mover: callable(paths, target_group) -> dict[old_path, new_path] 或 None"""
        super().__init__(parent)
        self._mover = mover
        self.splitter = QSplitter(Qt.Horizontal)
        self.group_list = GroupListWidget()
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
        self._tooltips = {}
        self._boxes = {}
        self.group_list.currentTextChanged.connect(self._on_group)
        self.group_list.paths_dropped.connect(self._on_drop)
        # 无 mover 的页不接受拖放
        self.group_list.setAcceptDrops(mover is not None)
        self.grid.setDragEnabled(mover is not None)

    def set_groups(self, groups, tooltips=None, keep_current=False, boxes=None):
        """groups: dict[str, list[str]] — 组名 -> 图片路径列表 (有序)
        boxes: dict[str, list] — 图片路径 -> 鳍框列表 (可选, 画在缩略图和大图上)"""
        cur = self.group_list.currentItem()
        cur_name = cur.data(Qt.UserRole) if (keep_current and cur) else None
        self._groups = groups
        self._tooltips = tooltips or {}
        self._boxes = boxes or {}
        self.group_list.clear()
        for name in groups:
            display = "%s  (%d)" % (name, len(groups[name]))
            item = QListWidgetItem(display, self.group_list)
            item.setData(Qt.UserRole, name)
            item.setToolTip(self._tooltips.get(name, name))
            # 显式给足宽度, 避免某些主题按短 sizeHint 换行/省略
            item.setSizeHint(QSize(400, 28))
        if not groups:
            return
        if cur_name in groups:
            self.group_list.setCurrentRow(list(groups).index(cur_name))
        else:
            self.group_list.setCurrentRow(0)

    def _on_group(self, text):
        item = self.group_list.currentItem()
        if item is None:
            return
        self.grid.show_images(self._groups.get(item.data(Qt.UserRole), []),
                              boxes=self._boxes)

    def _on_drop(self, paths, target_name):
        if self._mover is None or target_name not in self._groups:
            return
        try:
            moved = self._mover(paths, target_name)
        except Exception as e:
            QMessageBox.warning(self, "移动失败", str(e))
            return
        if not moved:
            return
        for old, new in moved.items():
            for g in self._groups.values():
                if old in g:
                    g.remove(old)
            self._groups[target_name].append(new)
        self.set_groups(self._groups, self._tooltips, keep_current=True,
                        boxes=self._boxes)


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
        self.meta_dir = os.path.join(self.root, "METAINFO")
        self.image_df = self._load_meta("IMAGE_METAINFO.csv")
        self.fin_df = self._load_meta("FIN_METAINFO.csv")
        if not self.fin_df.empty:
            self.fin_df["fullpath"] = self.meta_dir + "/" + self.fin_df["path"]

    def _load_meta(self, fname):
        p = os.path.join(self.meta_dir, fname)
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    # 1. 全图
    def full_images(self):
        return self.root + "/" + self.image_df["orig_img_name"]

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

    def fin_boxes(self):
        """全图绝对路径 -> [(x0, y0, x1, y1, cls), ...] (stored 像素坐标)"""
        boxes = {}
        if self.fin_df.empty or "orig_img_name" not in self.fin_df.columns:
            return boxes
        need = ["x_min", "y_min", "x_max", "y_max"]
        if not all(c in self.fin_df.columns for c in need):
            return boxes
        for name, g in self.fin_df.groupby("orig_img_name"):
            p = os.path.join(self.root, str(name))
            if not os.path.isfile(p):
                continue
            boxes[p] = [(int(a), int(b), int(c), int(d), str(cls))
                        for a, b, c, d, cls in zip(g["x_min"], g["y_min"],
                                                   g["x_max"], g["y_max"],
                                                   g["class"])]
        return boxes

    # 3. 部位分组
    def aspects(self):
        groups = {label: [] for label in ASPECT_LABELS.values()}  # 空组也列出, 作为拖放目标
        df = self.fin_df.dropna(subset=["fullpath"])
        for cls, g in df.groupby("class"):
            label = ASPECT_LABELS.get(cls, str(cls))
            groups.setdefault(label, []).extend(g["fullpath"])
        return groups

    # 4. 质量分组: blur, clear, crop confidence?
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
                    groups[name] = list_images(d)  # 空目录也列出, 作为拖放目标
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
            # 组内成员的鳍图 -> 对应原始全图 (去重)
            fin_names = {os.path.basename(p) for m in members
                         for p in id_imgs.get(m, [])}
            orig = df[df["fin_name"].isin(fin_names)]["orig_img_name"] \
                .dropna().unique()
            imgs = [os.path.join(self.root, n) for n in sorted(orig)
                    if os.path.isfile(os.path.join(self.root, n))]
            groups[name] = imgs
        return groups

    # ------------------------------------------------------------------
    # 拖拽改组 (写盘)
    # ------------------------------------------------------------------
    def _save_csv(self, df, fname):
        path = os.path.join(self.meta_dir, fname)
        if os.path.isfile(path):
            shutil.copy2(path, path + "." +
                         datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak")
        df.drop(columns=["fullpath", "fin_name", "fin_id"], errors="ignore") \
            .to_csv(path, index=False)

    def move_to_fin_id(self, paths, group):
        """把鳍裁剪图移入 FIN/<FinIDxxx>/ 目录。"""
        dest = os.path.join(self.root, "FIN", group)
        os.makedirs(dest, exist_ok=True)
        moved = {}
        for p in paths:
            np_ = os.path.join(dest, os.path.basename(p))
            if os.path.abspath(np_) != os.path.abspath(p):
                shutil.move(p, np_)
            moved[p] = np_
        return moved

    def move_blur(self, paths, group):
        """Blur -> 移入 FIN/BLUR/; Clear -> 移回 FIN/ 平铺目录。"""
        dest = os.path.join(self.root, "FIN", "BLUR") if group == "Blur" \
            else os.path.join(self.root, "FIN")
        os.makedirs(dest, exist_ok=True)
        moved = {}
        for p in paths:
            np_ = os.path.join(dest, os.path.basename(p))
            if os.path.abspath(np_) != os.path.abspath(p):
                shutil.move(p, np_)
            moved[p] = np_
        return moved

    def move_aspect(self, paths, group):
        """改部位类别: 重命名文件名后缀并更新 FIN_METAINFO.csv。"""
        cls = {v: k for k, v in ASPECT_LABELS.items()}.get(group, group)
        moved = {}
        dirty = False
        for p in paths:
            base = os.path.basename(p)
            new_base, n = re.subn(r"_(DL|DR|Others)(\.[^.]+)$",
                                  "_%s\\g<2>" % cls, base, flags=re.IGNORECASE)
            if not n:
                continue
            # 重命名 root 下所有同名副本 (FIN / FIN_MEGA / FinID / BLUR 等)
            new_p = None
            for dirpath, _, files in os.walk(self.root):
                if base in files:
                    tgt = os.path.join(dirpath, new_base)
                    os.rename(os.path.join(dirpath, base), tgt)
                    if os.path.abspath(dirpath) == os.path.abspath(os.path.dirname(p)):
                        new_p = tgt
            if new_p is None:
                continue
            mask = self.fin_df["path"].map(lambda x: os.path.basename(str(x))) == base
            if mask.any():
                self.fin_df.loc[mask, "path"] = \
                    self.fin_df.loc[mask, "path"].map(
                        lambda x: os.path.join(os.path.dirname(str(x)), new_base))
                self.fin_df.loc[mask, "class"] = cls
                self.fin_df.loc[mask, "fullpath"] = new_p
                dirty = True
            moved[p] = new_p
        if dirty:
            self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return moved

    def move_shot(self, paths, group):
        """改连拍分组: 更新 IMAGE_METAINFO.csv 和 FIN_METAINFO.csv 的 shot_id。"""
        m = re.match(r"shot_(\d+)$", group)
        if not m:
            return {}
        sid = int(m.group(1))
        names = {os.path.basename(p) for p in paths}
        moved = {p: p for p in paths}
        if not self.image_df.empty:
            mask = self.image_df["orig_img_name"].isin(names)
            if mask.any():
                self.image_df.loc[mask, "shot_id"] = sid
                self._save_csv(self.image_df, "IMAGE_METAINFO.csv")
        if not self.fin_df.empty and "orig_img_name" in self.fin_df.columns:
            mask = self.fin_df["orig_img_name"].isin(names)
            if mask.any():
                self.fin_df.loc[mask, "shot_id"] = sid
                self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return moved


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
        grid.setDragEnabled(False)  # Full Images 页不支持拖出
        grid.show_images(ds.full_images())
        lay.addWidget(grid)
        self.tabs.addTab(tab1, "Full Images")

        # 2-6. 分组页 (前四个支持拖拽改组并写盘)
        shot_boxes = ds.fin_boxes()  # Continuous Shots 页画鳍框
        for title, fn, mover in [
            ("Continuous Shots", ds.shots, ds.move_shot),
            ("Fin Aspect", ds.aspects, ds.move_aspect),
            ("Blur", ds.blur, ds.move_blur),
            ("Fin ID", ds.fin_ids, ds.move_to_fin_id),
            ("Social Structure", ds.social, None),
        ]:
            tab = GroupedTab(mover=mover)
            groups = fn()
            if title == "Fin ID":  # 只显示图片数大于 1 的个体
                groups = {k: v for k, v in groups.items() if len(v) > 1}
            if groups:
                tooltips = getattr(ds, "social_tooltips", None) if title == "Social Structure" else None
                boxes = shot_boxes if title == "Continuous Shots" else None
                tab.set_groups(groups, tooltips, boxes=boxes)
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
