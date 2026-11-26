# -*- coding: utf-8 -*-
"""海豚页相关控件：DolphinTab、FinSlot、CandidateList、MatchPanel。"""

import os
import re

import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QListWidget, QListWidgetItem, QLabel, QVBoxLayout,
    QHBoxLayout, QSplitter, QScrollArea, QPushButton, QFrame,
    QMessageBox, QDialog,
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QImageReader, QPainter, QColor, QPen, QDrag, QFont,
    QKeySequence, QShortcut,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QMimeData, QPoint, QRect, QRectF

from .common import (
    ImageGrid, DolphinLoader, ThumbOverlayDelegate, ZoomableView,
    open_image_viewer, open_fin_viewer, open_orig_viewer,
    MIME_PATHS, THUMB_SIZE, FIN_CELL_H,
    BOX_COLORS, ThumbLoader
)


def _show_fin_viewer(parent, paths, current, labels):
    """鳍裁剪图查看器: 组内翻页 + 标签叠加, 从 current 所在位置开始。"""
    if not paths:
        return
    idx = paths.index(current) if current in paths else 0
    open_image_viewer(open_fin_viewer(paths, idx, parent, labels=labels))


def _show_orig_viewer(parent, fins, current, orig_resolver, boxes, labels):
    """原图查看器: fins 映射到原图 (去重保序), 从 current 对应原图开始,
    叠加鳍框/标签。"""
    if orig_resolver is None or not fins:
        return
    origs = []
    for p in fins:
        o = orig_resolver(p)
        if o and o not in origs:
            origs.append(o)
    if not origs:
        return
    cur = orig_resolver(current) if current else None
    idx = origs.index(cur) if cur in origs else 0
    open_image_viewer(open_orig_viewer(origs, idx, parent, boxes=boxes,
                                       labels=labels))


def _side_browse_set(fin_lister, name, cls, current):
    """某海豚某侧的全部鳍图作为大图浏览集; 无提供者/无记录退回当前一张。"""
    paths = fin_lister(name, cls) if fin_lister is not None else []
    if paths:
        return paths
    return [current] if current else []

class DolphinTab(QListWidget):
    """海豚配对页: 每只海豚占一格, 上=DL 鳍图, 下=DR 鳍图, 标题 Dphxxx。
    初始所有 DL FinID 放在上格; DR 按与 DL 的最高相似度自动配对。
    下格 (DR) 可拖到其它海豚格: 双方都有 DR 时互换, 目标无 DR 时移动。
    双击鳍图 (上半格 DL / 下半格 DR) 打开大图查看窗口。"""

    dr_swapped = pyqtSignal(str, str)  # (源 Dph 名, 目标 Dph 名)
    dr_dropped_on_empty = pyqtSignal(str)  # 源 Dph 名
    count_changed = pyqtSignal(int)      # 海豚总数 (顶栏状态标签用)
    DR_MIME = "application/x-doris-dolphin-dr"
    DOLPHIN_MIME = "application/x-doris-dolphin-cell"

    def __init__(self, parent=None, fin_lister=None, fin_labels=None,
                 orig_resolver=None, photo_boxes=None, photo_labels=None):
        super().__init__(parent)
        cell = FIN_CELL_H
        self._fin_lister = fin_lister     # (DolphinID, DL/DR) -> [fin 路径]
        self._fin_labels = fin_labels or {}   # fin 路径 -> 标注
        self._orig_resolver = orig_resolver   # fin 路径 -> 原图路径
        self._photo_boxes = photo_boxes or {} # 原图路径 -> 鳍框
        self._photo_labels = photo_labels or {}  # 原图路径 -> 标注
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setIconSize(QSize(THUMB_SIZE, cell * 2))  # 上下两格
        self.setGridSize(QSize(THUMB_SIZE + 16, cell * 2 + 16))
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Movement.Static)
        # 滚轮按像素平滑滚动
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.horizontalScrollBar().setSingleStep(24)
        self.setItemDelegate(ThumbOverlayDelegate(self))
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.setDropIndicatorShown(True)
        # Movement.Static 会让 Qt 关掉 viewport 的 acceptDrops (静态视图不允许
        # 内部移动), 导致拖放根本投递不进来; 当前的 dropEvent 自己处理落点,
        # 不依赖视图移动, 显式重新打开
        self.viewport().setAcceptDrops(True)
        self._loaders = []
        self._items = {}
        self._press_top = False     # 按下位置是否在 DL (上) 格
        self._press_bottom = False  # 按下位置是否在 DR (下) 格
        self._dolphin_order = []
        self._dolphin_data = {}
        self._social = {}

    def show_dolphins(self, dolphins, social=None):
        """dolphins: [(name, dl_fid, dr_fid, dl_path, dr_path)];
        social: {DolphinID: SocialStructure 标签}; 显示在每个海豚格左上角。
        后台分片加载合成缩略图。"""
        self._dolphin_order = [d[0] for d in dolphins]
        self._dolphin_data = {d[0]: d for d in dolphins}
        self._social = social or {}
        self.count_changed.emit(len(dolphins))
        self._stop_loaders()
        self.clear()
        self._items = {}
        cell = FIN_CELL_H
        social = self._social
        for name, dl_fid, dr_fid, dl, dr in dolphins:
            item = QListWidgetItem()  # 名称由合成图画在上下鳍之间, 不再用顶部条
            item.setData(Qt.ItemDataRole.UserRole, (name, dl_fid, dr_fid, dl, dr))
            # 创建时给定尺寸, 否则图标加载后视图不重排
            item.setSizeHint(QSize(THUMB_SIZE + 16, cell * 2 + 16))
            self.addItem(item)
            self._items[name] = item
        if not dolphins:
            return
        allp = [p for d in dolphins for p in (d[3], d[4]) if p]
        try:
            cache_dir = os.path.join(os.path.commonpath(allp),
                                     ".doris_thumbs") if allp else None
        except ValueError:
            cache_dir = None
        # (name, dl_fid, dr_fid, dl_path, dr_path, social_label)
        jobs = [(d[0], d[1], d[2], d[3], d[4], social.get(d[0], ""))
                for d in dolphins]
        n = min(8, os.cpu_count() or 4, len(jobs))
        for k in range(n):
            loader = DolphinLoader(jobs[k::n], cache_dir=cache_dir)
            loader.loaded.connect(self._set_thumb)
            loader.start()
            self._loaders.append(loader)

    def current_order(self):
        """Return the current display order of DolphinIDs."""
        return list(self._dolphin_order)

    def _stop_loaders(self):
        for l in self._loaders:
            l.stop()
        for l in self._loaders:
            l.wait()
        self._loaders = []

    def _set_thumb(self, name, img):
        item = self._items.get(name)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    # -- DR 格拖拽交换 -------------------------------------------------------
    def _cell_side(self, event):
        """落点所在格的海豚名 / 侧 (DL=上半, DR=下半) / 该侧鳍图路径;
        空白处返回 (None, None, None)。"""
        item = self.itemAt(event.position().toPoint())
        if item is None:
            return None, None, None
        r = self.visualItemRect(item)
        bottom = event.pos().y() > r.top() + r.height() // 2
        name, dl_fid, dr_fid, dl, dr = item.data(Qt.ItemDataRole.UserRole)
        return name, ("DR" if bottom else "DL"), (dr if bottom else dl)

    def _open_fin_side(self, name, cls, path):
        """双击鳍格: 鳍裁剪图查看器 (该海豚该侧全部鳍 + 标签)。"""
        paths = _side_browse_set(self._fin_lister, name, cls, path)
        _show_fin_viewer(self, paths, path, self._fin_labels)

    def _open_orig_side(self, name, cls, path):
        """中键鳍格: 原图查看器 (鳍框/标签叠加)。"""
        fins = _side_browse_set(self._fin_lister, name, cls, path)
        _show_orig_viewer(self, fins, path, self._orig_resolver,
                          self._photo_boxes, self._photo_labels)

    def mousePressEvent(self, event):
        self._press_top = False
        self._press_bottom = False
        if event.button() == Qt.MouseButton.MiddleButton:
            name, cls, path = self._cell_side(event)
            if path:
                self._open_orig_side(name, cls, path)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                r = self.visualItemRect(item)
                self._press_bottom = event.pos().y() > r.top() + r.height() // 2
                self._press_top = not self._press_bottom
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """双击鳍图打开鳍裁剪图查看器: 上半格看 DL, 下半格看 DR。"""
        name, cls, path = self._cell_side(event)
        if path:
            self._open_fin_side(name, cls, path)
        super().mouseDoubleClickEvent(event)

    def startDrag(self, actions):
        """上半格 (DL) 拖拽用于重排海豚格, 下半格 (DR) 拖拽用于交换/移动 DR 鳍。"""
        item = self.currentItem()
        if item is None:
            return
        name, dl_fid, dr_fid, dl, dr = item.data(Qt.ItemDataRole.UserRole)
        if self._press_top:
            # 移动整个海豚格
            md = QMimeData()
            md.setData(self.DOLPHIN_MIME, name.encode("utf-8"))
            drag = QDrag(self)
            drag.setMimeData(md)
            pm = item.icon().pixmap(THUMB_SIZE, FIN_CELL_H * 2)
            if pm.isNull():
                pm = QPixmap(96, 128)
                pm.fill(Qt.GlobalColor.lightGray)
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
            drag.exec(Qt.DropAction.MoveAction)
            return
        if not self._press_bottom or dr_fid is None:
            return
        md = QMimeData()
        md.setData(self.DR_MIME, name.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(md)
        pm = item.icon().pixmap(THUMB_SIZE, FIN_CELL_H * 2)
        pm = pm.copy(0, FIN_CELL_H, THUMB_SIZE, FIN_CELL_H)  # 只取 DR 半格
        if pm.isNull():
            pm = QPixmap(96, 64)
            pm.fill(Qt.GlobalColor.lightGray)
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        # 接受海豚格重排 (DOLPHIN_MIME) 与 DR 鳍交换 (DR_MIME)
        if event.mimeData().hasFormat(self.DR_MIME) \
                or event.mimeData().hasFormat(self.DOLPHIN_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.DR_MIME) \
                or event.mimeData().hasFormat(self.DOLPHIN_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasFormat(self.DOLPHIN_MIME):
            src = bytes(md.data(self.DOLPHIN_MIME)).decode("utf-8")
            if not src:
                event.ignore()
                return
            item = self.itemAt(event.position().toPoint())
            if item is None:
                self._move_dolphin_to_end(src)
            else:
                dst = item.data(Qt.ItemDataRole.UserRole)[0]
                if src != dst:
                    self._move_dolphin_before(src, dst)
                else:
                    event.ignore()
                    return
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            return
        if not md.hasFormat(self.DR_MIME):
            event.ignore()
            return
        src = bytes(md.data(self.DR_MIME)).decode("utf-8")
        if not src:
            event.ignore()
            return
        item = self.itemAt(event.position().toPoint())
        if item is None:
            # 拖到空白处: 用该海豚的 DR 鳍创建一只新海豚
            self.dr_dropped_on_empty.emit(src)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            return
        dst = item.data(Qt.ItemDataRole.UserRole)[0]
        if src != dst:
            self.dr_swapped.emit(src, dst)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def _move_dolphin_before(self, src, dst):
        """把 src 海豚移到 dst 海豚前面。"""
        if src == dst or src not in self._dolphin_order:
            return
        order = list(self._dolphin_order)
        order.remove(src)
        idx = order.index(dst) if dst in order else len(order)
        order.insert(idx, src)
        self._apply_dolphin_order(order)

    def _move_dolphin_to_end(self, src):
        """把 src 海豚移到末尾。"""
        if src not in self._dolphin_order:
            return
        order = list(self._dolphin_order)
        order.remove(src)
        order.append(src)
        self._apply_dolphin_order(order)

    def _apply_dolphin_order(self, order):
        """按 order 重新显示海豚格 (不改底层 DolphinID)。"""
        reordered = [self._dolphin_data[name] for name in order
                     if name in self._dolphin_data]
        self.show_dolphins(reordered, self._social)


class FinSlot(QLabel):
    """Match Mode 左侧鳍格: 显示当前海豚某一侧 (DL/DR) 的鳍图。
    滚轮缩放 (以光标为中心), 放大后左键拖拽平移; 未放大时把鳍图拖出本格
    (落到候选栏或海豚列表等接受拖出的位置) = 确认不匹配;
    接受右侧候选鳍的拖入 = 确认配对。"""

    OUT_MIME = "application/x-doris-dolphin-out"
    dropped_out = pyqtSignal(str, str)      # (cls, fid): 拖出确认不匹配
    candidate_dropped = pyqtSignal(str)     # (fid): 候选拖入确认配对

    def __init__(self, cls, parent=None):
        super().__init__(parent)
        self.cls = cls
        self.fid = None
        self._path = None
        self._pm = None        # 原始鳍图
        self._zoom = 1.0       # 相对适应窗口的倍率 (1 = 适应)
        self._off = QPoint(0, 0)  # 平移偏移 (屏幕像素)
        self._press = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 140)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAcceptDrops(True)
        self.setToolTip("滚轮或 +/- 缩放, 0 复位, 放大后左键拖拽平移;\n"
                        "未放大时把鳍图拖到格外 (候选栏/海豚列表) = 确认不匹配;\n"
                        "把右侧候选鳍拖入本格 = 确认配对")
        self._show_empty()

    def set_fin(self, fid, path):
        self.fid = fid
        self._path = path
        self._press = None
        self._pm = None
        if fid is None or not path:
            self._show_empty()
            return
        reader = QImageReader(path)
        sz = QSize(reader.size())
        sz.scale(1280, 800, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(sz)
        pm = QPixmap.fromImage(reader.read())
        if pm.isNull():
            self._show_empty("(load failed)")
            return
        self._pm = pm
        self._zoom = 1.0
        self._off = QPoint(0, 0)
        self.setText("")
        self.update()

    def _show_empty(self, text=None):
        self._pm = None
        self._zoom = 1.0
        self._off = QPoint(0, 0)
        self.setText(text or "(no %s fin)" % self.cls)
        self.update()

    def _fit_scale(self):
        if self._pm is None or self._pm.isNull():
            return 1.0
        return min(self.width() / self._pm.width(),
                   self.height() / self._pm.height())

    def _img_rect(self):
        """当前缩放/偏移下鳍图在控件内的绘制矩形 (浮点)。"""
        s = self._fit_scale() * self._zoom
        tw, th = self._pm.width() * s, self._pm.height() * s
        x = (self.width() - tw) / 2 + self._off.x()
        y = (self.height() - th) / 2 + self._off.y()
        return x, y, tw, th

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pm is None or self._pm.isNull():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        x, y, tw, th = self._img_rect()
        p.drawPixmap(QRectF(x, y, tw, th), self._pm,
                     QRectF(self._pm.rect()))
        p.end()

    # -- 滚轮/键盘缩放, 放大后左键平移 ----------------------------------------
    def _zoom_to(self, factor, anchor):
        """以 anchor 为锚点缩放 (保持锚点下的图像点不动)。"""
        if self._pm is None or self._pm.isNull():
            return
        new = max(1.0, min(10.0, self._zoom * factor))
        if new == self._zoom:
            return
        x0, y0, _, _ = self._img_rect()
        k = new / self._zoom
        s = self._fit_scale() * new
        self._off = QPoint(
            int(anchor.x() - (anchor.x() - x0) * k
                - (self.width() - self._pm.width() * s) / 2),
            int(anchor.y() - (anchor.y() - y0) * k
                - (self.height() - self._pm.height() * s) / 2))
        self._zoom = new
        if self._zoom == 1.0:
            self._off = QPoint(0, 0)
        self.update()

    def zoom_in(self):
        self._zoom_to(1.25, QPoint(self.width() // 2, self.height() // 2))

    def zoom_out(self):
        self._zoom_to(0.8, QPoint(self.width() // 2, self.height() // 2))

    def reset_zoom(self):
        self._zoom = 1.0
        self._off = QPoint(0, 0)
        self.update()

    def wheelEvent(self, event):
        self._zoom_to(1.25 if event.angleDelta().y() > 0 else 0.8,
                      event.position().toPoint())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            # 中键: 原图查看器 (鳍框/标签叠加)
            panel = self.parent()
            if isinstance(panel, MatchPanel):
                panel.open_slot_orig(self)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.fid is not None:
            self._press = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press is None:
            super().mouseMoveEvent(event)
            return
        if self._zoom > 1.0:
            # 放大状态: 左键拖拽 = 平移查看区域
            self._off += event.pos() - self._press
            self._press = event.pos()
            self.update()
        elif (event.pos() - self._press).manhattanLength() > 8:
            # 未放大: 拖出本格 = 确认不匹配
            self._press = None
            md = QMimeData()
            md.setData(self.OUT_MIME,
                       ("%s|%s" % (self.cls, self.fid)).encode("utf-8"))
            drag = QDrag(self)
            drag.setMimeData(md)
            if self._pm is not None and not self._pm.isNull():
                drag.setPixmap(self._pm.scaledToWidth(
                    96, Qt.TransformationMode.SmoothTransformation))
            # 只有落点接受拖出 (候选栏/海豚列表) 才算确认不匹配;
            # 拖到空白处松手返回 IgnoreAction, 视为误操作
            if drag.exec(Qt.DropAction.MoveAction) != Qt.DropAction.IgnoreAction:
                self.dropped_out.emit(self.cls, str(self.fid))
        super().mouseMoveEvent(event)

    # -- 双击/中键打开大图查看窗口 ---------------------------------------------
    def mouseDoubleClickEvent(self, event):
        """双击: 鳍裁剪图查看器 (该海豚该侧全部鳍 + 标签)。"""
        panel = self.parent()
        if isinstance(panel, MatchPanel):
            panel.open_slot_fin(self)
        super().mouseDoubleClickEvent(event)

    # -- 候选拖入 = 确认配对 --------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(CandidateList.MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(CandidateList.MIME):
            event.ignore()
            return
        fid = bytes(event.mimeData().data(CandidateList.MIME)).decode("utf-8")
        if fid:
            self.candidate_dropped.emit(fid)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()


class CandidateList(QListWidget):
    """Match Mode 右侧候选鳍列表: 与当前海豚缺失侧最相似的异侧鳍,
    拖动候选到左侧鳍格确认配对; 同时接受左侧鳍格的拖出落点 (使
    "拖出=确认不匹配" 的拖拽能完成)。"""

    MIME = "application/x-doris-candidate-fin"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setIconSize(QSize(220, 150))
        self.setGridSize(QSize(240, 170))
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Movement.Static)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.viewport().setAcceptDrops(True)
        self.setItemDelegate(ThumbOverlayDelegate(self))
        self._loaders = []
        self._items = {}
        self._imgs = {}  # path -> 已加载的 QImage (重缩放用)
        self._labels = {}      # FinID -> sim 标签, 由 delegate 叠加
        self._orig_dims = {}   # FinID -> (原图宽, 原图高)
        self._cand_fids = []   # 最近一批候选 (大图查看器浏览集)
        self._cand_paths = []

    def candidate_paths(self):
        return list(self._cand_paths)

    def candidate_label_map(self):
        """候选鳍图路径 -> 'Fin001 sim=0.83' 标注 (fin 查看器用)。"""
        return {p: self._labels[str(f)]
                for f, p in zip(self._cand_fids, self._cand_paths)
                if str(f) in self._labels}

    def show_candidates(self, fids, paths, labels):
        """(fids, paths, labels): 后台分片加载候选鳍缩略图,
        'Fin001 sim=0.83' 由 delegate 按 FinID 键叠加在图上。
        候选图已是 fin 裁剪, 不再叠加鳍框。"""
        self._stop_loaders()
        self.clear()
        self._items = {}
        self._imgs = {}
        self._cand_fids = [str(f) for f in fids]
        self._cand_paths = list(paths)
        self._labels = {str(f): labels[p]
                        for f, p in zip(fids, paths) if p in labels}
        self._orig_dims = {}
        for fid, p in zip(fids, paths):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(fid))
            item.setData(Qt.ItemDataRole.UserRole + 1, p)
            item.setSizeHint(self.gridSize())
            self.addItem(item)
            self._items[p] = item
        if not paths:
            return
        try:
            cache_dir = os.path.join(os.path.commonpath(paths),
                                     ".doris_thumbs")
        except ValueError:
            cache_dir = None
        n = min(4, os.cpu_count() or 4, len(paths))
        for k in range(n):
            loader = ThumbLoader(paths[k::n], cache_dir=cache_dir,
                                 thumb_size=768)
            loader.loaded.connect(self._set_thumb)
            loader.start()
            self._loaders.append(loader)

    def _set_thumb(self, path, img, ow=None, oh=None):
        self._imgs[path] = img
        item = self._items.get(path)
        if item is not None:
            if ow is not None and oh is not None:
                self._orig_dims[str(item.data(Qt.ItemDataRole.UserRole))] \
                    = (ow, oh)
            # QIcon 不会把小的源图放大到 iconSize, 需先手动放大再设入,
            # 否则 256px 的缩略图在大格子里仍显示为小图
            sz = self.iconSize()
            if img.width() < sz.width() or img.height() < sz.height():
                img = img.scaled(sz, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _stop_loaders(self):
        for l in self._loaders:
            l.stop()
        for l in self._loaders:
            l.wait()
        self._loaders = []

    def sync_cell_size(self, w, h):
        """候选鳍图与中间列鳍格等大 (图标尺寸 + 每格尺寸同步更新)。"""
        self.setIconSize(QSize(w - 20, h - 20))
        self.setGridSize(QSize(w, h))
        for i in range(self.count()):
            self.item(i).setSizeHint(QSize(w, h))
        # 已加载的图标按新尺寸重缩放 (QIcon 不会自动放大)
        for p, img in self._imgs.items():
            if p in self._items:
                self._set_thumb(p, img)

    # -- 拖出候选 ------------------------------------------------------------
    def startDrag(self, actions):
        item = self.currentItem()
        if item is None:
            return
        md = QMimeData()
        md.setData(self.MIME, str(item.data(Qt.ItemDataRole.UserRole)).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(md)
        pm = item.icon().pixmap(self.iconSize())
        if pm.isNull():
            pm = QPixmap(96, 64)
            pm.fill(Qt.GlobalColor.lightGray)
        drag.setPixmap(pm)
        drag.exec(Qt.DropAction.MoveAction)

    # -- 双击/中键打开大图查看窗口 ---------------------------------------------
    def mouseDoubleClickEvent(self, event):
        """双击: 鳍裁剪图查看器 (候选集 + 标签)。"""
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            p = item.data(Qt.ItemDataRole.UserRole + 1)
            panel = self.parent()
            if p and isinstance(panel, MatchPanel):
                panel.open_candidate_fin(p)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            # 中键: 原图查看器 (鳍框/标签叠加)
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                p = item.data(Qt.ItemDataRole.UserRole + 1)
                panel = self.parent()
                if p and isinstance(panel, MatchPanel):
                    panel.open_candidate_orig(p)
            event.accept()
            return
        super().mousePressEvent(event)

    # -- 滚轮: 减小步长, 滚动更细腻 -------------------------------------------
    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        delta = event.angleDelta().y()
        # 默认一格 (delta=120) 会滚过多行, 改为约 1/4 格像素量
        bar.setValue(bar.value() - delta // 4)
        event.accept()

    # -- 接受左侧鳍格的拖出落点 (no-op, 拖拽完成即确认不匹配) -----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(FinSlot.OUT_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(FinSlot.OUT_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()  # 不匹配由 FinSlot 在拖拽完成时发出信号
        else:
            event.ignore()


class MatchPanel(QWidget):
    """Match Mode 面板: 左列上下两格为当前海豚的 DL/DR 鳍图
    (把鳍图拖出格子 = 确认不匹配); 中列为海豚列表; 右列为缺失侧
    (DL 或 DR) 的候选鳍 (拖入鳍格 = 确认配对)。"""

    assign_requested = pyqtSignal(str, str, str)  # (dolphin name, cls, fid)
    unmatch_requested = pyqtSignal(str, str)      # (cls, fid)

    def __init__(self, candidate_provider, parent=None, fin_lister=None,
                 fin_labels=None, orig_resolver=None, photo_boxes=None,
                 photo_labels=None):
        super().__init__(parent)
        # (fid, cls) -> (fids, paths, labels), 由 MainWindow 注入
        self._candidates_of = candidate_provider
        self._fin_lister = fin_lister     # (DolphinID, DL/DR) -> [fin 路径]
        self._fin_labels = fin_labels or {}   # fin 路径 -> 标注
        self._orig_resolver = orig_resolver   # fin 路径 -> 原图路径
        self._photo_boxes = photo_boxes or {} # 原图路径 -> 鳍框
        self._photo_labels = photo_labels or {}  # 原图路径 -> 标注
        self._dolphins = []
        self._name = None      # 当前选中的海豚名
        self._cand_cls = None  # 候选鳍对应的侧 (当前海豚缺失的那侧)

        lay = QHBoxLayout(self)
        self.dolphin_list = QListWidget()
        self.dolphin_list.setAcceptDrops(True)
        self.dolphin_list.viewport().setAcceptDrops(True)
        self.dolphin_list.setWordWrap(False)
        self.dolphin_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.dolphin_list.setMaximumWidth(200)
        self.dolphin_list.currentRowChanged.connect(self._on_select)
        lay.addWidget(self.dolphin_list, 0)
        # 海豚列表也接受鳍格拖出 (落点之一)
        self.dolphin_list.dragEnterEvent = self._list_accept_out
        self.dolphin_list.dragMoveEvent = self._list_accept_out
        self.dolphin_list.dropEvent = self._list_drop_out

        left = QVBoxLayout()
        self.cap_dl = QLabel("DL")
        self.slot_dl = FinSlot("DL")
        self.cap_dr = QLabel("DR")
        self.slot_dr = FinSlot("DR")
        left.addWidget(self.cap_dl)
        left.addWidget(self.slot_dl, 1)
        left.addWidget(self.cap_dr)
        left.addWidget(self.slot_dr, 1)
        lay.addLayout(left, 2)

        self.cand_list = CandidateList()
        lay.addWidget(self.cand_list, 2)

        self.slot_dl.dropped_out.connect(self.unmatch_requested)
        self.slot_dr.dropped_out.connect(self.unmatch_requested)
        self.slot_dl.candidate_dropped.connect(self._assign)
        self.slot_dr.candidate_dropped.connect(self._assign)

        # 缩放宽快捷键: 用 QShortcut 而不是 keyPressEvent, 否则焦点落在
        # 列表/鳍格子控件时按键会被它们吃掉, 面板的 keyPressEvent 收不到
        for key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            sc = QShortcut(QKeySequence(key), self,
                           context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._zoom_in_all)
        for key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            sc = QShortcut(QKeySequence(key), self,
                           context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._zoom_out_all)
        sc = QShortcut(QKeySequence(Qt.Key.Key_0), self,
                       context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self._zoom_reset_all)

    def _zoom_in_all(self):
        self.slot_dl.zoom_in()
        self.slot_dr.zoom_in()

    def _zoom_out_all(self):
        self.slot_dl.zoom_out()
        self.slot_dr.zoom_out()

    def _zoom_reset_all(self):
        self.slot_dl.reset_zoom()
        self.slot_dr.reset_zoom()

    @staticmethod
    def _list_accept_out(event):
        if event.mimeData().hasFormat(FinSlot.OUT_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def _list_drop_out(event):
        MatchPanel._list_accept_out(event)

    def current_name(self):
        return self._name

    # -- 大图查看窗口 (鳍格/候选的双击与中键) -----------------------------------
    def open_slot_fin(self, slot):
        """双击鳍格: 鳍裁剪图查看器 (该海豚该侧全部鳍 + 标签)。"""
        self._open_side_viewer(slot.cls, slot._path, orig=False)

    def open_slot_orig(self, slot):
        """中键鳍格: 原图查看器 (鳍框/标签叠加)。"""
        self._open_side_viewer(slot.cls, slot._path, orig=True)

    def open_candidate_fin(self, path):
        """双击候选: 鳍裁剪图查看器 (候选集 + 标签)。"""
        paths = [p for p in self.cand_list.candidate_paths() if p]
        _show_fin_viewer(self, paths, path,
                         self.cand_list.candidate_label_map())

    def open_candidate_orig(self, path):
        """中键候选: 原图查看器 (鳍框/标签叠加)。"""
        fins = [p for p in self.cand_list.candidate_paths() if p]
        _show_orig_viewer(self, fins, path, self._orig_resolver,
                          self._photo_boxes, self._photo_labels)

    def _open_side_viewer(self, cls, path, orig):
        if self._name is None or not path:
            return
        fins = _side_browse_set(self._fin_lister, self._name, cls, path)
        if orig:
            _show_orig_viewer(self, fins, path, self._orig_resolver,
                              self._photo_boxes, self._photo_labels)
        else:
            _show_fin_viewer(self, fins, path, self._fin_labels)

    def resizeEvent(self, event):
        """候选鳍图与中间列鳍格等大。"""
        sz = self.slot_dl.size()
        if sz.width() > 60 and sz.height() > 60:
            self.cand_list.sync_cell_size(sz.width(), sz.height())
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        """+/= 放大, - 缩小, 0 复位 (快捷键兜底, 见 __init__ 的 QShortcut)。"""
        k = event.key()
        if k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_in_all()
        elif k in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._zoom_out_all()
        elif k == Qt.Key.Key_0:
            self._zoom_reset_all()
        else:
            super().keyPressEvent(event)

    def show_dolphins(self, dolphins, keep=None):
        """dolphins: [(name, dl_fid, dr_fid, dl_path, dr_path)];
        keep: 刷新后尽量保持选中的海豚名。"""
        self._dolphins = list(dolphins)
        self.dolphin_list.blockSignals(True)
        self.dolphin_list.clear()
        for name, dl, dr, _, _ in self._dolphins:
            QListWidgetItem("%s  L=%s R=%s"
                            % (name, dl if dl is not None else "None  ",
                               dr if dr is not None else "None  "),
                            self.dolphin_list)
        self.dolphin_list.blockSignals(False)
        if not self._dolphins:
            self._on_select(-1)
            return
        row = 0
        if keep is not None:
            for i, d in enumerate(self._dolphins):
                if d[0] == keep:
                    row = i
                    break
        self.dolphin_list.setCurrentRow(row)

    def _on_select(self, row):
        if row < 0 or row >= len(self._dolphins):
            self._name = None
            self._cand_cls = None
            self.slot_dl.set_fin(None, None)
            self.slot_dr.set_fin(None, None)
            self.cap_dl.setText("DL")
            self.cap_dr.setText("DR")
            self.cand_list.show_candidates([], [], {})
            return
        name, dl_fid, dr_fid, dl_path, dr_path = self._dolphins[row]
        self._name = name
        self.slot_dl.set_fin(dl_fid, dl_path)
        self.slot_dr.set_fin(dr_fid, dr_path)
        self.cap_dl.setText("DL  %s" % dl_fid
                            if dl_fid is not None else "DL  (missing)")
        self.cap_dr.setText("DR  %s" % dr_fid
                            if dr_fid is not None else "DR  (missing)")
        # 缺哪侧就为哪侧找候选: 用已有侧的鳍去相似检索异侧
        if dl_fid is None and dr_fid is not None:
            self._cand_cls = "DL"
            fids, paths, labels = self._candidates_of(dr_fid, "DR")
        elif dr_fid is None and dl_fid is not None:
            self._cand_cls = "DR"
            fids, paths, labels = self._candidates_of(dl_fid, "DL")
        else:  # 两侧齐全或皆空: 无候选
            self._cand_cls = None
            fids, paths, labels = [], [], {}
        self.cand_list.show_candidates(fids, paths, labels)

    def _assign(self, fid):
        if self._name is not None and self._cand_cls is not None:
            self.assign_requested.emit(self._name, self._cand_cls, fid)
