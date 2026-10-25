# -*- coding: utf-8 -*-
"""海豚页相关控件：DolphinTab、FinSlot、CandidateList、MatchPanel。"""

import os
import re

import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QListWidget, QListWidgetItem, QLabel, QVBoxLayout,
    QHBoxLayout, QSplitter, QScrollArea, QPushButton, QFrame,
    QMessageBox, QDialog, QShortcut,
)
from PyQt5.QtGui import (
    QPixmap, QIcon, QImageReader, QPainter, QColor, QPen, QDrag, QFont,
    QKeySequence,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QMimeData, QPoint, QRect, QRectF

from .common import (
    ImageGrid, DolphinLoader, ThumbOverlayDelegate, ZoomableView,
    ImageViewerDialog, MIME_PATHS, THUMB_SIZE, FIN_CELL_H, BOX_COLORS,
    draw_boxes, draw_label, ThumbLoader
)

class DolphinTab(QListWidget):
    """海豚配对页: 每只海豚占一格, 上=DL 鳍图, 下=DR 鳍图, 标题 Dphxxx。
    初始所有 DL FinID 放在上格; DR 按与 DL 的最高相似度自动配对。
    下格 (DR) 可拖到其它海豚格: 双方都有 DR 时互换, 目标无 DR 时移动。
    双击鳍图 (上半格 DL / 下半格 DR) 打开大图查看窗口。"""

    dr_swapped = pyqtSignal(str, str)  # (源 Dph 名, 目标 Dph 名)
    DR_MIME = "application/x-doris-dolphin-dr"

    def __init__(self, parent=None):
        super().__init__(parent)
        cell = FIN_CELL_H
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setIconSize(QSize(THUMB_SIZE, cell * 2))  # 上下两格
        self.setGridSize(QSize(THUMB_SIZE + 16, cell * 2 + 16))
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Static)
        # 滚轮按像素平滑滚动
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.horizontalScrollBar().setSingleStep(24)
        self.setItemDelegate(ThumbOverlayDelegate(self))
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setDropIndicatorShown(True)
        # Movement.Static 会让 Qt 关掉 viewport 的 acceptDrops (静态视图不允许
        # 内部移动), 导致拖放根本投递不进来; 当前的 dropEvent 自己处理落点,
        # 不依赖视图移动, 显式重新打开
        self.viewport().setAcceptDrops(True)
        self._loaders = []
        self._items = {}
        self._press_bottom = False  # 按下位置是否在 DR (下) 格

    def show_dolphins(self, dolphins):
        """dolphins: [(name, dl_fid, dr_fid, dl_path, dr_path)];
        后台分片加载合成缩略图。"""
        self._stop_loaders()
        self.clear()
        self._items = {}
        cell = FIN_CELL_H
        for name, dl_fid, dr_fid, dl, dr in dolphins:
            item = QListWidgetItem()  # 名称由合成图画在上下鳍之间, 不再用顶部条
            item.setData(Qt.UserRole, (name, dl_fid, dr_fid, dl, dr))
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
        # (name, dl_fid, dr_fid, dl_path, dr_path)
        jobs = [(d[0], d[1], d[2], d[3], d[4]) for d in dolphins]
        n = min(8, os.cpu_count() or 4, len(jobs))
        for k in range(n):
            loader = DolphinLoader(jobs[k::n], cache_dir=cache_dir)
            loader.loaded.connect(self._set_thumb)
            loader.start()
            self._loaders.append(loader)

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
    def mousePressEvent(self, event):
        self._press_bottom = False
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item is not None:
                r = self.visualItemRect(item)
                self._press_bottom = event.pos().y() > r.top() + r.height() // 2
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """双击鳍图打开大图查看窗口: 上半格看 DL, 下半格看 DR。"""
        item = self.itemAt(event.pos())
        if item is not None:
            r = self.visualItemRect(item)
            bottom = event.pos().y() > r.top() + r.height() // 2
            name, dl_fid, dr_fid, dl, dr = item.data(Qt.UserRole)
            path = dr if bottom else dl
            if path:
                ImageViewerDialog([path], 0, self).exec_()
        super().mouseDoubleClickEvent(event)

    def startDrag(self, actions):
        """只有按下点在 DR 格且该海豚有 DR 时才可拖出; 拖拽图为下半格。"""
        item = self.currentItem()
        if item is None or not self._press_bottom:
            return
        name, dl_fid, dr_fid, dl, dr = item.data(Qt.UserRole)
        if dr_fid is None:
            return
        md = QMimeData()
        md.setData(self.DR_MIME, name.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(md)
        pm = item.icon().pixmap(THUMB_SIZE, FIN_CELL_H * 2)
        pm = pm.copy(0, FIN_CELL_H, THUMB_SIZE, FIN_CELL_H)  # 只取 DR 半格
        if pm.isNull():
            pm = QPixmap(96, 64)
            pm.fill(Qt.lightGray)
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        # 只认本页自定义的 DR_MIME; 不检查 event.source() (PyQt 包装对象
        # 身份比较不可靠, 可能导致真实拖拽被误拒)
        if event.mimeData().hasFormat(self.DR_MIME):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.DR_MIME):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None or not event.mimeData().hasFormat(self.DR_MIME):
            event.ignore()
            return
        src = bytes(event.mimeData().data(self.DR_MIME)).decode("utf-8")
        dst = item.data(Qt.UserRole)[0]
        if src and src != dst:
            self.dr_swapped.emit(src, dst)
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()


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
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 140)
        self.setFrameShape(QFrame.StyledPanel)
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
        sz.scale(1280, 800, Qt.KeepAspectRatio)
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
        p.setRenderHint(QPainter.SmoothPixmapTransform)
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
                      event.pos())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.fid is not None:
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
                    96, Qt.SmoothTransformation))
            # 只有落点接受拖出 (候选栏/海豚列表) 才算确认不匹配;
            # 拖到空白处松手返回 IgnoreAction, 视为误操作
            if drag.exec_(Qt.MoveAction) != Qt.IgnoreAction:
                self.dropped_out.emit(self.cls, str(self.fid))
        super().mouseMoveEvent(event)

    # -- 双击打开大图查看窗口 --------------------------------------------------
    def mouseDoubleClickEvent(self, event):
        if self._path:
            ImageViewerDialog([self._path], 0, self).exec_()
        super().mouseDoubleClickEvent(event)

    # -- 候选拖入 = 确认配对 --------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(CandidateList.MIME):
            event.setDropAction(Qt.MoveAction)
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
            event.setDropAction(Qt.MoveAction)
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
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setIconSize(QSize(220, 150))
        self.setGridSize(QSize(240, 170))
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Static)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.viewport().setAcceptDrops(True)
        self._loaders = []
        self._items = {}
        self._imgs = {}  # path -> 已加载的 QImage (重缩放用)

    def show_candidates(self, fids, paths, labels):
        """(fids, paths, labels): 后台分片加载候选鳍缩略图,
        'FinID_x sim=0.83' 由加载器叠加在图上。"""
        self._stop_loaders()
        self.clear()
        self._items = {}
        self._imgs = {}
        for fid, p in zip(fids, paths):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, str(fid))
            item.setData(Qt.UserRole + 1, p)
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
            loader = ThumbLoader(paths[k::n], labels=labels,
                                 cache_dir=cache_dir, thumb_size=768)
            loader.loaded.connect(self._set_thumb)
            loader.start()
            self._loaders.append(loader)

    def _set_thumb(self, path, img):
        self._imgs[path] = img
        item = self._items.get(path)
        if item is not None:
            # QIcon 不会把小的源图放大到 iconSize, 需先手动放大再设入,
            # 否则 256px 的缩略图在大格子里仍显示为小图
            sz = self.iconSize()
            if img.width() < sz.width() or img.height() < sz.height():
                img = img.scaled(sz, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
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
        md.setData(self.MIME, str(item.data(Qt.UserRole)).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(md)
        pm = item.icon().pixmap(self.iconSize())
        if pm.isNull():
            pm = QPixmap(96, 64)
            pm.fill(Qt.lightGray)
        drag.setPixmap(pm)
        drag.exec_(Qt.MoveAction)

    # -- 双击打开大图查看窗口 --------------------------------------------------
    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None:
            p = item.data(Qt.UserRole + 1)
            if p:
                ImageViewerDialog([p], 0, self).exec_()
        super().mouseDoubleClickEvent(event)

    # -- 接受左侧鳍格的拖出落点 (no-op, 拖拽完成即确认不匹配) -----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(FinSlot.OUT_MIME):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(FinSlot.OUT_MIME):
            event.setDropAction(Qt.MoveAction)
            event.accept()  # 不匹配由 FinSlot 在拖拽完成时发出信号
        else:
            event.ignore()


class MatchPanel(QWidget):
    """Match Mode 面板: 左列上下两格为当前海豚的 DL/DR 鳍图
    (把鳍图拖出格子 = 确认不匹配); 中列为海豚列表; 右列为缺失侧
    (DL 或 DR) 的候选鳍 (拖入鳍格 = 确认配对)。"""

    assign_requested = pyqtSignal(str, str, str)  # (dolphin name, cls, fid)
    unmatch_requested = pyqtSignal(str, str)      # (cls, fid)

    def __init__(self, candidate_provider, parent=None):
        super().__init__(parent)
        # (fid, cls) -> (fids, paths, labels), 由 MainWindow 注入
        self._candidates_of = candidate_provider
        self._dolphins = []
        self._name = None      # 当前选中的海豚名
        self._cand_cls = None  # 候选鳍对应的侧 (当前海豚缺失的那侧)

        lay = QHBoxLayout(self)
        self.dolphin_list = QListWidget()
        self.dolphin_list.setAcceptDrops(True)
        self.dolphin_list.viewport().setAcceptDrops(True)
        self.dolphin_list.setWordWrap(False)
        self.dolphin_list.setTextElideMode(Qt.ElideNone)
        self.dolphin_list.setMaximumWidth(180)
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
        for key in (Qt.Key_Plus, Qt.Key_Equal):
            sc = QShortcut(QKeySequence(key), self,
                           context=Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(self._zoom_in_all)
        for key in (Qt.Key_Minus, Qt.Key_Underscore):
            sc = QShortcut(QKeySequence(key), self,
                           context=Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(self._zoom_out_all)
        sc = QShortcut(QKeySequence(Qt.Key_0), self,
                       context=Qt.WidgetWithChildrenShortcut)
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
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def _list_drop_out(event):
        MatchPanel._list_accept_out(event)

    def current_name(self):
        return self._name

    def resizeEvent(self, event):
        """候选鳍图与中间列鳍格等大。"""
        sz = self.slot_dl.size()
        if sz.width() > 60 and sz.height() > 60:
            self.cand_list.sync_cell_size(sz.width(), sz.height())
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        """+/= 放大, - 缩小, 0 复位 (快捷键兜底, 见 __init__ 的 QShortcut)。"""
        k = event.key()
        if k in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_in_all()
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom_out_all()
        elif k == Qt.Key_0:
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
            QListWidgetItem("%s  DL=%s DR=%s"
                            % (name, dl if dl is not None else "-",
                               dr if dr is not None else "-"),
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
        self.cap_dl.setText("DL  FinID_%s" % dl_fid
                            if dl_fid is not None else "DL  (missing)")
        self.cap_dr.setText("DR  FinID_%s" % dr_fid
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
