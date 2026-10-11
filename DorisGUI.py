#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DorisGUI — 中华白海豚照片浏览与分组 GUI

7 个标签页:
  1. 按连拍分组 (Continuous Shots, IMAGE_METAINFO.csv 的 shot_id)
  2. 按鳍部位分组 (Fin Aspect: DL=Left / DR=Right / Others=Misc / Wrong=0 )
  3. 按模糊与否分组 (Blur / Clear / Mid, 依据 FIN_METAINFO.csv 的 clear 列)
  4. 按个体编号分组 (Fin ID, FIN_METAINFO.csv 的 FinID 列; 顶栏三个互斥模式:
     Review=分组浏览 / Match=每 shot 最佳鳍图三栏匹配 / Overview=每个个体一张代表图
     (可拖放一张代表图到另一张上合并两个 FinID, 对应 DolphinID 一并合并))
  5. The Dolphin (顶栏 "Mode" 菜单三互斥模式: Overview=海豚格 DR 拖拽交换 /
     Match=匹配面板 / Review=左侧 DolphinID 列表 + 右侧全部鳍图)
  6. 按社会结构分组 (Social Structure, 同一连拍出现多于一个个体 (不含 FinID_0) 的 shot 分组为 NNxxx)
  7. 统计 (Statistics, FinID 鳍图数柱状图 + FinID shot 数柱状图 + 部位占比饼图)

用法:
  python DorisGUI.py [root_folder]
默认 root 为演示目录。
"""

import os
import re
import sys
import shutil
import hashlib
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QDialog,
    QScrollArea, QFileDialog, QToolBar, QAction, QMessageBox,
    QGraphicsView, QGraphicsScene, QStyledItemDelegate, QStyle,
    QStyleOptionViewItem, QShortcut, QHBoxLayout, QPushButton,
    QStackedWidget, QFrame, QToolButton, QMenu, QCheckBox,
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QImageIOHandler, QPainter, QBrush, QColor, QDrag, QPen, QPalette, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPoint, QRect, QRectF, QTimer
import PyQt5

from doris_dataset import Dataset

# PyQt5 wheel bug: Qt derives its plugin path from the library location and
# mangles non-ASCII characters in it, leaving the plugin search path empty
# ("Could not find the Qt platform plugin ... in ''" / wayland shell
# integration not found). Point Qt at the whole bundled plugin tree
# explicitly before QApplication is created.
os.environ.setdefault(
    "QT_PLUGIN_PATH",
    os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins"),
)

IMG_EXTS = (".jpg", ".jpeg", ".png")
THUMB_SIZE = 256
# Dolphin 配对鳍格高度: 与格宽保持 18:10, 上下两鳍图无缝相叠不留缝
FIN_CELL_H = int(THUMB_SIZE * 10 / 18)
MIME_PATHS = "application/x-doris-paths"


BOX_COLORS = {"DL": QColor(0, 200, 0), "DR": QColor(255, 140, 0),
              "Others": QColor(220, 0, 0), "ND": QColor(128, 128, 128)}


def draw_boxes(target, boxes, orig_w, orig_h):
    """在目标 QImage/QPixmap 上画鳍框 (orig_w/h 为框坐标对应的图像尺寸)。
    boxes: [(x0, y0, x1, y1, cls[, extra]), ...] 图像像素坐标;
    extra (如 FinID) 非空时追加在类别文字后。
    cls 以 '_' 开头时只用它取框颜色, 不叠加类别文字 (社会结构页用)。"""
    sx = target.width() / orig_w
    sy = target.height() / orig_h
    pen_w = max(2, target.width() // 300)
    p = QPainter(target)
    f = p.font()
    f.setPixelSize(max(12, target.width() // 40))
    f.setBold(True)
    p.setFont(f)
    placed = []  # 已放置文字的矩形, 避免多个标注互相重叠
    box_rects = [QRect(int(b[0] * sx), int(b[1] * sy),
                       int((b[2] - b[0]) * sx), int((b[3] - b[1]) * sy))
                 for b in boxes]
    for box, rect in zip(boxes, box_rects):
        cls = box[4]
        hide_cls = cls.startswith("_")  # '_' 前缀: 只用类别取颜色, 不叠加类别文字
        if hide_cls:
            cls = cls[1:]
        color = BOX_COLORS.get(cls, Qt.yellow)
        p.setPen(QPen(color, pen_w))
        p.drawRect(rect)
        text = "" if hide_cls else cls
        if len(box) > 5 and box[5]:
            text = "%s %s" % (text, box[5]) if text else box[5]
        if not text:
            continue
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        # 候选位置 (文字包围盒 top-left): 只放框外, 框上方 -> 框下方
        candidates = [
            (rect.left() + 2, rect.top() - 2 - th),
            (rect.left() + 2, rect.bottom() + 2),
        ]
        label_rect = None
        for cx, cy in candidates:
            if cx + tw > target.width() - 2:  # 右侧越界则左移
                cx = max(2, target.width() - 2 - tw)
            r = QRect(cx, cy, tw, th)
            if r.top() < 0 or r.bottom() > target.height():
                continue  # 上下越界的位置不可用
            if any(r.intersects(b) for b in box_rects):
                continue  # 不压任何鳍框
            if any(r.intersects(o) for o in placed):
                continue  # 与已有标注重叠
            label_rect = r
            break
        if label_rect is None:
            # 全部冲突时退回框上/下方, 钳制到图像内且仍不压鳍框
            for cy in (rect.top() - 2 - th, rect.bottom() + 2):
                cx = min(max(2, rect.left() + 2),
                         max(2, target.width() - 2 - tw))
                cy = min(max(0, cy), max(0, target.height() - th))
                r = QRect(cx, cy, tw, th)
                if (not any(r.intersects(b) for b in box_rects)
                        and not any(r.intersects(o) for o in placed)):
                    label_rect = r
                    break
            if label_rect is None:  # 鳍框占满全图等极端情况, 放弃该标注
                continue
        placed.append(label_rect)
        p.drawText(label_rect.left(),
                   label_rect.top() + fm.ascent(), text)
    p.end()


def draw_label(target, text):
    """在图像左下角叠加半透明底文字标签 (shot_id 等)。"""
    p = QPainter(target)
    f = p.font()
    f.setPixelSize(max(12, min(56, target.width() // 14)))
    f.setBold(True)
    p.setFont(f)
    fm = p.fontMetrics()
    w, h = fm.horizontalAdvance(text) + 12, fm.height() + 6
    rect = QRect(2, target.height() - h - 2, w, h)
    p.fillRect(rect, QColor(0, 0, 0, 140))
    p.setPen(Qt.white)
    p.drawText(rect, Qt.AlignCenter, text)
    p.end()


# ---------------------------------------------------------------------------
# 缩略图后台加载 (多线程分片 + 磁盘缓存)
# ---------------------------------------------------------------------------
class ThumbLoader(QThread):
    loaded = pyqtSignal(str, QImage)

    def __init__(self, paths, boxes=None, labels=None, cache_dir=None,
                 thumb_size=THUMB_SIZE):
        super().__init__()
        self._paths = list(paths)
        self._boxes = boxes or {}
        self._labels = labels or {}
        self._cache_dir = cache_dir
        self._thumb_size = thumb_size
        self._stop = False

    def stop(self):
        self._stop = True

    def _cache_path(self, p, w=None, h=None):
        key = hashlib.md5(p.encode()).hexdigest()[:10]
        if w is None:  # 查找已有缓存
            prefix = key + "_"
            try:
                for fn in os.listdir(self._cache_dir):
                    if fn.startswith(prefix):
                        m = re.match(r"[0-9a-f]+_(\d+)x(\d+)_", fn)
                        if m:
                            return os.path.join(self._cache_dir, fn), \
                                int(m.group(1)), int(m.group(2))
            except OSError:
                pass
            return None, None, None
        return os.path.join(self._cache_dir,
                            "%s_%dx%d_%s.jpg" % (key, w, h, os.path.basename(p))), w, h

    def _load_clean(self, p):
        """读干净缩略图(不带框/标签), 优先磁盘缓存。返回 (img, orig_w, orig_h)。"""
        if self._cache_dir:
            cp, ow, oh = self._cache_path(p)
            try:
                if cp and os.path.getmtime(cp) >= os.path.getmtime(p):
                    img = QImage(cp)
                    # 缓存图太小 (如 256px 缩略图) 不满足大尺寸请求时重读原图
                    if not img.isNull() \
                            and max(img.width(), img.height()) \
                            >= self._thumb_size * 0.9:
                        return img, ow, oh
            except OSError:
                pass
        reader = QImageReader(p)
        trans = reader.transformation()
        stored_w, stored_h = reader.size().width(), reader.size().height()
        sz = QSize(reader.size())
        if trans in (QImageIOHandler.TransformationRotate90,
                     QImageIOHandler.TransformationRotate270,
                     QImageIOHandler.TransformationFlipAndRotate90,
                     QImageIOHandler.TransformationMirrorAndRotate90):
            sz.transpose()
        # setScaledSize 不保持宽高比, 需按原始尺寸(含 EXIF 旋转)手动算目标尺寸
        sz.scale(self._thumb_size, self._thumb_size, Qt.KeepAspectRatio)
        reader.setScaledSize(sz)
        img = reader.read()
        if not img.isNull() and self._cache_dir:
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
                cp, _, _ = self._cache_path(p, stored_w, stored_h)
                img.save(cp, quality=85)
            except OSError:
                pass
        return img, stored_w, stored_h

    def run(self):
        for p in self._paths:
            if self._stop:
                break
            try:
                img, ow, oh = self._load_clean(p)
            except Exception:
                continue  # 单个文件失败不中断整个加载
            if img is None or img.isNull():
                continue
            boxes = self._boxes.get(p)
            if boxes:
                draw_boxes(img, boxes, ow, oh)
            label = self._labels.get(p) or self._labels.get(os.path.basename(p))
            if label:
                draw_label(img, label)
            self.loaded.emit(p, img)


class DolphinLoader(ThumbLoader):
    """海豚配对页加载器: 每个条目两张鳍图 (上 DL / 下 DR) 合成一张缩略图。
    每张鳍图先居中裁剪到鳍格宽高比再缩放填满, 拉伸变形由裁剪避免。
    self._paths 的元素为 (key, dl_path, dr_path); 复用 _load_clean 的磁盘缓存。"""

    def run(self):
        for key, dl_fid, dr_fid, dl, dr in self._paths:
            if self._stop:
                break
            try:
                img = self._compose(key, dl_fid, dr_fid, dl, dr)
            except Exception:
                continue  # 单只失败不中断整个加载
            if img is not None and not img.isNull():
                self.loaded.emit(key, img)

    @staticmethod
    def _crop_ratio(img, rw=18, rh=10):
        """居中裁剪 img 到 rw:rh 宽高比 (裁中间, 不旋转不拉伸)。"""
        w, h = img.width(), img.height()
        if w * rh > h * rw:  # 图比目标更宽: 裁宽度
            cw = h * rw // rh
            return img.copy((w - cw) // 2, 0, cw, h)
        ch = w * rh // rw  # 图比目标更高: 裁高度
        return img.copy(0, (h - ch) // 2, w, ch)

    def _compose(self, name, dl_fid, dr_fid, dl, dr):
        w, cell = THUMB_SIZE, FIN_CELL_H
        out = QImage(w, cell * 2, QImage.Format_RGB32)
        out.fill(QColor(100, 100, 100))
        p = QPainter(out)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        for i, (path, tag, fid) in enumerate(
                ((dl, "DL", dl_fid), (dr, "DR", dr_fid))):
            y0 = i * cell
            if path:
                img, _, _ = self._load_clean(path)
                if img is not None and not img.isNull():
                    # 居中裁到鳍格宽高比再精确填满, 上下两鳍无缝相叠
                    img = self._crop_ratio(img, w, cell)
                    t = img.scaled(w, cell, Qt.IgnoreAspectRatio,
                                   Qt.SmoothTransformation)
                    p.drawImage(0, y0, t)
            p.setPen(QColor(255, 255, 255))
            f = p.font()
            f.setPixelSize(14)
            f.setBold(True)
            p.setFont(f)
            if fid is not None:  # 每格左下角标 DL/DR, 右侧标对应 FinID
                try:  # FinID 常被读成浮点, 显示为整数
                    fid_s = str(int(float(fid)))
                except (TypeError, ValueError):
                    fid_s = str(fid)
                p.drawText(6, y0 + cell - 8,
                           "FinID_%s" % (fid_s))
        # 海豚编号居中叠在上下背鳍之间 (半透明底条保证可读)
        f = p.font()
        f.setPixelSize(16)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        tw, th = fm.horizontalAdvance(name), fm.height()
        rect = QRect((w - tw) // 2 - 8, cell - th // 2 - 2,
                     tw + 16, th + 4)
        p.fillRect(rect, QColor(0, 0, 0, 0))
        p.setPen(Qt.white)
        p.drawText(rect, Qt.AlignCenter, name)
        p.end()
        return out


class ThumbOverlayDelegate(QStyledItemDelegate):
    """文件名叠加在缩略图顶部(半透明底条), 而不是默认的图标下方。"""

    BAR_COLOR = QColor(0, 0, 0, 140)

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        opt.text = ""  # 文本由我们叠加绘制 
        selected = bool(opt.state & QStyle.State_Selected)
        # 选中/焦点态由我们在图标上绘制; 交给默认样式会在空文本区留下一小截高亮条
        opt.state &= ~QStyle.State_Selected
        opt.state &= ~QStyle.State_HasFocus
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        icon_rect = style.subElementRect(
            QStyle.SE_ItemViewItemDecoration, opt, opt.widget)
        if not icon_rect.isValid():
            icon_rect = opt.rect
        painter.save()
        if selected:  # 选中: 图标描边 + 轻微染色
            hl = opt.palette.color(QPalette.Highlight)
            painter.setPen(QPen(hl, 4))
            painter.drawRect(icon_rect.adjusted(1, 1, -2, -2))
            painter.fillRect(icon_rect, QColor(hl.red(), hl.green(),
                                               hl.blue(), 40))
        if text:
            painter.setFont(opt.font)
            bar_h = opt.fontMetrics.height() + 6
            bar = QRect(icon_rect.left(), icon_rect.top(),
                        icon_rect.width(), min(bar_h, icon_rect.height()))
            painter.fillRect(bar, self.BAR_COLOR)
            painter.setPen(Qt.white)
            elided = opt.fontMetrics.elidedText(text, Qt.ElideMiddle,
                                                bar.width() - 8)
            painter.drawText(bar.adjusted(4, 0, -4, 0),
                             Qt.AlignLeft | Qt.AlignVCenter, elided)
        painter.restore()


class ImageGrid(QListWidget):
    """缩略图网格;双击弹窗查看大图。"""

    middle_clicked = pyqtSignal(str)  # 中键点击图片 (携带文件路径)
    paths_dropped = pyqtSignal(list)  # 拖入图片路径 (set_accept_drops(True) 后生效)
    paths_dropped_on = pyqtSignal(list, str)  # 拖放到某张图上 (额外携带目标图路径)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setIconSize(QSize(THUMB_SIZE, int(0.67*THUMB_SIZE)))
        # 文件名叠加在图上(delegate 绘制), 单元格不再为文字预留高度
        self.setGridSize(QSize(THUMB_SIZE + 16, int(0.67*THUMB_SIZE + 16)))
        #self.setSpacing(8)
        self.setUniformItemSizes(True)
        # 滚轮按像素平滑滚动, 而不是整行跳动
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.horizontalScrollBar().setSingleStep(24)
        self.setMovement(QListWidget.Static)
        self.setSelectionMode(QListWidget.ExtendedSelection)  # 多选
        self.setDragEnabled(True)                             # 可拖出
        self.setDragDropMode(QListWidget.DragOnly)            # 不允许放回自身
        self.setItemDelegate(ThumbOverlayDelegate(self))
        self._loaders = []
        self._items = {}
        self._boxes = {}
        self._labels = {}
        self._loaded = set()    # 已加载缩略图的路径
        self._pending = set()   # 已派发给加载线程、尚未完成的路径
        self._cache_dir = None
        # 懒加载: 滚动/缩放后延迟 120ms 只加载可视区域的缩略图
        self._lazy_timer = QTimer(self)
        self._lazy_timer.setSingleShot(True)
        self._lazy_timer.setInterval(120)
        self._lazy_timer.timeout.connect(self._load_visible)
        # valueChanged(int) 的参数会传给 QTimer.start(msec), 必须吞掉
        self.verticalScrollBar().valueChanged.connect(
            lambda *_: self._lazy_timer.start())
        self.horizontalScrollBar().valueChanged.connect(
            lambda *_: self._lazy_timer.start())
        self.itemDoubleClicked.connect(self._open_full)
        self._accept_drops = False  # 默认不接受拖入 (DragOnly)

    def set_accept_drops(self, on):
        """允许/禁止从其它网格拖入图片路径 (自身拖出的不算)。"""
        self._accept_drops = on
        self.setDragDropMode(QListWidget.DragDrop if on
                             else QListWidget.DragOnly)

    def _drop_ok(self, event):
        return (self._accept_drops and event.source() is not self
                and event.mimeData().hasFormat(MIME_PATHS))

    def dragEnterEvent(self, event):
        if self._drop_ok(event):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._drop_ok(event):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self._drop_ok(event):
            event.ignore()
            return
        paths = bytes(event.mimeData().data(MIME_PATHS)).decode("utf-8").splitlines()
        paths = [p for p in paths if p]
        if paths:
            self.paths_dropped.emit(paths)
            item = self.itemAt(event.pos())
            if item is not None:
                self.paths_dropped_on.emit(paths, item.data(Qt.UserRole))
            event.setDropAction(Qt.MoveAction)
            event.accept()

    def mimeData(self, items):
        """拖拽时携带选中项的文件路径。"""
        md = QMimeData()
        md.setData(MIME_PATHS, "\n".join(
            it.data(Qt.UserRole) for it in items).encode("utf-8"))
        return md

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            item = self.itemAt(event.pos())
            if item is not None:
                self.middle_clicked.emit(item.data(Qt.UserRole))
                event.accept()
                return
        super().mousePressEvent(event)

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

    def show_images(self, paths, boxes=None, labels=None):
        paths = list(paths)  # 调用方可能传 pandas Series
        self._stop_loaders()
        self._boxes = boxes or {}
        self._labels = labels or {}
        self.clear()
        self._items = {}
        self._loaded = set()
        self._pending = set()
        for p in paths:
            # 文件名由 delegate 叠加在缩略图顶部(过长时中间省略), 无需换行处理
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            # 必须在创建时给定尺寸, 否则视图按纯文字布局后图标加载不重排, 图像被裁
            item.setSizeHint(QSize(THUMB_SIZE + 16, THUMB_SIZE + 16))
            self.addItem(item)
            self._items[p] = item
        if not paths:
            return
        # 磁盘缓存目录: 图片公共路径下的 .doris_thumbs (跨 tab 共享)
        try:
            self._cache_dir = os.path.join(os.path.commonpath(paths),
                                           ".doris_thumbs")
        except ValueError:
            self._cache_dir = None
        # 懒加载: 不立即全量加载, 等布局完成后只加载可视区域
        QTimer.singleShot(0, self._load_visible)

    def _stop_loaders(self):
        for loader in self._loaders:
            try:
                loader.loaded.disconnect(self._set_thumb)
            except TypeError:
                pass
            loader.stop()
        for loader in self._loaders:
            loader.wait()
        self._loaders = []

    def _visible_paths(self):
        """当前可视区域(上下各预取一屏)内、尚未加载的图片路径。"""
        rect = self.viewport().rect()
        rect.adjust(0, -rect.height(), 0, rect.height())
        out = []
        for i in range(self.count()):
            item = self.item(i)
            if self.visualItemRect(item).intersects(rect):
                p = item.data(Qt.UserRole)
                if p not in self._loaded:
                    out.append(p)
        return out

    def _load_visible(self):
        if not self.isVisible():
            return  # 未显示的 tab 不加载, showEvent 时再触发
        need = set(self._visible_paths())
        if not need:
            return
        if need <= self._pending and any(l.isRunning() for l in self._loaders):
            return  # 当前加载队列已覆盖所有可见项
        # 可见集合变了: 停掉旧队列, 只加载现在可见的
        self._stop_loaders()
        self._pending = set()
        paths = [p for p in self._visible_paths()]
        if not paths:
            return
        self._pending = set(paths)
        # 多线程分片加载 (大 JPEG 解码是瓶颈, 并行数倍提速)
        n_workers = min(8, (os.cpu_count() or 4), len(paths))
        for k in range(n_workers):
            loader = ThumbLoader(paths[k::n_workers], self._boxes,
                                 self._labels, self._cache_dir)
            loader.loaded.connect(self._set_thumb)
            loader.finished.connect(
                lambda l=loader: self._on_loader_done(l))
            loader.start()
            self._loaders.append(loader)

    def _on_loader_done(self, loader):
        if loader not in self._loaders:
            return  # 已被替换的旧加载器, 忽略
        for p in loader._paths:
            self._pending.discard(p)  # 含加载失败的, 下次滚动可重试

    def showEvent(self, event):
        super().showEvent(event)
        self._lazy_timer.start(0)  # tab 切到前台时补载可视区域

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._lazy_timer.start()

    def _set_thumb(self, path, img):
        self._pending.discard(path)
        self._loaded.add(path)
        item = self._items.get(path)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _open_full(self, item):
        paths = [self.item(i).data(Qt.UserRole) for i in range(self.count())]
        dlg = ImageViewerDialog(paths, self.row(item), self, boxes=self._boxes,
                                labels=self._labels)
        dlg.exec_()


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
        self.dolphin_list.currentRowChanged.connect(self._on_select)
        lay.addWidget(self.dolphin_list, 1)
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

    def zoom(self, factor):
        """以视图中心为锚点缩放 (键盘快捷键用; 滚轮仍以光标为中心)。"""
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        # ←/→ 交给对话框做翻页, 不做滚动
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)


class ImageViewerDialog(QDialog):
    """大图查看窗口。←/→ 切换组内上一张/下一张, =/- 放大/缩小, Ctrl+W 关闭。"""

    def __init__(self, paths, index, parent=None, boxes=None, labels=None):
        super().__init__(parent)
        self._paths = paths
        self._index = index
        self._boxes = boxes or {}
        self._labels = labels or {}
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
        #reader.setAutoTransform(True)  # 按 EXIF 方向旋转
        pm = QPixmap.fromImage(reader.read())
        boxes = self._boxes.get(path)
        if boxes and not pm.isNull():
            draw_boxes(pm, boxes, pm.width(), pm.height())
        label = self._labels.get(path)
        if label and not pm.isNull():
            draw_label(pm, label)
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
        elif event.key() in (Qt.Key_Equal, Qt.Key_Plus):
            self.view.zoom(1.25)          # =/+ 放大
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self.view.zoom(0.8)           # - 缩小
        elif event.key() == Qt.Key_W and \
                event.modifiers() & Qt.ControlModifier:
            self.close()                  # Ctrl+W 关闭
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
    def __init__(self, mover=None, full_resolver=None, full_boxes=None,
                 labels=None, alt_move=False, on_moved=None, best_mark=None,
                 similar_provider=None, track_mover=None, exclude=None,
                 overview_provider=None, merger=None, clear_mark=None,
                 parent=None):
        """mover: callable(paths, target_group) -> dict[old_path, new_path] 或 None
        full_resolver: callable(fin_path) -> 原始全图路径 或 None; 给定后中键鳍图可看原图
        full_boxes: 全图路径 -> 鳍框列表 (画在原图查看器上, 可选)
        labels: 图片路径 -> 叠加文字 (如画 shot_id, 可选)
        alt_move: True 时注册 Alt+1..9 快捷键, 把选中图片移入侧边第 N 组
        on_moved: callable(tab) — 改组成功后回调, 用于刷新其它 tab 的标注
        best_mark: callable(path) -> bool; 给定后顶栏出现 "Match Mode" 按钮,
        进入后只显示返回 True 的图 (FIN_METAINFO 的 BestInShot 列)
        clear_mark: callable(path) -> bool; 给定后顶栏出现 "Clear" 复选框,
        勾选后 Review 模式只显示 clear==1 的清晰鳍图 (FIN_METAINFO 的 clear 列)
        similar_provider: callable(group_name) -> (paths, labels) 或 None;
        给定后勾选最佳复选框时网格区分成三栏: 上=当前组, 左下=相似个体的鳍图
        (可拖入上方并入当前组), 右下=排除区
        track_mover: 勾选最佳复选框时侧边拖放改用的 mover (只改同 track 的鳍)
        exclude: (fn_current, fn_similar) — 三栏模式右下排除区的处理:
        fn_current(paths, group) 当前组的鳍拖入 (同 track 赋予新 FinID);
        fn_similar(paths, group) 左下的鳍拖入 (两 FinID 相似度置 0)
        overview_provider: callable() -> (paths, labels) 或 None; 给定后顶栏
        出现 "Overview" 按钮, 进入后网格显示每个个体的代表鳍图 (隐藏侧边栏)
        merger: callable(src_paths, target_path) 或 None; 给定后 Overview 模式
        下可把代表鳍图拖放到另一张上, 合并两个 FinID (含对应 DolphinID)"""
        super().__init__(parent)
        self._mover = mover
        self._track_mover = track_mover
        self._exclude = exclude
        self._on_moved = on_moved
        self._full_resolver = full_resolver
        self._full_boxes = full_boxes or {}
        self._labels = labels or {}
        self._alt_move = alt_move and mover is not None
        self._best_mark = best_mark
        self._best_only = False
        self._clear_mark = clear_mark
        self._clear_only = False
        self._similar_provider = similar_provider
        self._overview_provider = overview_provider
        self._overview = False      # 总览模式
        self._merger = merger       # Overview 模式拖放合并 FinID
        self._tri = False           # 三栏模式
        self.vsplit = None
        self.similar_grid = None
        self.excluded_grid = None
        self._excluded = []         # [(参考组名, 鳍图路径)] 拖入排除区的记录
        self.splitter = QSplitter(Qt.Horizontal)
        self.group_list = GroupListWidget()
        self.group_list.setMinimumWidth(60)   # 保证初始宽度, 否则 splitter 按空列表 sizeHint 给得很窄
        self.group_list.setMaximumWidth(120)
        self.group_list.setWordWrap(False)
        self.group_list.setTextElideMode(Qt.ElideNone)  # 不省略
        self.grid = ImageGrid()
        self.splitter.addWidget(self.group_list)
        self.splitter.addWidget(self.grid)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([160, 1000])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if self._best_mark is not None or self._overview_provider is not None \
                or self._clear_mark is not None:
            bar = QWidget()
            bar.setFixedHeight(26)  # 顶栏限高, 默认会被按钮撑得很高
            top = QHBoxLayout(bar)
            top.setContentsMargins(4, 0, 4, 0)
            # 模式选择: 单个 "Mode" 图标按钮 + 弹出菜单 (Review 默认 / Match / Overview)
            # 仅当有其它模式可选时才出现 (Dolphin 审阅页只有 Review, Mode 按钮
            # 由 Dolphin 页顶栏提供, 这里不再重复)
            if self._best_mark is not None or self._overview_provider is not None:
                modes = [("review", "Review Mode",
                          "浏览模式: 侧边分组列表 + 当前组全部鳍图")]
                if self._best_mark is not None:
                    modes.append(("match", "Match Mode",
                                  "匹配模式: 只显示每 shot 最佳鳍图 (BestInShot), "
                                  "网格分三栏: 当前组 / 相似个体 / 排除区"))
                if self._overview_provider is not None:
                    modes.append(("overview", "Overview Mode",
                                  "总览模式: 每个个体一张最清晰的代表鳍图"))
                self._mode_actions = {}
                menu = QMenu(self)
                for key, text, tip in modes:
                    act = menu.addAction(text)
                    act.setCheckable(True)
                    act.setToolTip(tip)
                    act.triggered.connect(
                        lambda _=False, k=key: self._set_mode(k))
                    self._mode_actions[key] = act
                self._mode_actions["review"].setChecked(True)
                self.mode_btn = QToolButton()
                self.mode_btn.setText("Mode: Review Mode")
                self.mode_btn.setIcon(self._mode_icon())
                self.mode_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                self.mode_btn.setPopupMode(QToolButton.InstantPopup)
                self.mode_btn.setToolTip("选择浏览模式")
                self.mode_btn.setMenu(menu)
                top.addWidget(self.mode_btn)
                self.count_label = QLabel()
                top.addWidget(self.count_label)
            self._clear_cb = None
            if self._clear_mark is not None:
                self._clear_cb = QCheckBox("Clear")
                self._clear_cb.setToolTip("勾选后 Review 模式只显示 "
                                          "clear==1 的清晰鳍图")
                self._clear_cb.toggled.connect(self._on_clear_toggled)
                top.addWidget(self._clear_cb)
            top.addStretch(1)
            lay.addWidget(bar)
        lay.addWidget(self.splitter)
        self._groups = {}
        self._tooltips = {}
        self._boxes = {}
        self.group_list.currentTextChanged.connect(self._on_group)
        self.group_list.paths_dropped.connect(self._on_drop)
        # 无 mover 的页不接受拖放
        self.group_list.setAcceptDrops(mover is not None)
        self.grid.setDragEnabled(mover is not None)
        self.grid.middle_clicked.connect(self._on_middle)
        self.grid.paths_dropped_on.connect(self._on_merge_drop)
        if self._alt_move:
            for i in range(1, 10):
                sc = QShortcut(QKeySequence("Alt+%d" % i), self)
                sc.setContext(Qt.WidgetWithChildrenShortcut)
                sc.activated.connect(
                    lambda i=i: self._move_to_group_index(i - 1))

    def _move_to_group_index(self, idx):
        """Alt+N: 把网格中选中的图片移入侧边第 N 组 (与拖放同一条路径)。"""
        if idx >= self.group_list.count():
            return
        items = self.grid.selectedItems()
        if not items:
            return
        target = self.group_list.item(idx).data(Qt.UserRole)
        paths = [it.data(Qt.UserRole) for it in items]
        self._on_drop(paths, target)

    def _on_middle(self, path):
        """中键鳍图: 弹窗查看对应原始全图, 可翻页浏览当前组内其它原图。"""
        if self._full_resolver is None:
            return
        orig = self._full_resolver(path)
        if orig is None:
            QMessageBox.information(self, "Doris", "找不到对应的原图: %s" % path)
            return
        cur = self.group_list.currentItem()
        group_paths = self._groups.get(
            cur.data(Qt.UserRole), [path]) if cur else [path]
        origs = []
        for p in group_paths:
            o = self._full_resolver(p)
            if o and o not in origs:
                origs.append(o)
        if orig not in origs:
            # 三栏模式下中键来自相似区/排除区, 图不在当前组: 插到最前
            origs.insert(0, orig)
        dlg = ImageViewerDialog(origs, origs.index(orig), self,
                                boxes=self._full_boxes)
        dlg.exec_()

    def set_groups(self, groups, tooltips=None, keep_current=False, boxes=None):
        """groups: dict[str, list[str]] — 组名 -> 图片路径列表 (有序)
        boxes: dict[str, list] — 图片路径 -> 鳍框列表 (可选, 画在缩略图和大图上)"""
        cur = self.group_list.currentItem()
        cur_name = cur.data(Qt.UserRole) if (keep_current and cur) else None
        self._groups = groups
        self._tooltips = tooltips or {}
        self._boxes = boxes or {}
        self.group_list.clear()
        for i, name in enumerate(groups, 1):
            display = "%s  (%d)" % (name, len(groups[name]))
            if self._alt_move and i <= 9:  # 标注 Alt+N 快捷键
                display = "%d. %s" % (i, display)
            item = QListWidgetItem(display, self.group_list)
            item.setData(Qt.UserRole, name)
            item.setToolTip(self._tooltips.get(name, name))
            # 给足行高; 宽度不强行撑大, 让侧边栏保持紧凑
            item.setSizeHint(QSize(160, 28))
        if getattr(self, "count_label", None) is not None:
            n = sum(1 for name in groups if name != "FinID_0")
            self.count_label.setText("FinID 总数: %d" % n)
        if not groups:
            return
        # FinID_0 不是真实个体, 默认选中其后第一个组 (如 FinID_1)
        if cur_name in groups:
            self.group_list.setCurrentRow(list(groups).index(cur_name))
        else:
            names = list(groups)
            row = 1 if len(names) > 1 and names[0] == "FinID_0" else 0
            self.group_list.setCurrentRow(row)

    @staticmethod
    def _mode_icon():
        """"Mode" 按钮图标: 三行滑块样式。"""
        pm = QPixmap(16, 16)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(90, 90, 90))
        for i in range(3):
            p.drawRect(1, 2 + i * 5, 14, 3)
        p.end()
        return QIcon(pm)

    def _set_mode(self, mode):
        """切换顶栏模式 (互斥): review=分组浏览 (默认), match=只显示每 shot
        最佳鳍图并进入三栏匹配, overview=每个个体一张代表图 (隐藏侧边栏,
        有 merger 时可拖放合并 FinID)。"""
        for k, act in self._mode_actions.items():
            act.setChecked(k == mode)  # QAction 无 toggled 触发合并逻辑, 无需 block
        if hasattr(self, "mode_btn"):
            self.mode_btn.setText("Mode: %s"
                                  % self._mode_actions[mode].text())
        self._overview = (mode == "overview")
        self._best_only = (mode == "match")
        if self._clear_cb is not None:
            self._clear_cb.setEnabled(mode == "review")  # 筛选只在 Review 模式生效
        self.group_list.setVisible(not self._overview)
        if self._best_only and self._similar_provider is not None:
            self._enter_tri_mode()
        else:
            self._exit_tri_mode()
        if self._overview and self._merger is not None:
            # 拖放合并: 可拖出且接受落到另一张图上
            self.grid.setDragEnabled(True)
            self.grid.set_accept_drops(True)
        elif not self._tri:  # 三栏模式自己管理拖放
            self.grid.setDragEnabled(self._mover is not None)
            self.grid.set_accept_drops(False)  # 退出 overview 合并拖放
        self._on_group(None)  # 重绘网格

    def _on_merge_drop(self, paths, target_path):
        """Overview 模式: 把被拖代表图所在 FinID 并入目标图所在 FinID。"""
        if not self._overview or self._merger is None:
            return
        paths = [p for p in paths if p != target_path]  # 拖到自身不算
        if not paths:
            return
        try:
            self._merger(paths, target_path)
        except Exception as e:
            QMessageBox.warning(self, "合并失败", str(e))
            return
        if self._on_moved is not None:
            self._on_moved(self)  # 元数据已变, 刷新所有 tab (本页总览随 _on_group 重绘)

    @staticmethod
    def _titled(title, w):
        """给子窗口加一行标题。"""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(2, 2, 2, 0)
        lay.addWidget(QLabel(title))
        lay.addWidget(w)
        return box

    def _enter_tri_mode(self):
        """网格区分成三栏: 上=当前组(接受拖入), 左下=相似鳍图, 右下=排除区。"""
        if self._tri:
            return
        self._tri = True
        self.grid.set_accept_drops(True)
        self.grid.paths_dropped.connect(self._on_top_drop)
        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.addWidget(self.grid)  # 从主 splitter 移入上半部
        bottom = QSplitter(Qt.Horizontal)
        self.similar_grid = ImageGrid()
        self.similar_grid.setDragEnabled(True)   # 可拖出到上方/排除区
        self.similar_grid.middle_clicked.connect(self._on_middle)
        bottom.addWidget(self._titled("相似个体 (拖到上方并入当前组)",
                                      self.similar_grid))
        self.excluded_grid = ImageGrid()
        self.excluded_grid.setDragEnabled(True)   # 可拖回上方 (改回当前 FinID)
        self.excluded_grid.set_accept_drops(True)
        self.excluded_grid.paths_dropped.connect(self._on_exclude_drop)
        self.excluded_grid.middle_clicked.connect(self._on_middle)
        bottom.addWidget(self._titled("排除区 (拖入 = 判定为不同个体)",
                                      self.excluded_grid))
        bottom.setSizes([600, 600])
        self.vsplit.addWidget(bottom)
        self.vsplit.setSizes([500, 400])
        self.splitter.addWidget(self.vsplit)

    def _exit_tri_mode(self):
        if not self._tri:
            return
        self._tri = False
        self.grid.set_accept_drops(False)
        self.grid.paths_dropped.disconnect(self._on_top_drop)
        for g in self.vsplit.findChildren(ImageGrid):
            g._stop_loaders()  # 先停加载线程再销毁
        self.splitter.addWidget(self.grid)  # 移回主 splitter
        self.vsplit.setParent(None)
        self.vsplit.deleteLater()
        self.vsplit = None
        self.similar_grid = None
        self.excluded_grid = None

    def _refresh_excluded(self):
        """排除区只显示当前参考组的排除记录, 随组切换刷新。"""
        if self.excluded_grid is None:
            return
        cur = self.group_list.currentItem()
        g = cur.data(Qt.UserRole) if cur else None
        paths = [p for gp, p in self._excluded if gp == g]
        self.excluded_grid.show_images(paths, labels=self._labels)

    def _on_top_drop(self, paths):
        """拖入上半部: 来自排除区 -> 同 track 的鳍改回当前 FinID (track mover);
        来自左下相似区 -> 整个体并入当前组 (普通 mover)。"""
        cur = self.group_list.currentItem()
        if cur is None:
            return
        gname = cur.data(Qt.UserRole)
        excl = {p for gp, p in self._excluded if gp == gname}
        from_excl = [p for p in paths if p in excl]
        other = [p for p in paths if p not in excl]
        if from_excl:
            self._on_drop(from_excl, gname, use_track=True)
            self._excluded = [(gp, p) for gp, p in self._excluded
                              if not (gp == gname and p in from_excl)]
            self._refresh_excluded()
        if other:
            self._on_drop(other, gname, use_track=False)

    def _on_exclude_drop(self, paths):
        """拖入右下排除区:
        来自上方(当前组) -> 同 track 的鳍赋予新 FinID, 新旧 FinID 相似度置 0;
        来自左下(其它个体) -> 被拖鳍的 FinID 与当前 FinID 相似度置 0。"""
        if self._exclude is None:
            return
        cur = self.group_list.currentItem()
        if cur is None:
            return
        gname = cur.data(Qt.UserRole)
        cur_paths = set(self._groups.get(gname, []))
        from_cur = [p for p in paths if p in cur_paths]
        from_sim = [p for p in paths if p not in cur_paths]
        fn_cur, fn_sim = self._exclude
        moved = {}
        try:
            if from_cur:
                moved = fn_cur(from_cur, gname) or {}
            if from_sim:
                fn_sim(from_sim, gname)
        except Exception as e:
            QMessageBox.warning(self, "排除失败", str(e))
            return
        for p in paths:  # 排除区按参考组累积记录
            if (gname, p) not in self._excluded:
                self._excluded.append((gname, p))
        self._refresh_excluded()
        if moved:  # FinID 变了: 从旧组移除并刷新全部 tab
            for old in moved:
                for g in self._groups.values():
                    if old in g:
                        g.remove(old)
            self.set_groups(self._groups, self._tooltips, keep_current=True,
                            boxes=self._boxes)
            if self._on_moved is not None:
                self._on_moved(self)
        else:  # 只改了相似度: 重绘让左下被排除的对消失
            self._on_group(None)

    def _filter_best(self, paths):
        """只保留 BestInShot 标记的图; 无标记信息的图也保留。"""
        return [p for p in paths if self._best_mark(p) is not False]

    def _on_clear_toggled(self, checked):
        """"Clear" 复选框: 勾选后 Review 模式只显示 clear==1 的清晰鳍图。"""
        self._clear_only = checked
        self._on_group(None)

    def _on_group(self, text):
        if self._overview and self._overview_provider is not None:
            paths, labels = self._overview_provider()
            self.grid.show_images(paths, labels=labels)
            return
        item = self.group_list.currentItem()
        if item is None:
            return
        paths = self._groups.get(item.data(Qt.UserRole), [])
        if self._best_only and self._best_mark is not None:
            paths = self._filter_best(paths)
        elif self._clear_only and self._clear_mark is not None:
            paths = [p for p in paths if self._clear_mark(p)]
        self.grid.show_images(paths,
                              boxes=self._boxes, labels=self._labels)
        if self._tri and self.similar_grid is not None:
            sim_paths, sim_labels = self._similar_provider(
                item.data(Qt.UserRole))
            labels = dict(self._labels)
            labels.update(sim_labels)  # 相似度标签优先
            self.similar_grid.show_images(sim_paths, labels=labels)
            self._refresh_excluded()  # 排除区跟随当前参考组

    def _on_drop(self, paths, target_name, use_track=None):
        """use_track=None 时跟随最佳复选框: 勾选后侧边拖放只改同 track 的鳍。"""
        if use_track is None:
            use_track = self._best_only
        mover = self._track_mover if (use_track and self._track_mover) \
            else self._mover
        if mover is None or target_name not in self._groups:
            return
        try:
            moved = mover(paths, target_name)
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
        if self._on_moved is not None:
            self._on_moved(self)  # 元数据已变, 刷新所有 tab 的分组和标注


# ---------------------------------------------------------------------------
# 统计页
# ---------------------------------------------------------------------------
class StatisticsTab(QWidget):
    """统计页: 每个 FinID 鳍图数柱状图 + 每个 FinID shot 数柱状图 + 部位占比饼图。
    数据经 show_stats(fin_counts, shot_counts, aspect_counts) 传入,
    元素为 (label, count)。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        lay = QHBoxLayout(self)
        hists = QVBoxLayout()
        self.fig_hist = Figure()
        self.fig_shot = Figure()
        self.fig_pie = Figure()
        hists.addWidget(FigureCanvasQTAgg(self.fig_hist))
        hists.addWidget(FigureCanvasQTAgg(self.fig_shot))
        lay.addLayout(hists, 2)
        lay.addWidget(FigureCanvasQTAgg(self.fig_pie), 1)

    def _bar(self, fig, counts, title):
        fig.clear()
        ax = fig.add_subplot(111)
        if counts:
            labels = [k for k, _ in counts]
            values = [v for _, v in counts]
            ax.bar(range(len(values)), values, color="#4C72B0")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            for i, v in enumerate(values):
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=5)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def show_stats(self, fin_counts, shot_counts, aspect_counts):
        self._bar(self.fig_hist, fin_counts, "Fin images per FinID")
        self._bar(self.fig_shot, shot_counts, "Shots per FinID")

        fig = self.fig_pie
        fig.clear()
        ax = fig.add_subplot(111)
        if aspect_counts:
            labels = [k for k, _ in aspect_counts]
            values = [v for _, v in aspect_counts]
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": 9})
        ax.set_title("Fin aspect distribution")
        fig.tight_layout()
        fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.setWindowTitle("Doris v0.2 — %s" % root)
        self.resize(1280, 800)

        toolbar = QToolBar()
        act = QAction("Open folder…", self)
        act.triggered.connect(self._pick_folder)
        toolbar.addAction(act)
        self.addToolBar(toolbar)

        self.tabs = QTabWidget()
        # 切到统计页时惰性重算, 保证任何改组后看到的都是最新数据
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self.load(root)

    def _on_tab_changed(self, i):
        if self._stats_tab is not None \
                and self.tabs.widget(i) is self._stats_tab:
            self._refresh_stats()

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择 PHOTO 根目录")
        if d:
            self.load(d)

    def closeEvent(self, event):
        # 退出前停掉所有缩略图加载线程, 否则线程在解释器退出时析构会段错误
        # findChildren 覆盖所有网格 (含 Fin ID 三栏模式的网格和 Dolphin 页)
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            grids = ([w] if isinstance(w, DolphinTab) else []) \
                + w.findChildren(ImageGrid) + w.findChildren(DolphinTab) \
                + w.findChildren(CandidateList)
            for g in grids:
                for loader in getattr(g, "_loaders", []):
                    loader.stop()
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            grids = ([w] if isinstance(w, DolphinTab) else []) \
                + w.findChildren(ImageGrid) + w.findChildren(DolphinTab) \
                + w.findChildren(CandidateList)
            for g in grids:
                for loader in getattr(g, "_loaders", []):
                    loader.wait(2000)
        super().closeEvent(event)

    def load(self, root):
        self.root = root
        ds = Dataset(root)
        self.ds = ds
        self._group_tabs = []
        self._dolphin_tab = None
        self._stats_tab = None
        self.tabs.clear()

        # 1-4. 分组页 (支持拖拽改组: 只更新 dataframe 并写回 CSV, 不动文件)
        shot_boxes = ds.fin_boxes()  # Continuous Shots 页画鳍框
        fin_tabs = {"Fin Aspect", "Blur", "Fin ID"}  # 鳍图页: 中键查看原图, 叠加 shot_id
        alt_tabs = {"Fin Aspect", "Blur"}            # 支持 Alt+N 移入第 N 组
        shot_labels = ds.fin_shot_labels()  # 鳍图路径 -> shot_id 叠加文字
        best_marks = ds.fin_best_marks()    # Fin ID 页 "每 shot 最佳" 筛选
        clear_marks = ds.fin_clear_marks()  # Fin ID / Dolphin 页 "Clear" 筛选
        for title, fn, mover in [
            ("Continuous Shots", ds.shots, ds.move_shot),
            ("Fin Aspect", ds.aspects, ds.move_aspect),
            ("Blur", ds.blur, ds.move_blur),
            ("Fin ID", ds.fin_ids, ds.move_to_fin_id),
        ]:
            tab = GroupedTab(
                mover=mover,
                full_resolver=ds.orig_of_fin if title in fin_tabs else None,
                full_boxes=shot_boxes if title in fin_tabs else None,
                labels=shot_labels if title in fin_tabs else None,
                alt_move=title in alt_tabs,
                on_moved=self._refresh_tabs,
                best_mark=best_marks.__contains__ if title == "Fin ID" else None,
                clear_mark=clear_marks.__contains__ if title == "Fin ID" else None,
                similar_provider=ds.similar_fins if title == "Fin ID" else None,
                track_mover=ds.move_track_to_fin_id if title == "Fin ID" else None,
                exclude=(ds.exclude_to_new_fin_id, ds.exclude_fin_id_pair)
                if title == "Fin ID" else None,
                overview_provider=ds.fin_id_previews
                if title == "Fin ID" else None,
                merger=ds.merge_fin_paths if title == "Fin ID" else None)
            groups = fn()
            if title == "Fin ID":  # 只显示图片数大于 1 的个体
                groups = {k: v for k, v in groups.items() if len(v) > 0}
            if groups:
                boxes = shot_boxes if title == "Continuous Shots" else None
                tab.set_groups(groups, boxes=boxes)
            else:
                tab.grid.show_images([])
                QListWidgetItem("(no data)", tab.group_list)
            self.tabs.addTab(tab, title)
            self._group_tabs.append((title, tab, fn))

        # 5. 海豚配对页: 每只海豚一格, 上 DL 下 DR, DR 格可拖拽交换;
        #    "新建海豚"按钮创建空海豚供手工配对; 顶栏 "Mode" 图标按钮弹出菜单
        #    选择三个互斥模式: Overview=海豚格 (DR 可拖拽交换, 默认) /
        #    Match=匹配面板 / Review=左侧 DolphinID 列表 + 右侧全部鳍图
        tab9 = QWidget()
        lay = QVBoxLayout(tab9)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self._btn_new = QPushButton("新建海豚")
        self._btn_new.setToolTip("创建一只空海豚 (两格皆空), 可把其它海豚的 DR 拖入")
        self._btn_new.clicked.connect(self._new_dolphin)
        top.addWidget(self._btn_new)
        self._dolphin_mode_actions = {}
        menu = QMenu(self)
        for key, text, tip in [
                ("review", "Review Mode",
                 "审阅模式: 左侧 DolphinID 列表, 右侧显示选中海豚的全部鳍图"),
                ("match", "Match Mode",
                 "匹配模式: 核对每只海豚的 DL/DR 配对;\n"
                 "候选鳍拖入鳍格 = 确认配对, 鳍图拖出格子 = 确认不匹配"),
                ("overview", "Overview Mode",
                 "总览模式: 每只海豚一格 (上 DL 下 DR), DR 格可拖拽交换")]:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _=False, k=key: self._set_dolphin_mode(k))
            self._dolphin_mode_actions[key] = act
        self._dolphin_mode_actions["overview"].setChecked(True)
        self._dolphin_mode_btn = QToolButton()
        self._dolphin_mode_btn.setText("Mode: Overview Mode")
        self._dolphin_mode_btn.setIcon(GroupedTab._mode_icon())
        self._dolphin_mode_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._dolphin_mode_btn.setPopupMode(QToolButton.InstantPopup)
        self._dolphin_mode_btn.setMenu(menu)
        top.addWidget(self._dolphin_mode_btn)
        top.addStretch(1)
        lay.addLayout(top)
        self._dolphin_stack = QStackedWidget()
        self._dolphin_tab = DolphinTab()
        self._dolphin_tab.show_dolphins(ds.dolphins())
        self._dolphin_tab.dr_swapped.connect(self._swap_dolphin_dr)
        self._match_panel = MatchPanel(ds.dolphin_candidates)
        self._match_panel.assign_requested.connect(self._match_assign)
        self._match_panel.unmatch_requested.connect(self._match_unmatch)
        self._dolphin_stack.addWidget(self._dolphin_tab)
        self._dolphin_stack.addWidget(self._match_panel)
        # Review 模式: 左侧 DolphinID 列表, 右侧该海豚的全部鳍图 (只读)
        self._dolphin_review_tab = GroupedTab(
            mover=None, full_resolver=ds.orig_of_fin,
            labels=ds.dolphin_fin_labels(),
            clear_mark=ds.fin_clear_marks().__contains__)
        groups = ds.dolphin_groups()
        if groups:
            self._dolphin_review_tab.set_groups(groups)
        else:
            self._dolphin_review_tab.grid.show_images([])
            QListWidgetItem("(no data)", self._dolphin_review_tab.group_list)
        self._dolphin_stack.addWidget(self._dolphin_review_tab)
        lay.addWidget(self._dolphin_stack)
        self.tabs.addTab(tab9, "Dolphin")

        # 6. 社会结构页 (只读): 同一连拍出现多于一个个体 (不含 FinID_0) 的
        #    shot 各成一组 (NN001...), 组内只收同一张原图含多个个体鳍的图,
        #    组名写回 IMAGE/FIN_METAINFO.csv 的 SocialStructure 列;
        #    仿 Continuous Shots 页在原图上画鳍框, 标注 FinID 和 DolphinID
        #    (须排在 Dolphin 页后, ds.dolphins() 已分配 DolphinID 列)
        tab_social = GroupedTab(mover=None, on_moved=self._refresh_tabs)
        groups = ds.social()
        if groups:
            tab_social.set_groups(groups, ds.social_tooltips,
                                  boxes=ds.fin_boxes(dolphin_id=True))
        else:
            tab_social.grid.show_images([])
            QListWidgetItem("(no data)", tab_social.group_list)
        self.tabs.addTab(tab_social, "Social Structure")
        self._group_tabs.append(("Social Structure", tab_social, ds.social))

        # 7. 统计页: FinID 鳍图数柱状图 + 部位占比饼图
        self._stats_tab = StatisticsTab()
        self._stats_tab.show_stats(ds.fin_id_counts(), ds.fin_id_shot_counts(),
                                   ds.aspect_counts())
        self.tabs.addTab(self._stats_tab, "Statistics")


    def _set_dolphin_mode(self, mode):
        """切换 Dolphin 页模式 (互斥): overview=海豚格 (DR 可拖拽交换, 默认),
        match=匹配面板, review=左侧 DolphinID 列表 + 右侧该海豚全部鳍图。"""
        for k, act in self._dolphin_mode_actions.items():
            act.setChecked(k == mode)
        self._dolphin_mode_btn.setText(
            "Mode: %s" % self._dolphin_mode_actions[mode].text())
        self._btn_new.setVisible(mode != "review")  # 审阅列表与新建无关
        if mode == "review":
            self._dolphin_review_tab.set_groups(self.ds.dolphin_groups(),
                                                keep_current=True)
            self._dolphin_stack.setCurrentWidget(self._dolphin_review_tab)
        elif mode == "match":
            self._match_panel.show_dolphins(self.ds.dolphins())
            self._dolphin_stack.setCurrentWidget(self._match_panel)
        else:  # overview
            self._dolphin_tab.show_dolphins(self.ds.dolphins())
            self._dolphin_stack.setCurrentWidget(self._dolphin_tab)

    def _match_assign(self, name, cls, fid):
        """Match Mode: 候选鳍拖入鳍格, 确认配对后刷新并保持当前海豚。"""
        self.ds.assign_dolphin_fin(name, fid, cls)
        self._match_panel.show_dolphins(self.ds.dolphins(), keep=name)
        self._refresh_stats()

    def _match_unmatch(self, cls, fid):
        """Match Mode: 鳍图拖出格子, 确认不匹配后刷新。"""
        self.ds.unassign_dolphin_fin(fid, cls)
        self._match_panel.show_dolphins(
            self.ds.dolphins(), keep=self._match_panel.current_name())
        self._refresh_stats()

    def _new_dolphin(self):
        """新建空海豚并刷新 Dolphin 页。"""
        self.ds.new_dolphin()
        self._dolphin_tab.show_dolphins(self.ds.dolphins())

    def _swap_dolphin_dr(self, src, dst):
        """Dolphin 页拖 DR 格到其它海豚: 交换/移动 DR 配对后刷新本页。"""
        self.ds.swap_dolphin_dr(src, dst)
        self._dolphin_tab.show_dolphins(self.ds.dolphins())
        self._refresh_stats()

    def _refresh_stats(self):
        if self._stats_tab is not None:
            self._stats_tab.show_stats(
                self.ds.fin_id_counts(), self.ds.fin_id_shot_counts(),
                self.ds.aspect_counts())

    def _refresh_tabs(self, origin=None):
        """元数据被拖拽改组更新后, 重建所有分组页的分组和标注 (含发起页)。"""
        ds = self.ds
        fin_tabs = {"Fin Aspect", "Blur", "Fin ID"}
        shot_boxes = ds.fin_boxes()
        shot_labels = ds.fin_shot_labels()
        for title, tab, fn in self._group_tabs:
            groups = fn()
            if title == "Fin ID":  # 只显示图片数大于 1 的个体
                groups = {k: v for k, v in groups.items() if len(v) > 1}
                tab._best_mark = ds.fin_best_marks().__contains__  # 标记可能已变
                tab._clear_mark = ds.fin_clear_marks().__contains__
            if not groups:
                continue
            tooltips = getattr(ds, "social_tooltips", None) \
                if title == "Social Structure" else None
            tab._labels = shot_labels if title in fin_tabs else {}
            tab._full_boxes = shot_boxes if title in fin_tabs else {}
            if title == "Continuous Shots":
                boxes = shot_boxes
            elif title == "Social Structure":
                # FinID/DolphinID 可能已变 (须先经 ds.dolphins() 分配 DolphinID)
                ds.dolphins()
                boxes = ds.fin_boxes(dolphin_id=True)
            else:
                boxes = None
            tab.set_groups(groups, tooltips, keep_current=True, boxes=boxes)
            tab._on_group(None)  # 当前组名没变时信号不发, 强制用新标注重绘网格
        if getattr(self, "_dolphin_review_tab", None) is not None:
            # Dolphin 审阅页 (配对/FinID 可能已变)
            groups = self.ds.dolphin_groups()
            if groups:
                rt = self._dolphin_review_tab
                rt._labels = self.ds.dolphin_fin_labels()
                rt._clear_mark = self.ds.fin_clear_marks().__contains__
                rt.set_groups(groups, keep_current=True)
                rt._on_group(None)  # 当前组名没变时信号不发, 强制重绘
        if self._dolphin_tab is not None:  # 配对页同步 (FinID 可能增删)
            self._dolphin_tab.show_dolphins(ds.dolphins())
        if self._stats_tab is not None:  # 统计页同步
            self._refresh_stats()


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
