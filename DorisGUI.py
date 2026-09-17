#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DorisGUI — 中华白海豚照片浏览与分组 GUI

6 个标签页:
  1. 单张全图浏览 (Full Images)
  2. 按连拍分组 (Continuous Shots, IMAGE_METAINFO.csv 的 shot_id)
  3. 按鳍部位分组 (Fin Aspect: DL=Left / DR=Right / Others=Tail, Head, lateral Fin / Wrong=0 )
  4. 按模糊与否分组 (Blur / Clear / Mid, 依据 FIN_METAINFO.csv 的 clear 列)
  5. 按个体编号分组 (Fin ID, FIN_METAINFO.csv 的 FinID 列)
  6. 按社会结构分组 (Social Structure, 同一连拍中个体共现的连通分量)

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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QDialog,
    QScrollArea, QFileDialog, QToolBar, QAction, QMessageBox,
    QGraphicsView, QGraphicsScene, QStyledItemDelegate, QStyle,
    QStyleOptionViewItem, QShortcut,
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QImageReader, QImageIOHandler, QPainter, QBrush, QColor, QDrag, QPen, QPalette, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPoint, QRect
import PyQt5

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
MIME_PATHS = "application/x-doris-paths"

ASPECT_LABELS = {"DL": "Left (DL)", "DR": "Right (DR)", "Others": "Tail/Head (Others)", "ND":"Not Dolphin"}

BOX_COLORS = {"DL": QColor(0, 200, 0), "DR": QColor(255, 140, 0),
              "Others": QColor(220, 0, 0), "ND": QColor(128, 128, 128)}


def draw_boxes(target, boxes, orig_w, orig_h):
    """在目标 QImage/QPixmap 上画鳍框 (orig_w/h 为框坐标对应的图像尺寸)。
    boxes: [(x0, y0, x1, y1, cls[, extra]), ...] 图像像素坐标;
    extra (如 FinID) 非空时追加在类别文字后。"""
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
        color = BOX_COLORS.get(cls, Qt.yellow)
        p.setPen(QPen(color, pen_w))
        p.drawRect(rect)
        text = cls
        if len(box) > 5 and box[5]:
            text = "%s %s" % (cls, box[5])
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

    def __init__(self, paths, boxes=None, labels=None, cache_dir=None):
        super().__init__()
        self._paths = list(paths)
        self._boxes = boxes or {}
        self._labels = labels or {}
        self._cache_dir = cache_dir
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
                    if not img.isNull():
                        return img, ow, oh
            except OSError:
                pass
        reader = QImageReader(p)
        #reader.setAutoTransform(True)  # 按 EXIF 方向旋转
        trans = reader.transformation()
        stored_w, stored_h = reader.size().width(), reader.size().height()
        sz = QSize(reader.size())
        if trans in (QImageIOHandler.TransformationRotate90,
                     QImageIOHandler.TransformationRotate270,
                     QImageIOHandler.TransformationFlipAndRotate90,
                     QImageIOHandler.TransformationMirrorAndRotate90):
            sz.transpose()
        # setScaledSize 不保持宽高比, 需按原始尺寸(含 EXIF 旋转)手动算目标尺寸
        sz.scale(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio)
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
        self._loader = None
        self._loaders = []
        self._items = {}
        self._boxes = {}
        self._labels = {}
        self.itemDoubleClicked.connect(self._open_full)

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
        for loader in self._loaders:
            try:
                loader.loaded.disconnect(self._set_thumb)
            except TypeError:
                pass
            loader.stop()
        for loader in self._loaders:
            loader.wait()
        self._loaders = []
        self._boxes = boxes or {}
        self._labels = labels or {}
        self.clear()
        self._items = {}
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
            cache_dir = os.path.join(os.path.commonpath(paths), ".doris_thumbs")
        except ValueError:
            cache_dir = None
        # 多线程分片加载 (大 JPEG 解码是瓶颈, 并行数倍提速)
        n_workers = min(8, (os.cpu_count() or 4), len(paths))
        for k in range(n_workers):
            loader = ThumbLoader(paths[k::n_workers], self._boxes,
                                 self._labels, cache_dir)
            loader.loaded.connect(self._set_thumb)
            loader.start()
            self._loaders.append(loader)

    def _set_thumb(self, path, img):
        item = self._items.get(path)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _open_full(self, item):
        paths = [self.item(i).data(Qt.UserRole) for i in range(self.count())]
        dlg = ImageViewerDialog(paths, self.row(item), self, boxes=self._boxes,
                                labels=self._labels)
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
    """大图查看窗口。←/→ 切换组内上一张/下一张, =/- 放大/缩小。"""

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
                 labels=None, alt_move=False, on_moved=None, parent=None):
        """mover: callable(paths, target_group) -> dict[old_path, new_path] 或 None
        full_resolver: callable(fin_path) -> 原始全图路径 或 None; 给定后中键鳍图可看原图
        full_boxes: 全图路径 -> 鳍框列表 (画在原图查看器上, 可选)
        labels: 图片路径 -> 叠加文字 (如画 shot_id, 可选)
        alt_move: True 时注册 Alt+1..9 快捷键, 把选中图片移入侧边第 N 组
        on_moved: callable(tab) — 改组成功后回调, 用于刷新其它 tab 的标注"""
        super().__init__(parent)
        self._mover = mover
        self._on_moved = on_moved
        self._full_resolver = full_resolver
        self._full_boxes = full_boxes or {}
        self._labels = labels or {}
        self._alt_move = alt_move and mover is not None
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
        self.grid.middle_clicked.connect(self._on_middle)
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
                              boxes=self._boxes, labels=self._labels)

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
        if self._on_moved is not None:
            self._on_moved(self)  # 元数据已变, 刷新所有 tab 的分组和标注


# ---------------------------------------------------------------------------
# 数据加载 (pandas)
# ---------------------------------------------------------------------------
class Dataset:
    """从 root 目录加载全部元数据。"""

    def __init__(self, root):
        self.root = root
        self.meta_dir = os.path.join(self.root, "METAINFO")
        self.image_df = self._load_meta("IMAGE_METAINFO.csv")
        self.fin_df = self._load_meta("FIN_METAINFO.csv")
        if not self.fin_df.empty:
            self.fin_df["fullpath"] = self.root + "/" + self.fin_df["path"]
            if "clear" not in self.fin_df.columns:
                # True=Clear, False=Blur, 空=Mid(未判定)
                self.fin_df["clear"] = pd.NA

    def _load_meta(self, fname):
        p = os.path.join(self.meta_dir, fname)
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    # 1. 全图
    def full_images(self):
        return self.root + "/" + self.image_df["orig_img_name"]

    def orig_of_fin(self, fin_path):
        """鳍裁剪图路径 -> 原始全图路径 (找不到返回 None)。"""
        if self.fin_df.empty or "orig_img_name" not in self.fin_df.columns:
            return None
        rows = self.fin_df[self.fin_df["fullpath"] == fin_path]
        if rows.empty:
            return None
        p = os.path.join(self.root, str(rows["orig_img_name"].iloc[0]))
        return p if os.path.isfile(p) else None

    # 2. 连拍分组
    def shots(self):
        if self.image_df.empty:  # 无元数据时退化为单组
            return {"all": self.full_images()}
        groups = {}
        for sid, g in self.image_df.groupby("shot_id", sort=True):
            groups["shot_%s" % sid] = [os.path.join(self.root, n)
                                       for n in g["orig_img_name"]]
        return groups

    def fin_boxes(self):
        """全图绝对路径 -> [(x0, y0, x1, y1, cls, extra), ...] (stored 像素坐标)
        extra 为叠加文字 (FinID 'F3' 和/或 clearness '0.87'), 都没有时为 None。"""
        boxes = {}
        if self.fin_df.empty or "orig_img_name" not in self.fin_df.columns:
            return boxes
        need = ["x_min", "y_min", "x_max", "y_max"]
        if not all(c in self.fin_df.columns for c in need):
            return boxes
        has_fid = "FinID" in self.fin_df.columns
        has_cle = "clearness" in self.fin_df.columns
        has_conf = "crop_conf" in self.fin_df.columns

        def extra(fid, cle, conf):
            parts = []
            if fid is not None and not pd.isna(fid):
                parts.append("F%d" % int(float(fid)))
            if conf is not None and not pd.isna(conf):
                parts.append("p=%.2f" % float(conf))
            if cle is not None and not pd.isna(cle):
                parts.append("c=%.2f" % float(cle))
            return " ".join(parts) or None

        for name, g in self.fin_df.groupby("orig_img_name"):
            p = os.path.join(self.root, str(name))
            if not os.path.isfile(p):
                continue
            fids = g["FinID"] if has_fid else [None] * len(g)
            cles = g["clearness"] if has_cle else [None] * len(g)
            confs = g["crop_conf"] if has_conf else [None] * len(g)
            boxes[p] = [
                (int(a), int(b), int(c), int(d), str(cls), extra(fid, cle, conf))
                for a, b, c, d, cls, fid, cle, conf in zip(
                    g["x_min"], g["y_min"], g["x_max"], g["y_max"],
                    g["class"], fids, cles, confs)]
        return boxes

    def fin_shot_labels(self):
        """鳍图 -> 'shot_<id> clr=<clearness> conf=<crop_conf>' 叠加标签。
        同时按绝对路径和文件名两种键收录, 以覆盖 FIN/FinIDxxx 里的副本。"""
        labels = {}
        if self.fin_df.empty or "shot_id" not in self.fin_df.columns:
            return labels
        has_cle = "clearness" in self.fin_df.columns
        has_conf = "crop_conf" in self.fin_df.columns
        cles = self.fin_df["clearness"] if has_cle else [float("nan")] * len(self.fin_df)
        confs = self.fin_df["crop_conf"] if has_conf else [float("nan")] * len(self.fin_df)
        for p, sid, cle, conf in zip(self.fin_df["fullpath"],
                                     self.fin_df["shot_id"], cles, confs):
            if pd.isna(sid):
                continue
            try:  # shot_id 常被读成浮点, 显示为整数
                sid = int(float(sid))
            except (ValueError, TypeError):
                pass
            text = "shot_%s" % sid
            if not pd.isna(cle):
                text += " clr=%.2f" % float(cle)
            if not pd.isna(conf):
                text += " conf=%.2f" % float(conf)
            if isinstance(p, str):
                labels[p] = text
                labels[os.path.basename(p)] = text
        return labels

    # 3. 部位分组
    def aspects(self):
        groups = {label: [] for label in ASPECT_LABELS.values()}  # 空组也列出, 作为拖放目标
        if self.fin_df.empty or "fullpath" not in self.fin_df.columns:
            return groups
        df = self.fin_df.dropna(subset=["fullpath"])
        for cls, g in df.groupby("class"):
            label = ASPECT_LABELS.get(cls, str(cls))
            groups.setdefault(label, []).extend(g["fullpath"])
        return groups

    # 4. 模糊分组: 依据 fin_df["clear"] 列 (True=Clear, False=Blurd)
    def blur(self):
        groups = {"Blur": [], "Clear": []}
        if self.fin_df.empty or "fullpath" not in self.fin_df.columns:
            return groups
        for val, g in self.fin_df.groupby("clear", dropna=False):
            key = "Clear" if val else "Blur"
            groups[key].extend(g["fullpath"])
        return groups

    # 5. 个体分组
    def fin_ids(self):
        groups = {}
        if self.fin_df.empty or "FinID" not in self.fin_df.columns:
            return groups
        for fid, g in self.fin_df.dropna(subset=["FinID"]).groupby("FinID", sort=True):
            groups["FinID_%s" % fid] = list(g["fullpath"])
        return groups

    # 6. 社会结构: 同一连拍共现个体的连通分量
    def social(self):
        # fin 文件名 -> FinID 组名
        fin_to_id = {}
        for fid, imgs in self.fin_ids().items():
            for p in imgs:
                fin_to_id[os.path.basename(p)] = fid
        if not fin_to_id:
            return {}
        # shot_id -> FinID 集合
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
    # 拖拽改组: 只更新 dataframe 并写回 CSV, 不移动/重命名任何文件
    # ------------------------------------------------------------------
    def _save_csv(self, df, fname):
        path = os.path.join(self.meta_dir, fname)
        if os.path.isfile(path):
            bdir = os.path.join(self.meta_dir, "backup")
            os.makedirs(bdir, exist_ok=True)
            shutil.copy2(path, os.path.join(
                bdir, fname + "." +
                datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"))
        df.drop(columns=["fullpath", "fin_name", "fin_id"], errors="ignore") \
            .to_csv(path, index=False)

    def move_shot(self, paths, group):
        """改连拍分组: 更新 image_df 和 fin_df 的 shot_id。"""
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

    def move_aspect(self, paths, group):
        """改部位类别: 只更新 fin_df 的 class 列。"""
        cls = {v: k for k, v in ASPECT_LABELS.items()}.get(group, group)
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        self.fin_df.loc[mask, "class"] = cls
        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return {p: p for p in paths}

    def move_blur(self, paths, group):
        """改质量分组: 只更新 fin_df 的 clear 列 (True=Clear, False=Blur, 空=Mid)。"""
        val = {"Clear": 1, "Blur": 0}.get(group, pd.NA)
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        self.fin_df.loc[mask, "clear"] = val
        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return {p: p for p in paths}

    def move_to_fin_id(self, paths, group):
        """改个体分组: 只更新 fin_df 的 FinID 列。"""
        m = re.match(r"FinID_(.+)$", group)
        if not m:
            return {}
        fid = m.group(1)
        try:  # FinID 为数值列时保持数值类型
            fid = int(float(fid))
        except ValueError:
            pass
        if "FinID" not in self.fin_df.columns:
            self.fin_df["FinID"] = pd.NA
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        self.fin_df.loc[mask, "FinID"] = fid
        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return {p: p for p in paths}


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

    def closeEvent(self, event):
        # 退出前停掉所有缩略图加载线程, 否则线程在解释器退出时析构会段错误
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            grids = ([w.grid] if hasattr(w, "grid")
                     else w.findChildren(ImageGrid))
            for g in grids:
                for loader in getattr(g, "_loaders", []):
                    loader.stop()
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            grids = ([w.grid] if hasattr(w, "grid")
                     else w.findChildren(ImageGrid))
            for g in grids:
                for loader in getattr(g, "_loaders", []):
                    loader.wait(2000)
        super().closeEvent(event)

    def load(self, root):
        self.root = root
        ds = Dataset(root)
        self.ds = ds
        self._group_tabs = []
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

        # 2-6. 分组页 (前四个支持拖拽改组: 只更新 dataframe 并写回 CSV, 不动文件)
        shot_boxes = ds.fin_boxes()  # Continuous Shots 页画鳍框
        fin_tabs = {"Fin Aspect", "Blur", "Fin ID"}  # 鳍图页: 中键查看原图, 叠加 shot_id
        alt_tabs = {"Fin Aspect", "Blur"}            # 支持 Alt+N 移入第 N 组
        shot_labels = ds.fin_shot_labels()  # 鳍图路径 -> shot_id 叠加文字
        for title, fn, mover in [
            ("Continuous Shots", ds.shots, ds.move_shot),
            ("Fin Aspect", ds.aspects, ds.move_aspect),
            ("Blur", ds.blur, ds.move_blur),
            ("Fin ID", ds.fin_ids, ds.move_to_fin_id),
            ("Social Structure", ds.social, None),
        ]:
            tab = GroupedTab(
                mover=mover,
                full_resolver=ds.orig_of_fin if title in fin_tabs else None,
                full_boxes=shot_boxes if title in fin_tabs else None,
                labels=shot_labels if title in fin_tabs else None,
                alt_move=title in alt_tabs,
                on_moved=self._refresh_tabs)
            groups = fn()
            if title == "Fin ID":  # 只显示图片数大于 1 的个体
                groups = {k: v for k, v in groups.items() if len(v) > 0}
            if groups:
                tooltips = getattr(ds, "social_tooltips", None) if title == "Social Structure" else None
                boxes = shot_boxes if title == "Continuous Shots" else None
                tab.set_groups(groups, tooltips, boxes=boxes)
            else:
                tab.grid.show_images([])
                QListWidgetItem("(no data)", tab.group_list)
            self.tabs.addTab(tab, title)
            self._group_tabs.append((title, tab, fn))

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
            if not groups:
                continue
            tooltips = getattr(ds, "social_tooltips", None) \
                if title == "Social Structure" else None
            tab._labels = shot_labels if title in fin_tabs else {}
            tab._full_boxes = shot_boxes if title in fin_tabs else {}
            boxes = shot_boxes if title == "Continuous Shots" else None
            tab.set_groups(groups, tooltips, keep_current=True, boxes=boxes)
            tab._on_group(None)  # 当前组名没变时信号不发, 强制用新标注重绘网格


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
