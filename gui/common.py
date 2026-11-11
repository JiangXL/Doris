# -*- coding: utf-8 -*-
"""通用工具、常量、缩略图加载、网格控件与图片查看器。"""

import os
import re
import shutil
import hashlib

from PyQt6.QtWidgets import (
    QApplication, QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QDialog,
    QScrollArea, QMessageBox, QGraphicsView, QGraphicsScene,
    QStyledItemDelegate, QStyle, QStyleOptionViewItem,
    QHBoxLayout, QPushButton, QMenu, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsItem, QInputDialog,
)
from PyQt6.QtGui import (
    QPixmap, QIcon, QImage, QImageReader, QImageIOHandler, QPainter,
    QBrush, QColor, QDrag, QPen, QPalette, QKeySequence, QFont, QShortcut,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPoint, QPointF, QRect, QRectF, QTimer
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
        color = BOX_COLORS.get(cls, Qt.GlobalColor.yellow)
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
    p.setPen(Qt.GlobalColor.white)
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
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

    @staticmethod
    def _is_fin_image(p):
        """路径是否在 FIN/ 目录下 (鳍图裁剪文件)。"""
        parts = re.split(r"[\\/]", os.path.normpath(p))
        return "FIN" in parts

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
        """读干净缩略图(不带框/标签), 优先磁盘缓存。返回 (img, orig_w, orig_h)。
        FIN/ 下的鳍图裁剪文件不读写磁盘缓存, 直接在内存中处理。"""
        use_cache = self._cache_dir and not self._is_fin_image(p)
        if use_cache:
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
        if trans & (QImageIOHandler.Transformation.TransformationRotate90 |
                    QImageIOHandler.Transformation.TransformationRotate270 |
                    QImageIOHandler.Transformation.TransformationFlipAndRotate90 |
                    QImageIOHandler.Transformation.TransformationMirrorAndRotate90):
            sz.transpose()
        # setScaledSize 不保持宽高比, 需按原始尺寸(含 EXIF 旋转)手动算目标尺寸
        sz.scale(self._thumb_size, self._thumb_size, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(sz)
        img = reader.read()
        if not img.isNull() and use_cache:
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
    self._paths 的元素为 (key, dl_fid, dr_fid, dl_path, dr_path, social_label);
    复用 _load_clean 的磁盘缓存。"""

    def run(self):
        for key, dl_fid, dr_fid, dl, dr, social in self._paths:
            if self._stop:
                break
            try:
                img = self._compose(key, dl_fid, dr_fid, dl, dr, social)
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

    _SOCIAL_COLORS = [
        QColor(255, 99, 71), QColor(30, 144, 255), QColor(255, 215, 0),
        QColor(50, 205, 50), QColor(255, 105, 180), QColor(0, 206, 209),
        QColor(255, 165, 0), QColor(148, 0, 211), QColor(0, 255, 127),
        QColor(255, 69, 0), QColor(100, 149, 237), QColor(220, 20, 60),
        QColor(0, 191, 255), QColor(255, 20, 147), QColor(124, 252, 0),
        QColor(255, 218, 185),
    ]

    @staticmethod
    def _social_color(label):
        """为每个 SocialStructure 标签生成稳定、可区分的颜色。"""
        if not label:
            return QColor(255, 255, 255)
        idx = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
        return DolphinLoader._SOCIAL_COLORS[idx % len(DolphinLoader._SOCIAL_COLORS)]

    def _compose(self, name, dl_fid, dr_fid, dl, dr, social_label=""):
        w, cell = THUMB_SIZE, FIN_CELL_H
        out = QImage(w, cell * 2, QImage.Format.Format_RGB32)
        out.fill(QColor(100, 100, 100))
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for i, (path, tag, fid) in enumerate(
                ((dl, "DL", dl_fid), (dr, "DR", dr_fid))):
            y0 = i * cell
            if path:
                img, _, _ = self._load_clean(path)
                if img is not None and not img.isNull():
                    # 居中裁到鳍格宽高比再精确填满, 上下两鳍无缝相叠
                    img = self._crop_ratio(img, w, cell)
                    t = img.scaled(w, cell, Qt.AspectRatioMode.IgnoreAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                    p.drawImage(0, y0, t)
            p.setPen(QColor(255, 255, 255))
            f = p.font()
            f.setPixelSize(14)
            f.setBold(True)
            p.setFont(f)
            if fid is not None:  # 每格左下角标 DL/DR, 右侧标对应 FinID
                fid_s = str(fid)
                p.drawText(6, y0 + cell - 8, fid_s)
        # SocialStructure 标签: 左上角, 每个组用自己的颜色
        if social_label:
            f = p.font()
            f.setPixelSize(13)
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            x = 6
            y = 18
            parts = [s.strip() for s in social_label.split(",")
                     if s.strip()]
            for i, part in enumerate(parts):
                p.setPen(self._social_color(part))
                p.drawText(x, y, part)
                x += fm.horizontalAdvance(part)
                if i < len(parts) - 1:
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(x, y, ", ")
                    x += fm.horizontalAdvance(", ")
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
        p.setPen(Qt.GlobalColor.white)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, name)
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
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        # 选中/焦点态由我们在图标上绘制; 交给默认样式会在空文本区留下一小截高亮条
        opt.state &= ~QStyle.StateFlag.State_Selected
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        icon_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemDecoration, opt, opt.widget)
        if not icon_rect.isValid():
            icon_rect = opt.rect
        painter.save()
        if selected:  # 选中: 图标描边 + 轻微染色
            hl = opt.palette.color(QPalette.ColorRole.Highlight)
            painter.setPen(QPen(hl, 4))
            painter.drawRect(icon_rect.adjusted(1, 1, -2, -2))
            painter.fillRect(icon_rect, QColor(hl.red(), hl.green(),
                                               hl.blue(), 40))
        # BestInFin 标记: 红色内边框
        best_set = getattr(opt.widget, "_best_in_fin", None)
        if best_set and index.data(Qt.ItemDataRole.UserRole) in best_set:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.drawRect(icon_rect.adjusted(3, 3, -3, -3))
        if text:
            painter.setFont(opt.font)
            bar_h = opt.fontMetrics.height() + 6
            bar = QRect(icon_rect.left(), icon_rect.top(),
                        icon_rect.width(), min(bar_h, icon_rect.height()))
            painter.fillRect(bar, self.BAR_COLOR)
            painter.setPen(Qt.GlobalColor.white)
            elided = opt.fontMetrics.elidedText(text, Qt.TextElideMode.ElideMiddle,
                                                bar.width() - 8)
            painter.drawText(bar.adjusted(4, 0, -4, 0),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)
        painter.restore()


class ImageGrid(QListWidget):
    """缩略图网格;双击弹窗查看大图。"""

    middle_clicked = pyqtSignal(str)  # 中键点击图片 (携带文件路径)
    right_clicked = pyqtSignal(QPoint)  # 右键点击 (携带全局坐标, 供上下文菜单)
    exclude_requested = pyqtSignal(list)  # Delete/Backspace 键请求排除选中图片
    paths_dropped = pyqtSignal(list)  # 拖入图片路径 (set_accept_drops(True) 后生效)
    paths_dropped_on = pyqtSignal(list, str)  # 拖放到某张图上 (额外携带目标图路径)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setIconSize(QSize(THUMB_SIZE, int(0.67*THUMB_SIZE)))
        # 文件名叠加在图上(delegate 绘制), 单元格不再为文字预留高度
        self.setGridSize(QSize(THUMB_SIZE + 16, int(0.67*THUMB_SIZE + 16)))
        #self.setSpacing(8)
        self.setUniformItemSizes(True)
        # 滚轮按像素平滑滚动, 而不是整行跳动
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.horizontalScrollBar().setSingleStep(24)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 多选
        self.setDragEnabled(True)                             # 可拖出
        self.setDragDropMode(QListWidget.DragDropMode.DragOnly)            # 不允许放回自身
        self.setItemDelegate(ThumbOverlayDelegate(self))
        self._loaders = []
        self._items = {}
        self._boxes = {}
        self._labels = {}
        self._loaded = set()    # 已加载缩略图的路径
        self._pending = set()   # 已派发给加载线程、尚未完成的路径
        self._cache_dir = None
        self._full_open_exclude = None
        self._full_open_best = None
        self._full_open_move = None
        self._box_editor = None
        self._best_in_fin = set()
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
        self._accept_self_drops = False  # 默认同网格内也不接受拖放 (防误触)

    def set_accept_drops(self, on):
        """允许/禁止从其它网格拖入图片路径 (自身拖出的不算)。"""
        self._accept_drops = on
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop if on
                             else QListWidget.DragDropMode.DragOnly)

    def set_accept_self_drops(self, on):
        """允许/禁止在同网格内拖放到另一张图上 (Overview 模式合并用)。"""
        self._accept_self_drops = on

    def set_full_open_callbacks(self, exclude=None, best=None, move=None):
        """设置大图查看窗口的排除/标记最佳/Alt+N 移组回调。"""
        self._full_open_exclude = exclude
        self._full_open_best = best
        self._full_open_move = move

    def set_box_editor(self, box_editor):
        """设置全图查看窗口的鳍框编辑回调。"""
        self._box_editor = box_editor

    def refresh_path(self, path):
        """强制重新加载指定路径的缩略图 (框坐标更新后调用)。"""
        self._loaded.discard(path)
        self._pending.discard(path)
        item = self._items.get(path)
        if item is not None:
            item.setIcon(QIcon())
        self._load_visible()

    def _drop_ok(self, event):
        if not event.mimeData().hasFormat(MIME_PATHS):
            return False
        from_self = event.source() is self
        if from_self:
            return self._accept_self_drops
        return self._accept_drops

    def dragEnterEvent(self, event):
        if self._drop_ok(event):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._drop_ok(event):
            event.setDropAction(Qt.DropAction.MoveAction)
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
                self.paths_dropped_on.emit(paths, item.data(Qt.ItemDataRole.UserRole))
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()

    def mimeData(self, items):
        """拖拽时携带选中项的文件路径。"""
        md = QMimeData()
        md.setData(MIME_PATHS, "\n".join(
            it.data(Qt.ItemDataRole.UserRole) for it in items).encode("utf-8"))
        return md

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            item = self.itemAt(event.pos())
            if item is not None:
                self.middle_clicked.emit(item.data(Qt.ItemDataRole.UserRole))
                event.accept()
                return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """右键: 光标下的图未选中时先单独选中它, 再弹出菜单 (全局坐标)。"""
        item = self.itemAt(event.pos())
        if item is not None and not item.isSelected():
            self.clearSelection()
            item.setSelected(True)
        self.right_clicked.emit(event.globalPos())

    def keyPressEvent(self, event):
        """Delete/Backspace: 把选中的图片路径发出排除请求, 由上层处理。"""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            items = self.selectedItems()
            if items:
                self.exclude_requested.emit(
                    [it.data(Qt.ItemDataRole.UserRole) for it in items])
                event.accept()
                return
        super().keyPressEvent(event)

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
        drag.exec(Qt.DropAction.MoveAction)

    @staticmethod
    def _drag_pixmap(items):
        base = items[0].icon().pixmap(96, 96)
        if base.isNull():
            # 缩略图尚未加载时直接从文件读一张小的
            reader = QImageReader(items[0].data(Qt.ItemDataRole.UserRole))
            reader.setAutoTransform(True)
            sz = QSize(reader.size())
            sz.scale(96, 96, Qt.AspectRatioMode.KeepAspectRatio)  # setScaledSize 不保宽高比
            reader.setScaledSize(sz)
            img = reader.read()
            base = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        if base.isNull():
            base = QPixmap(96, 96)
            base.fill(Qt.GlobalColor.lightGray)
        n = len(items)
        if n == 1:
            return base
        # 右上角画数量角标
        pm = base.copy()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        badge = QRect(pm.width() - 34, 2, 32, 32)
        p.setBrush(QColor(30, 144, 255, 220))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(badge)
        p.setPen(Qt.GlobalColor.white)
        f = p.font()
        f.setBold(True)
        f.setPixelSize(18)
        p.setFont(f)
        p.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(n))
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
            item.setData(Qt.ItemDataRole.UserRole, p)
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
                p = item.data(Qt.ItemDataRole.UserRole)
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
        paths = [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]
        dlg = ImageViewerDialog(paths, self.row(item), self, boxes=self._boxes,
                                labels=self._labels,
                                on_exclude=self._full_open_exclude,
                                on_best=self._full_open_best,
                                on_move=self._full_open_move,
                                box_editor=self._box_editor)
        dlg.exec()
class _HandleItem(QGraphicsRectItem):
    """ResizableBoxItem 的 8 个拖拽把手 (角 + 边中点)。"""

    HSIZE = 6

    def __init__(self, role, parent):
        super().__init__(parent)
        self.role = role
        self.setAcceptHoverEvents(True)
        self.setCursor(self._cursor_for(role))
        self.setBrush(QColor(255, 255, 255))
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setRect(-self.HSIZE, -self.HSIZE,
                     self.HSIZE * 2, self.HSIZE * 2)

    def _cursor_for(self, role):
        if role in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        if role in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        if role in ("t", "b"):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeHorCursor

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            parent = self.parentItem()
            if isinstance(parent, ResizableBoxItem):
                parent._start_handle_drag(self.role, event.scenePos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        parent = self.parentItem()
        if isinstance(parent, ResizableBoxItem):
            parent._do_handle_drag(self.role, event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        parent = self.parentItem()
        if isinstance(parent, ResizableBoxItem):
            parent._end_drag()
        event.accept()


class ResizableBoxItem(QGraphicsRectItem):
    """可拖拽移动/缩放的鳍框项, 带 8 个把手和类别标签。"""

    MIN_SIZE = 10

    def __init__(self, box, image_w, image_h, parent=None):
        super().__init__(parent)
        x0, y0, x1, y1, cls, extra = box
        self._box = box
        self._image_w = image_w
        self._image_h = image_h
        self._cls = cls.lstrip("_") if isinstance(cls, str) else cls
        self._extra = extra
        self.setRect(QRectF(float(x0), float(y0),
                            max(1, float(x1) - float(x0)),
                            max(1, float(y1) - float(y0))))
        color = BOX_COLORS.get(self._cls, Qt.GlobalColor.yellow)
        self.setPen(QPen(color, max(2, min(image_w, image_h) // 400)))
        self.setAcceptHoverEvents(True)
        self._handles = {}
        for role in ("tl", "t", "tr", "r", "br", "b", "bl", "l"):
            self._handles[role] = _HandleItem(role, self)
        self._label = QGraphicsTextItem(self)
        self._label.setDefaultTextColor(color)
        font = self._label.font()
        font.setPixelSize(max(12, min(image_w, image_h) // 40))
        font.setBold(True)
        self._label.setFont(font)
        self._label.setPlainText(self._label_text())
        self._update_handles()
        self._update_label()
        self._handle_drag_role = None
        self._body_drag = False
        self._drag_start_mouse = None
        self._drag_start_rect = None

    def _label_text(self):
        text = self._cls if self._cls else ""
        if self._extra:
            text = "%s %s" % (text, self._extra) if text else self._extra
        return text

    def _update_handles(self):
        r = self.rect()
        self._handles["tl"].setPos(r.left(), r.top())
        self._handles["t"].setPos(r.left() + r.width() / 2, r.top())
        self._handles["tr"].setPos(r.right(), r.top())
        self._handles["r"].setPos(r.right(), r.top() + r.height() / 2)
        self._handles["br"].setPos(r.right(), r.bottom())
        self._handles["b"].setPos(r.left() + r.width() / 2, r.bottom())
        self._handles["bl"].setPos(r.left(), r.bottom())
        self._handles["l"].setPos(r.left(), r.top() + r.height() / 2)

    def _update_label(self):
        r = self.rect()
        h = self._label.boundingRect().height()
        self._label.setPos(r.left(), max(0, r.top() - h - 2))

    def _clamp(self, x, y):
        return (max(0, min(self._image_w, x)),
                max(0, min(self._image_h, y)))

    def _start_handle_drag(self, role, pos):
        self._handle_drag_role = role
        self._drag_start_rect = self.rect()
        self._do_handle_drag(role, pos)

    def _do_handle_drag(self, role, pos):
        x, y = self._clamp(pos.x(), pos.y())
        r = self._drag_start_rect
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        if "l" in role:
            left = min(x, right - self.MIN_SIZE)
        if "r" in role:
            right = max(x, left + self.MIN_SIZE)
        if "t" in role:
            top = min(y, bottom - self.MIN_SIZE)
        if "b" in role:
            bottom = max(y, top + self.MIN_SIZE)
        new_rect = QRectF(left, top, right - left, bottom - top)
        self.setRect(new_rect)
        self._update_handles()
        self._update_label()

    def _start_body_drag(self, pos):
        self._body_drag = True
        self._drag_start_mouse = pos
        self._drag_start_rect = self.rect()

    def _do_body_drag(self, pos):
        delta = pos - self._drag_start_mouse
        r = self._drag_start_rect
        new_x = r.x() + delta.x()
        new_y = r.y() + delta.y()
        new_x = max(0, min(self._image_w - r.width(), new_x))
        new_y = max(0, min(self._image_h - r.height(), new_y))
        self.setRect(new_x, new_y, r.width(), r.height())
        self._update_handles()
        self._update_label()

    def _end_drag(self):
        self._handle_drag_role = None
        self._body_drag = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_body_drag(event.scenePos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._body_drag:
            self._do_body_drag(event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._body_drag:
            self._end_drag()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ZoomableView(QGraphicsView):
    """滚轮缩放(以光标为中心), 左键拖拽平移, 双击适应窗口。"""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)               # 拖拽平移
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)  # 以光标为中心缩放
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def zoom(self, factor):
        """以视图中心为锚点缩放 (键盘快捷键用; 滚轮仍以光标为中心)。"""
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        # ←/→ 交给对话框做翻页, 不做滚动
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)


class ImageViewerDialog(QDialog):
    """大图查看窗口。←/→ 切换组内上一张/下一张, =/- 放大/缩小, Ctrl+W 关闭。
    可选 on_exclude / on_best / on_misc 回调, 在大图窗口中按 Delete / B / M
    或右键排除/标记最佳/标记 PhotoType 为 MISC。
    传入 box_editor 后, 右键可进入鳍框编辑模式 (移动/缩放框并保存)。"""

    def __init__(self, paths, index, parent=None, boxes=None, labels=None,
                 on_exclude=None, on_best=None, on_misc=None, on_move=None,
                 box_editor=None):
        super().__init__(parent)
        self._paths = paths
        self._index = index
        self._boxes = boxes or {}
        self._labels = labels or {}
        self._on_exclude = on_exclude
        self._on_best = on_best
        self._on_misc = on_misc
        self._on_move = on_move
        self._box_editor = box_editor
        self._edit_mode = False
        self._edit_items = []
        self._edit_original_rects = []
        self._add_mode = False
        self._add_start = None
        self._add_rect_item = None
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
            self.view.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _enter_edit_mode(self):
        """进入框编辑模式: 显示无框原图, 并在其上放置可交互框。"""
        self._edit_mode = True
        path = self._paths[self._index]
        reader = QImageReader(path)
        pm = QPixmap.fromImage(reader.read())
        if not pm.isNull():
            self._item.setPixmap(pm)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        boxes = self._boxes.get(path, [])
        self._edit_original_rects = []
        self._edit_items = []
        w = pm.width() if not pm.isNull() else 0
        h = pm.height() if not pm.isNull() else 0
        for box in boxes:
            rb = ResizableBoxItem(box, w, h)
            self._scene.addItem(rb)
            self._edit_items.append(rb)
            self._edit_original_rects.append(rb.rect())
        self._scene.setSceneRect(self._scene.itemsBoundingRect())

    def _exit_edit_mode(self, save=False):
        """退出编辑模式; save=True 时将改动写回数据集。"""
        path = self._paths[self._index]
        if save and self._box_editor is not None:
            boxes = self._boxes.get(path, [])
            for i, rb in enumerate(self._edit_items):
                if i >= len(boxes):
                    continue
                r = rb.rect()
                orig = self._edit_original_rects[i]
                if r == orig:
                    continue
                old_box = boxes[i]
                new_box = (int(r.left()), int(r.top()),
                           int(r.right()), int(r.bottom()),
                           old_box[4], old_box[5])
                updated = self._box_editor(path, "update", i, new_box)
                if updated is not None:
                    self._boxes[path] = updated
        for item in self._edit_items:
            self._scene.removeItem(item)
        self._edit_items = []
        self._edit_original_rects = []
        self._edit_mode = False
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._show(self._index)

    def _enter_add_mode(self):
        """进入画新鳍框模式: 左键拖拽画框。"""
        if self._box_editor is None:
            return
        self._add_mode = True
        self._add_start = None
        self._add_rect_item = None
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().installEventFilter(self)

    def _cancel_add_mode(self):
        """取消画框模式。"""
        self._add_mode = False
        self._add_start = None
        if self._add_rect_item is not None:
            self._scene.removeItem(self._add_rect_item)
            self._add_rect_item = None
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.viewport().removeEventFilter(self)

    def _finish_add_box(self, rect):
        """拖拽完成后询问 Aspect 并回调 box_editor(action='add')。"""
        path = self._paths[self._index]
        pm = self._item.pixmap() if self._item is not None else None
        if pm is None or pm.isNull():
            return
        x0 = max(0, int(rect.left()))
        y0 = max(0, int(rect.top()))
        x1 = min(pm.width(), int(rect.right()))
        y1 = min(pm.height(), int(rect.bottom()))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return
        aspect, ok = QInputDialog.getItem(
            self, "Add Fin", "Aspect:", ["DL", "DR", "Others"], 0, False)
        if not ok:
            return
        box = (x0, y0, x1, y1, str(aspect), None)
        updated = self._box_editor(path, "add", None, box)
        if updated is not None:
            self._boxes[path] = updated
        self._show(self._index)

    def eventFilter(self, obj, event):
        if not self._add_mode or obj is not self.view.viewport():
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._add_start = self.view.mapToScene(event.pos())
            self._add_rect_item = self._scene.addRect(
                QRectF(self._add_start, self._add_start),
                QPen(QColor(0, 160, 255), 2))
            return True
        if event.type() == QEvent.Type.MouseMove and self._add_rect_item is not None:
            end = self.view.mapToScene(event.pos())
            self._add_rect_item.setRect(QRectF(self._add_start, end).normalized())
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton \
                and self._add_rect_item is not None:
            end = self.view.mapToScene(event.pos())
            rect = QRectF(self._add_start, end).normalized()
            self._scene.removeItem(self._add_rect_item)
            self._add_rect_item = None
            self._add_mode = False
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.viewport().removeEventFilter(self)
            self._finish_add_box(rect)
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if self._add_mode:
            if event.key() == Qt.Key.Key_Escape:
                self._cancel_add_mode()
                event.accept()
                return
            event.accept()
            return
        if self._edit_mode:
            if event.key() == Qt.Key.Key_Escape:
                self._exit_edit_mode(save=False)
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                # 编辑时禁止翻页
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
                self.view.zoom(1.25)
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
                self.view.zoom(0.8)
                event.accept()
                return
            if event.key() == Qt.Key.Key_W and \
                    event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.close()
                event.accept()
                return
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Left:
            self._show(self._index - 1)   # 上一张
        elif event.key() == Qt.Key.Key_Right:
            self._show(self._index + 1)   # 下一张
        elif event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
            self.view.zoom(1.25)          # =/+ 放大
        elif event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.view.zoom(0.8)           # - 缩小
        elif event.key() == Qt.Key.Key_A and self._box_editor is not None:
            self._enter_add_mode()
            event.accept()
        elif event.key() == Qt.Key.Key_W and \
                event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.close()                  # Ctrl+W 关闭
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) \
                and self._on_exclude is not None:
            self._on_exclude([self._paths[self._index]])
            self.close()
            event.accept()
        elif event.key() == Qt.Key.Key_B and self._on_best is not None:
            self._on_best([self._paths[self._index]])
            event.accept()
        elif event.key() == Qt.Key.Key_M and self._on_misc is not None:
            self._on_misc([self._paths[self._index]])
            event.accept()
        elif event.modifiers() & Qt.KeyboardModifier.AltModifier and self._on_move is not None:
            # Alt+1..9: 移入第 N 个分组
            key = event.key()
            if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
                self._on_move(self._paths[self._index], key - Qt.Key.Key_1)
                event.accept()
                return
            super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        """右键菜单: 编辑框 / 添加鳍 / 排除 / 标记最佳; 编辑/添加模式下显示保存/取消。"""
        menu = QMenu(self)
        if self._edit_mode:
            act = menu.addAction("Save changes")
            act.triggered.connect(lambda: self._exit_edit_mode(save=True))
            act = menu.addAction("Cancel")
            act.triggered.connect(lambda: self._exit_edit_mode(save=False))
        elif self._add_mode:
            act = menu.addAction("Cancel add")
            act.triggered.connect(self._cancel_add_mode)
        else:
            path = self._paths[self._index]
            boxes = self._boxes.get(path)
            if self._box_editor is not None:
                if boxes:
                    act = menu.addAction("Edit boxes")
                    act.triggered.connect(self._enter_edit_mode)
                act = menu.addAction("Add fin (A)")
                act.triggered.connect(self._enter_add_mode)
            if self._on_exclude is not None:
                act = menu.addAction("Exclude (Delete)")
                act.triggered.connect(
                    lambda: self._on_exclude([self._paths[self._index]])
                    or self.close())
            if self._on_best is not None:
                act = menu.addAction("Mark Best Fin (B)")
                act.triggered.connect(
                    lambda: self._on_best([self._paths[self._index]]))
            if self._on_misc is not None:
                act = menu.addAction("Mark MISC (M)")
                act.triggered.connect(
                    lambda: self._on_misc([self._paths[self._index]]))
        if not menu.isEmpty():
            menu.exec(event.globalPos())
