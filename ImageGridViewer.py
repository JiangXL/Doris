import os
import multiprocessing.connection

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QGridLayout,
    QScrollArea, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QRect

# =========================
# Selection Engine（核心）
# =========================
class SelectionEngine:
    def __init__(self, labels):
        self.labels = labels
        self.selected = set()
        self.anchor = None

    def click(self, label, modifiers):
        if modifiers & Qt.ShiftModifier and self.anchor:
            candidates = self._range(label)
            self._apply(candidates, "union")

        elif modifiers & Qt.ControlModifier:
            self._apply({label}, "toggle")
            self.anchor = label

        else:
            self._apply({label}, "replace")
            self.anchor = label

    def drag(self, rect, modifiers):
        candidates = {
            label for label in self.labels
            if rect.intersects(label.geometry())
        }

        if modifiers & Qt.ShiftModifier:
            self._apply(candidates, "union")

        elif modifiers & Qt.ControlModifier:
            self._apply(candidates, "toggle")

        else:
            self._apply(candidates, "replace")

    def _range(self, label):
        try:
            i1 = self.labels.index(self.anchor)
            i2 = self.labels.index(label)
        except ValueError:
            return set()

        start, end = sorted([i1, i2])
        return set(self.labels[start:end + 1])

    def _apply(self, candidates, mode):
        if mode == "replace":
            self.selected = set(candidates)

        elif mode == "union":
            self.selected |= candidates

        elif mode == "toggle":
            self.selected ^= candidates

    def get_selected(self):
        return self.selected
    
    def clear_selected(self):
        self.selected = set()

# =========================
# Clickable Label
# =========================
class ClickableLabel(QLabel):
    def __init__(self, img_path, img_id, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        self.img_id = img_id

    def mousePressEvent(self, event):
        if self.parent():
            self.parent().handle_click(self, event)

# =========================
# 主容器（支持框选）
# =========================
class SelectableWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.labels = []
        self.textlabels = []
        self.engine = SelectionEngine(self.labels)

        self.start_pos = None
        self.end_pos = None
        self.selecting = False

    # ---------- 点击 ----------
    def handle_click(self, label, event):
        modifiers = QApplication.keyboardModifiers()
        self.engine.click(label, modifiers)
        self.update_selection_style()

    # ---------- 鼠标拖拽 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.selecting:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.selecting:
            return

        self.selecting = False

        rect = QRect(self.start_pos, self.end_pos).normalized()
        modifiers = QApplication.keyboardModifiers()

        self.engine.drag(rect, modifiers)
        self.update_selection_style()
        self.update()

    # ---------- 绘制框 ----------
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.selecting and self.start_pos and self.end_pos:
            painter = QPainter(self)
            pen = QPen(Qt.blue, 2, Qt.DashLine)
            painter.setPen(pen)

            rect = QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect.normalized())

    # ---------- UI更新 ----------
    def update_selection_style(self):
        selected = self.engine.get_selected()

        for label in self.labels:
            if label in selected:
                label.setStyleSheet("border: 2px solid red;")
            else:
                label.setStyleSheet("")
        #print("Selected:", [l.img_path for l in selected])

# =========================
# 主界面
# =========================
class ImageGridViewer(QWidget):
    def __init__(self, cols=5, thumb_size=384):
        super().__init__()

        self.cols = cols
        self.thumb_size = thumb_size
        self.cilent = multiprocessing.connection.Client(
                ('localhost', 1126), authkey=b'dolphin')
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.container = SelectableWidget()
        grid = QGridLayout(self.container)


        fin_info = self.cilent.recv()
        images = fin_info["path"]
        annotation = fin_info["annotation"]
        images_id = fin_info["id"]

        row, col = 0, 0

        for i in range(len(images)):
            img_path = images[i]
            img_id = images_id[i]
            label = ClickableLabel(img_path, img_id, parent=self.container)

            pixmap = QPixmap(img_path).scaled(
                self.thumb_size,
                self.thumb_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            textlabel = QLabel(annotation[i])
            textlabel.setAlignment(Qt.AlignTop)

            self.container.labels.append(label)
            self.container.textlabels.append(textlabel)
            grid.addWidget(label, row, col)
            grid.addWidget(textlabel, row, col)

            col += 1
            if col >= self.cols:
                col = 0
                row += 1
        self.scroll.setWidget(self.container)
        confirm_button = QPushButton("Confirm Results and Process Next Group")
        confirm_button.clicked.connect(self.onButtonClick)
        layout.addWidget(confirm_button)
        layout.addWidget(self.scroll)
        self.container.engine.selected = set(self.container.labels)
        self.container.update_selection_style()

        self.setWindowTitle("Hi, Doris")
        self.resize(1080, 720)

    def onButtonClick(self):
        selected = self.container.engine.get_selected()
        print("Selected:", [l.img_id for l in selected])
        print("*")
        self.cilent.send([l.img_id for l in selected])

        # === 步骤 1: 删除旧 widget ===
        old_widget = self.scroll.widget()
        if old_widget:
            old_widget.deleteLater()

        # === 步骤 2: 创建新容器 ===
        self.container = SelectableWidget()
        grid = QGridLayout(self.container)

        # === 步骤 3: 接收新数据 ===
        fin_info = self.cilent.recv()
        images = fin_info["path"]
        annotation = fin_info["annotation"]
        images_id = fin_info["id"]

        # === 步骤 4: 填充新内容 ===
        row, col = 0, 0
        for i in range(len(images)):
            img_path = images[i]
            img_id = images_id[i]
            label = ClickableLabel(img_path, img_id, parent=self.container)

            pixmap = QPixmap(img_path).scaled(
                self.thumb_size,
                self.thumb_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            textlabel = QLabel(annotation[i])
            textlabel.setAlignment(Qt.AlignTop)

            self.container.labels.append(label)
            self.container.textlabels.append(textlabel)
            grid.addWidget(label, row, col)
            grid.addWidget(textlabel, row, col, Qt.AlignTop)
            self.container.engine.selected = set(self.container.labels)
            self.container.update_selection_style()

            col += 1
            if col >= self.cols:
                col = 0
                row += 1

        # === 步骤 5: 设置新 widget 到 scroll area ===
        self.scroll.setWidget(self.container)
