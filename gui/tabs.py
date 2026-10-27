# -*- coding: utf-8 -*-
"""通用分组页、统计页与 Dashboard 页。"""

import os
import re
import sys
import shutil
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx

from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QSplitter, QListWidget, QListWidgetItem,
    QLabel, QVBoxLayout, QHBoxLayout, QDialog, QScrollArea,
    QFileDialog, QMessageBox, QShortcut, QPushButton,
    QStackedWidget, QFrame, QToolButton, QMenu, QCheckBox, QComboBox,
    QGridLayout, QPlainTextEdit,
)
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QPen, QKeySequence, QFont, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData, QPoint, QRect, QTimer, QProcess

from doris_dataset import Dataset
from .common import (
    ImageGrid, ThumbLoader, MIME_PATHS, THUMB_SIZE,
    ZoomableView, ImageViewerDialog,
)
from .dolphin import DolphinTab, MatchPanel

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
                 best_marker=None, best_in_fin_provider=None, renamer=None,
                 track_move_checkbox=False, shot_type_setter=None,
                 shot_info_provider=None, box_editor=None, parent=None):
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
        best_marker: callable(paths) -> bool; 给定后右键网格或按 B 键把选中鳍图
        标记为最佳 (写回 FIN_METAINFO 的 BestInFin 列)
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
        下可把代表鳍图拖放到另一张上, 合并两个 FinID (含对应 DolphinID)
        renamer: callable() -> dict; 给定后顶栏出现 "Rename ID" 按钮,
        把所有 FinID 重新编号为 1..N (N=FinID 数量)
        track_move_checkbox: True 时顶栏出现 "Move whole track" 复选框,
        默认勾选; Review 模式下侧边拖放勾选时移动同 track, 未勾选时只移动选中图
        shot_type_setter: callable(shot_id, shot_type); 给定后顶栏出现 ShotType
        下拉框, 供 Continuous Shots 页为整组连拍设置类型
        shot_info_provider: callable(shot_id) -> dict; 给定后顶栏显示该 shot
        第一张图的 DateTime/FocusLength35mm/Distance 信息"""
        super().__init__(parent)
        self._mover = mover
        self._track_mover = track_mover
        self._exclude = exclude
        self._renamer = renamer
        self._on_moved = on_moved
        self._full_resolver = full_resolver
        self._full_boxes = full_boxes or {}
        self._labels = labels or {}
        self._alt_move = alt_move and mover is not None
        self._best_mark = best_mark
        self._best_only = False
        self._clear_mark = clear_mark
        self._clear_only = False
        self._best_marker = best_marker
        self._best_in_fin_provider = best_in_fin_provider
        self._similar_provider = similar_provider
        self._overview_provider = overview_provider
        self._overview = False      # 总览模式
        self._merger = merger       # Overview 模式拖放合并 FinID
        self._tri = False           # 三栏模式
        self._shot_type_setter = shot_type_setter
        self._shot_info_provider = shot_info_provider
        self._box_editor = box_editor
        self.vsplit = None
        self.similar_grid = None
        self.excluded_grid = None
        self._excluded = []         # [(参考组名, 鳍图路径)] 拖入排除区的记录
        self._needs_refresh = False  # 隐藏时标记脏, 切回前台再刷新
        self.splitter = QSplitter(Qt.Horizontal)
        self.group_list = GroupListWidget()
        self.group_list.setMinimumWidth(60)   # 保证初始宽度, 否则 splitter 按空列表 sizeHint 给得很窄
        self.group_list.setMaximumWidth(120)
        self.group_list.setWordWrap(False)
        self.group_list.setTextElideMode(Qt.ElideNone)  # 不省略
        self.grid = ImageGrid()
        self.grid.set_box_editor(box_editor)
        self.splitter.addWidget(self.group_list)
        self.splitter.addWidget(self.grid)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([160, 1000])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if self._best_mark is not None or self._overview_provider is not None \
                or self._clear_mark is not None \
                or self._shot_type_setter is not None \
                or self._shot_info_provider is not None \
                or (track_move_checkbox and self._track_mover is not None):
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
                                          "Clear==1 的清晰鳍图")
                self._clear_cb.toggled.connect(self._on_clear_toggled)
                top.addWidget(self._clear_cb)
            if self._renamer is not None:
                self._rename_btn = QPushButton("Rename ID")
                self._rename_btn.setToolTip(
                    "把所有 FinID 重新编号为 FinID1..FinIDN (N=当前 FinID 总数)")
                self._rename_btn.clicked.connect(self._rename_fin_ids)
                top.addWidget(self._rename_btn)
            self._track_move_cb = None
            if track_move_checkbox and self._track_mover is not None:
                self._track_move_cb = QCheckBox("Move whole track")
                self._track_move_cb.setToolTip(
                    "勾选后侧边拖放会同时移动同 ShotID+TrackIDInShot 的鳍图")
                self._track_move_cb.setChecked(True)
                top.addWidget(self._track_move_cb)
            self._shot_type_cb = None
            self._shot_info_label = None
            if self._shot_type_setter is not None \
                    or self._shot_info_provider is not None:
                if self._shot_type_setter is not None:
                    top.addWidget(QLabel("ShotType:"))
                    self._shot_type_cb = QComboBox()
                    self._shot_type_cb.addItems(
                        ["", "NOTE", "SEABIRD", "Dolphin", "MISC"])
                    self._shot_type_cb.setMinimumWidth(90)
                    self._shot_type_cb.currentTextChanged.connect(
                        self._on_shot_type_changed)
                    top.addWidget(self._shot_type_cb)
                if self._shot_info_provider is not None:
                    self._shot_info_label = QLabel()
                    self._shot_info_label.setStyleSheet("color: #666666;")
                    top.addWidget(self._shot_info_label)
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
        if self._best_marker is not None or self._exclude is not None:
            # 右键菜单: 标记最佳 / 排除
            self.grid.right_clicked.connect(self._on_grid_menu)
        if self._best_marker is not None:
            sc = QShortcut(QKeySequence("B"), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(self._mark_best_selected)
        if self._exclude is not None:
            # Delete 键: 把上方当前组选中的鳍图移入排除区
            self.grid.exclude_requested.connect(self._exclude_selected)
        if self._best_marker is not None or self._exclude is not None:
            # 大图查看窗口的右键/快捷键同样支持排除与标记最佳
            self.grid.set_full_open_callbacks(
                exclude=self._full_open_exclude_cb
                if self._exclude is not None else None,
                best=self._full_open_best_cb
                if self._best_marker is not None else None)

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

        fin_crop_path = path
        def _edit_for_middle(orig_path, action, index, new_box=None):
            if self._box_editor is None:
                return None
            updated = self._box_editor(orig_path, action, index, new_box)
            if updated is not None:
                # 原图框和 FIN/ 裁剪图已更新, 刷新鳍图页网格缩略图
                self.grid.refresh_path(fin_crop_path)
            return updated

        dlg = ImageViewerDialog(origs, origs.index(orig), self,
                                boxes=self._full_boxes,
                                box_editor=_edit_for_middle)
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
            n = sum(1 for name in groups if name != "FinID0")
            self.count_label.setText("FinID 总数: %d" % n)
        if not groups:
            return
        # FinID0 不是真实个体, 默认选中其后第一个组 (如 FinID1)
        if cur_name in groups:
            self.group_list.setCurrentRow(list(groups).index(cur_name))
        else:
            names = list(groups)
            row = 1 if len(names) > 1 and names[0] == "FinID0" else 0
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
        if self._track_move_cb is not None:
            self._track_move_cb.setEnabled(mode == "review")
        self.group_list.setVisible(not self._overview)
        if self._best_only and self._similar_provider is not None:
            self._enter_tri_mode()
        else:
            self._exit_tri_mode()
        if self._overview and self._merger is not None:
            # 拖放合并: 可拖出且接受同网格内落到另一张图上
            self.grid.setDragEnabled(True)
            self.grid.set_accept_drops(True)
            self.grid.set_accept_self_drops(True)
        elif not self._tri:  # 三栏模式自己管理拖放
            self.grid.setDragEnabled(self._mover is not None)
            self.grid.set_accept_drops(False)  # 退出 overview 合并拖放
            self.grid.set_accept_self_drops(False)
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
        self.similar_grid.right_clicked.connect(self._on_similar_grid_menu)
        self.similar_grid.exclude_requested.connect(
            self._exclude_similar_selected)
        self.similar_grid.set_full_open_callbacks(
            exclude=self._full_open_exclude_cb
            if self._exclude is not None else None,
            best=self._full_open_best_cb
            if self._best_marker is not None else None)
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

    def _on_grid_menu(self, pos):
        """右键网格: 弹出上下文菜单 (标记最佳 / 排除)。"""
        menu = QMenu(self)
        if self._best_marker is not None:
            act = menu.addAction("Mark Best Fin (B)")
            act.setToolTip("把选中鳍图标记为当前 FinID 的最佳 "
                           "(BestInFin=True, 同 FinID 其它鳍自动置 False)")
            act.triggered.connect(self._mark_best_selected)
        if self._exclude is not None:
            act = menu.addAction("Exclude (Delete)")
            act.setToolTip("把选中的鳍图移入排除区 (Review/Match 模式均可用)")
            act.triggered.connect(self._exclude_selected)
        if not menu.isEmpty():
            menu.exec_(pos)

    def _mark_best_selected(self):
        """把网格中选中的鳍图标记为该 FinID 的最佳并刷新标注。"""
        if self._best_marker is None:
            return
        items = self.grid.selectedItems()
        if not items:
            return
        paths = [it.data(Qt.UserRole) for it in items]
        try:
            changed = self._best_marker(paths)
        except Exception as e:
            QMessageBox.warning(self, "标记失败", str(e))
            return
        if not changed:
            return
        if self._on_moved is not None:
            self._on_moved(self)  # 元数据已变, 刷新所有 tab 的标注
        else:
            self._on_group(None)

    def _exclude_selected(self):
        """把上方当前组选中的鳍图移入排除区 (Review/Match 模式均可用)。"""
        if self._exclude is None:
            return
        items = self.grid.selectedItems()
        if not items:
            return
        paths = [it.data(Qt.UserRole) for it in items]
        self._on_exclude_drop(paths)

    def _on_similar_grid_menu(self, pos):
        """右键相似个体网格: 排除与当前 FinID 的相似关系。"""
        menu = QMenu(self)
        act = menu.addAction("Exclude from current FinID (Delete)")
        act.setToolTip("把相似个体的鳍图与当前 FinID 的相似度置 0")
        act.triggered.connect(self._exclude_similar_selected)
        menu.exec_(pos)

    def _exclude_similar_selected(self):
        """Match 模式下: 把左下相似网格中选中的鳍图排除。"""
        if self.similar_grid is None or self._exclude is None:
            return
        items = self.similar_grid.selectedItems()
        if not items:
            return
        paths = [it.data(Qt.UserRole) for it in items]
        self._on_exclude_drop(paths)

    def _full_open_exclude_cb(self, paths):
        """大图查看窗口的排除回调: 把当前查看的鳍图移入排除区。"""
        if not paths or self._exclude is None:
            return
        self._on_exclude_drop(paths)

    def _full_open_best_cb(self, paths):
        """大图查看窗口的标记最佳回调: 把当前查看的鳍图标为 BestInFin。"""
        if not paths or self._best_marker is None:
            return
        try:
            changed = self._best_marker(paths)
        except Exception as e:
            QMessageBox.warning(self, "标记失败", str(e))
            return
        if changed and self._on_moved is not None:
            self._on_moved(self)

    def _rename_fin_ids(self):
        """顶栏 Rename ID 按钮: 重新编号所有 FinID 并刷新界面。"""
        if self._renamer is None:
            return
        reply = QMessageBox.question(
            self, "Rename ID",
            "确定把所有 FinID 重新编号为 FinID1..FinIDN 吗?")
        if reply != QMessageBox.Yes:
            return
        try:
            mapping = self._renamer()
        except Exception as e:
            QMessageBox.warning(self, "重命名失败", str(e))
            return
        if not mapping:
            return
        # 排除区记录里的旧组名失效, 清空
        self._excluded = []
        if self._on_moved is not None:
            self._on_moved(self)
        else:
            self._on_group(None)

    def _on_clear_toggled(self, checked):
        """"Clear" 复选框: 勾选后 Review 模式只显示 Clear==1 的清晰鳍图。"""
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
        if self._best_in_fin_provider is not None:
            best_set = self._best_in_fin_provider()
            self.grid._best_in_fin = best_set
            if self._tri and self.similar_grid is not None:
                self.similar_grid._best_in_fin = best_set
        if self._tri and self.similar_grid is not None:
            sim_paths, sim_labels = self._similar_provider(
                item.data(Qt.UserRole))
            labels = dict(self._labels)
            labels.update(sim_labels)  # 相似度标签优先
            self.similar_grid.show_images(sim_paths, labels=labels)
            self._refresh_excluded()  # 排除区跟随当前参考组
        self._update_shot_widgets(item.data(Qt.UserRole))

    def _update_shot_widgets(self, group_name):
        """Continuous Shots 页: 根据当前 shot 更新 ShotType 下拉框与信息标签。"""
        if self._shot_info_provider is None and self._shot_type_setter is None:
            return
        if not isinstance(group_name, str) or "_" not in group_name:
            return
        shot_id = group_name.split("_", 1)[1]
        info = {}
        if self._shot_info_provider is not None:
            try:
                info = self._shot_info_provider(shot_id) or {}
            except Exception:
                info = {}
        if self._shot_type_cb is not None:
            self._shot_type_cb.blockSignals(True)
            self._shot_type_cb.setCurrentText(str(info.get("shot_type", "")))
            self._shot_type_cb.blockSignals(False)
        if self._shot_info_label is not None:
            parts = []
            if info.get("datetime"):
                parts.append("DateTime: %s" % info["datetime"][0:-4])
            if info.get("focus_length") is not None:
                parts.append("Focus Length: %smm" % info["focus_length"])
            if info.get("distance") is not None:
                parts.append("Distance: %.fm" % info["distance"])
            self._shot_info_label.setText("  |  ".join(parts))

    def _on_shot_type_changed(self, text):
        """ShotType 下拉框改变时保存到 PHOTO_METAINFO.csv。"""
        if self._shot_type_setter is None:
            return
        item = self.group_list.currentItem()
        if item is None:
            return
        group_name = item.data(Qt.UserRole)
        if not isinstance(group_name, str) or "_" not in group_name:
            return
        shot_id = group_name.split("_", 1)[1]
        try:
            self._shot_type_setter(shot_id, text)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _on_drop(self, paths, target_name, use_track=None):
        """use_track=None 时：优先看 "Move whole track" 复选框（Review 模式）;
        否则跟随最佳复选框 (Match 模式三栏)。勾选后侧边拖放改同 track 的鳍。"""
        if use_track is None:
            if self._track_move_cb is not None:
                use_track = self._track_move_cb.isChecked()
            else:
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
    """统计页: FinID 鳍图数 / FinID shot 数 / Clearness 分布 /
    部位占比饼图 / DolphinID shot 数 (DL/DR 分组)。
    数据经 show_stats(...) 传入; 可点击按钮保存整页概览图到 METAINFO。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        self._root = None
        self._fin_counts = None
        self._shot_counts = None
        self._aspect_counts = None
        self._clear_values = None
        self._dolphin_shot_counts = None
        self._needs_refresh = False

        self.fig_hist = Figure()
        self.fig_shot = Figure()
        self.fig_clear = Figure()
        self.fig_pie = Figure()
        self.fig_dolphin = Figure()
        self.fig_social = Figure()
        self._social_data = (None, None, None)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部保存按钮
        bar = QWidget()
        bar.setFixedHeight(26)
        top = QHBoxLayout(bar)
        top.setContentsMargins(4, 0, 4, 0)
        self._save_statistics_btn = QPushButton("Save Statistics")
        self._save_statistics_btn.setToolTip("将整个统计页保存为一张图")
        self._save_statistics_btn.clicked.connect(self._save_statistics)
        top.addWidget(self._save_statistics_btn)

        top.addStretch(1)
        main_layout.addWidget(bar)

        plots = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.addWidget(FigureCanvasQTAgg(self.fig_hist))
        left.addWidget(FigureCanvasQTAgg(self.fig_shot))
        left.addWidget(FigureCanvasQTAgg(self.fig_social))
        right.addWidget(FigureCanvasQTAgg(self.fig_pie))
        right.addWidget(FigureCanvasQTAgg(self.fig_dolphin))
        right.addWidget(FigureCanvasQTAgg(self.fig_clear))
        plots.addLayout(left, 2)
        plots.addLayout(right, 1)
        main_layout.addLayout(plots)

    def set_root(self, root):
        """设置项目根目录, 用于决定保存路径。"""
        self._root = root


    def _save_statistics(self):
        """把整个统计页保存为一张 PNG/PDF 到 METAINFO/。"""
        if not self._root:
            QMessageBox.warning(self, "Doris", "未设置根目录, 无法保存")
            return
        out_dir = os.path.join(self._root, "METAINFO")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "Statistics.pdf")
        try:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(14, 20))
            self._render_overview(fig)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            QMessageBox.information(
                self, "Doris", "已保存整页概览图到:\n%s" % path)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _render_overview(self, fig):
        """在传入的 Figure 上绘制整页统计图 (3 行 x 2 列布局,
        右侧第三行为社会网络图)。"""
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[2, 1])

        self._draw_bar(ax1, self._fin_counts, "Fin images per FinID",
                       fontsize=8)
        self._draw_stacked_bar(ax2, self._shot_counts,
                               "Shots per FinID", fontsize=8)
        self._draw_hist(ax3, self._clear_values, "Clearness distribution",
                        fontsize=8)
        self._draw_pie(ax4, self._aspect_counts, "Fin aspect distribution",
                       fontsize=10)
        self._draw_stacked_bar(ax5, self._dolphin_shot_counts,
                               "Shots per DolphinID", fontsize=8)
        self._draw_social_network(ax6, self._social_data,
                                  title="Dolphin social network")
        fig.tight_layout()

    def _draw_bar(self, ax, counts, title, fontsize=5):
        if counts:
            labels = [k for k, _ in counts]
            values = [v for _, v in counts]
            ax.bar(range(len(values)), values, color="#4C72B0")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
            for i, v in enumerate(values):
                ax.text(i, v, str(v), ha="center", va="bottom",
                        fontsize=fontsize)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.margins(y=0.15)

    def _bar(self, fig, counts, title):
        fig.clear()
        ax = fig.add_subplot(111)
        if counts and len(counts[0]) == 3:
            self._draw_stacked_bar(ax, counts, title)
        else:
            self._draw_bar(ax, counts, title)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _draw_hist(self, ax, values, title, bins=30, fontsize=7):
        if values:
            ax.hist(values, bins=bins, color="#55A868", edgecolor="white")
            ax.axvline(np.mean(values), color="#C44E52", linestyle="--",
                       linewidth=1.5, label="mean=%.2f" % np.mean(values))
            ax.legend(loc="best", fontsize=fontsize)
        ax.set_title(title)
        ax.set_xlabel("Clearness")
        ax.set_ylabel("count")
        ax.margins(y=0.15)

    def _hist(self, fig, values, title, bins=30):
        fig.clear()
        ax = fig.add_subplot(111)
        self._draw_hist(ax, values, title, bins)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _draw_stacked_bar(self, ax, counts, title, fontsize=5):
        """counts: [(label, dl_count, dr_count), ...]; DL 在下, DR 在上堆叠。"""
        if counts:
            labels = [k for k, _, _ in counts]
            dl_vals = [dl for _, dl, _ in counts]
            dr_vals = [dr for _, _, dr in counts]
            x = np.arange(len(labels))
            width = 0.6
            ax.bar(x, dl_vals, width, label="DL", color="#4C72B0")
            ax.bar(x, dr_vals, width, bottom=dl_vals, label="DR",
                   color="#DD8452")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
            ax.legend(loc="upper right", fontsize=max(6, fontsize - 1))
            for i, (dl, dr) in enumerate(zip(dl_vals, dr_vals)):
                if dl:
                    ax.text(i, dl / 2, str(dl), ha="center", va="center",
                            fontsize=max(4, fontsize - 1), color="white")
                if dr:
                    ax.text(i, dl + dr / 2, str(dr), ha="center",
                            va="center", fontsize=max(4, fontsize - 1),
                            color="white")
        ax.set_title(title)
        ax.set_ylabel("shots")
        ax.margins(y=0.15)

    def _stacked_bar(self, fig, counts, title):
        fig.clear()
        ax = fig.add_subplot(111)
        self._draw_stacked_bar(ax, counts, title)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _draw_social_network(self, ax, social_data, title="Dolphin social network"):
        """用 networkx 绘制海豚社会网络图。

        social_data: (nodes, edges, node_counts)
            nodes: list[str] — DolphinID
            edges: list[(u, v, weight)] — 共同出现的不同 ShotID 数
            node_counts: dict[DolphinID, int] — 每只海豚出现的 shot 数
        """
        nodes, edges, node_counts = social_data
        ax.clear()
        if not nodes or not edges:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No co-occurrence data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return
        connected = set()
        for u, v, _ in edges:
            connected.add(u)
            connected.add(v)
        nodes = [n for n in nodes if n in connected]
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        counts = np.array([node_counts.get(n, 1) for n in nodes], dtype=float)
        min_c, max_c = counts.min(), counts.max()
        if max_c > min_c:
            norm = (np.sqrt(counts) - np.sqrt(min_c)) / (
                np.sqrt(max_c) - np.sqrt(min_c))
        else:
            norm = np.zeros_like(counts)
        node_sizes = (200 + norm * 1000).tolist()
        # 增大斥力常数 k 并增加迭代次数, 让节点/标签尽量不重叠
        k = max(4.0, 10.0 / np.sqrt(len(nodes)))
        pos = nx.spring_layout(G, weight="weight", seed=42, k=k,
                               iterations=200, threshold=1e-4)
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        max_w = max(weights) if weights else 1
        widths = [1.5 + 4 * w / max_w for w in weights]
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=widths, alpha=0.7,
            edge_color="#377eb8", connectionstyle="arc3,rad=0.1",
            arrows=True, arrowstyle="-")
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes,
            node_color="skyblue", edgecolors="white")
        nx.draw_networkx_labels(
            G, pos, ax=ax, labels={n: n for n in nodes}, font_size=8,
            font_color="#333333")
        edge_labels = {(u, v): "%d" % int(d["weight"])
                       for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_labels, font_size=7,
            font_color="#377eb8", bbox=dict(boxstyle="round,pad=0.12",
                                            facecolor="white",
                                            edgecolor="none", alpha=0.85))
        ax.set_title("%s (%d connected)" % (title, len(nodes)))
        ax.axis("off")

    def _social_network(self, fig, social_data, title="Dolphin social network"):
        fig.clear()
        ax = fig.add_subplot(111)
        self._draw_social_network(ax, social_data, title)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def _draw_pie(self, ax, aspect_counts, title, fontsize=9):
        if aspect_counts:
            labels = [k for k, _ in aspect_counts]
            values = [v for _, v in aspect_counts]
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": fontsize})
        ax.set_title(title)

    def _pie(self, fig, aspect_counts, title):
        fig.clear()
        ax = fig.add_subplot(111)
        self._draw_pie(ax, aspect_counts, title)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def show_stats(self, fin_counts, shot_counts, aspect_counts, clear_values,
                   dolphin_shot_counts, social_network_data):
        self._fin_counts = fin_counts
        self._shot_counts = shot_counts
        self._aspect_counts = aspect_counts
        self._clear_values = clear_values
        self._dolphin_shot_counts = dolphin_shot_counts
        self._social_data = social_network_data

        self._bar(self.fig_hist, fin_counts, "Fin images per FinID")
        self._bar(self.fig_shot, shot_counts, "Shots per FinID")
        self._stacked_bar(self.fig_dolphin, dolphin_shot_counts,
                          "Shots per DolphinID")
        self._hist(self.fig_clear, clear_values, "Clearness distribution")
        self._pie(self.fig_pie, aspect_counts, "Fin aspect distribution")
        self._social_network(self.fig_social, social_network_data,
                             "Dolphin social network")


# ---------------------------------------------------------------------------
# Dashboard 页
# ---------------------------------------------------------------------------
class DashboardTab(QWidget):
    """Dashboard 页: 处理流程入口 (布局参考 dashboard.ui)。

    按钮 1 选择根目录 (发射 root_changed, 由主窗口重载);
    其余按钮以子进程运行对应流水线脚本 (扫描 / 切鳍 / 提特征 / 聚类 /
    导出数据库 / 导出 Dolphin Overview PPT), 运行期间按钮禁用, 结束后恢复并提示。"""

    root_changed = pyqtSignal(str)
    scan_finished = pyqtSignal()
    locate_finished = pyqtSignal()
    cluster_finished = pyqtSignal()

    STEPS = [
        # (按钮文字, 脚本文件名, 是否需要先弹出保存对话框)
        ("2. Scan Photos", "../Step1_scan_photo.py", False),
        ("3. Locate Fin", "../Step2_locate_fin.py", False),
        ("4. Extract Fin Features", "../Step3_extract_fin_feature.py", False),
        ("5. Select good quality fin", "../Step4_filter_low_quality.py", False),
        ("6. Clustering", "../Step5_sort_fin_automatic.py", False),
        ("7. Export Database", "../Step6a_export_database.py", False),
        ("8. Export Dolphin Overview PPT", "../Step6b_export_pptx.py", True),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = None
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        btn_root = QPushButton("1.  Set Root Folder")
        btn_root.clicked.connect(self._pick_root)
        grid.addWidget(btn_root, 0, 0)
        self.root_label = QLabel("(未设置)")
        self.root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.root_label, 0, 1)
        for i, (text, script, save_dialog) in enumerate(self.STEPS, 1):
            btn = QPushButton(text)
            if save_dialog:
                btn.clicked.connect(lambda _=False, s=script, b=btn:
                                    self._run_step_with_save_dialog(s, b))
            else:
                btn.clicked.connect(lambda _=False, s=script, b=btn:
                                    self._run_step(s, b))
            grid.addWidget(btn, i, 0)
            if i == 1:  # Scan Images 右侧显示 PHOTO_METAINFO 状态
                self._scan_status = QLabel()
                grid.addWidget(self._scan_status, i, 1)
            if i == 2:  # Locate Fin 右侧显示 FIN_METAINFO 状态
                self._fin_status = QLabel()
                grid.addWidget(self._fin_status, i, 1)
            if i == 3:  # Extract Fin Features 右侧显示 FIN_DEEPFEATURES 状态
                self._feature_status = QLabel()
                grid.addWidget(self._feature_status, i, 1)
        # 底部终端式日志: 显示各 QProcess 的 stdout/stderr
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c; }")
        self._log.setMaximumBlockCount(500)
        self._log.setPlaceholderText("子进程输出将显示在这里...")
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(self._log, stretch=1)
        lay.setStretchFactor(self._log, 1)

    def set_root(self, root):
        """主窗口加载/切换根目录后同步显示; 同时刷新各步骤状态标签。"""
        self._root = root
        self.root_label.setText(root)
        self._update_scan_status()
        self._update_fin_status()
        self._update_feature_status()

    def _pick_root(self):
        """弹出非原生目录选择对话框, 避免某些平台下原生对话框点击"打开"不关闭的问题。"""
        dlg = QFileDialog(self, "选择 PHOTO 根目录", self._root or "")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        if dlg.exec_() == QDialog.Accepted:
            files = dlg.selectedFiles()
            if files:
                self.root_changed.emit(files[0])

    def _update_scan_status(self):
        """Scan Images 按钮右侧: 显示 PHOTO_METAINFO.csv 是否存在及图片数量。"""
        if not self._root or not hasattr(self, "_scan_status"):
            return
        csv = os.path.join(self._root, "METAINFO", "PHOTO_METAINFO.csv")
        if not os.path.isfile(csv):
            self._scan_status.setText("Wait to Scan")
            self._scan_status.setStyleSheet("color: #d9534f;")
            return
        try:
            import pandas as pd
            n = len(pd.read_csv(csv))
            self._scan_status.setText(
                "METAINFO/PHOTO_METAINFO.csv found, %d photos" % n)
            self._scan_status.setStyleSheet("color: #5cb85c;")
        except Exception:
            self._scan_status.setText("METAINFO/PHOTO_METAINFO.csv found")
            self._scan_status.setStyleSheet("color: #5cb85c;")

    def _update_fin_status(self):
        """Locate Fin 按钮右侧: 显示 FIN_METAINFO.csv 是否存在及鳍图数量。"""
        if not self._root or not hasattr(self, "_fin_status"):
            return
        csv = os.path.join(self._root, "METAINFO", "FIN_METAINFO.csv")
        if not os.path.isfile(csv):
            self._fin_status.setText("Wait to Locate")
            self._fin_status.setStyleSheet("color: #d9534f;")
            return
        try:
            import pandas as pd
            n = len(pd.read_csv(csv))
            self._fin_status.setText(
                "METAINFO/FIN_METAINFO.csv found, %d fin images" % n)
            self._fin_status.setStyleSheet("color: #5cb85c;")
        except Exception:
            self._fin_status.setText("METAINFO/FIN_METAINFO.csv found")
            self._fin_status.setStyleSheet("color: #5cb85c;")

    def _update_feature_status(self):
        """Extract Fin Features 按钮右侧: 显示 FIN_DEEPFEATURES 是否存在。"""
        if not self._root or not hasattr(self, "_feature_status"):
            return
        feat = os.path.join(self._root, "METAINFO", "FIN_DEEPFEATURES")
        if not os.path.exists(feat):
            self._feature_status.setText("Wait to Extract")
            self._feature_status.setStyleSheet("color: #d9534f;")
            return
        self._feature_status.setText("METAINFO/FIN_DEEPFEATURES found")
        self._feature_status.setStyleSheet("color: #5cb85c;")

    def _run_step(self, script, btn):
        """以子进程运行流水线脚本 (带当前根目录), 完成后恢复按钮并提示。"""
        self._start_step(script, btn, [self._root])

    def _run_step_with_save_dialog(self, script, btn):
        """弹出保存对话框并把选中的路径作为额外参数传给脚本。"""
        if not self._root:
            QMessageBox.warning(self, "Doris",
                                "请先通过 \"1. Set Root Folder\" 设置根目录")
            return
        default = os.path.join(self._root, "EXPORT", "dolphin_overview.pptx")
        out, _ = QFileDialog.getSaveFileName(
            self, "导出 Dolphin Overview 到 PPT", default,
            "PowerPoint (*.pptx)")
        if not out:
            return
        self._start_step(script, btn, [self._root, out])

    def _append_log(self, text, color=None):
        """向底部终端日志追加一行/多行文本 (可选 HTML 颜色)。"""
        if not text:
            return
        from datetime import datetime
        for line in text.rstrip("\n").split("\n"):
            stamp = datetime.now().strftime("%H:%M:%S")
            if color:
                self._log.appendHtml('<span style="color:%s">[%s] %s</span>'
                                     % (color, stamp, line))
            else:
                self._log.appendPlainText("[%s] %s" % (stamp, line))

    def _start_step(self, script, btn, args):
        """启动 QProcess 运行脚本, args 为传给脚本的参数列表。"""
        if not self._root:
            QMessageBox.warning(self, "Doris",
                                "请先通过 \"1. Set Root Folder\" 设置根目录")
            return
        path = os.path.join(self._script_dir, script)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Doris", "找不到脚本: %s" % path)
            return
        self._append_log(
            "启动 %s: %s %s" % (btn.text(), sys.executable,
                               " ".join([path] + args)))
        btn.setEnabled(False)
        self._set_status_running(btn.text())
        proc = QProcess(self)
        proc.readyReadStandardOutput.connect(
            lambda: self._append_log(
                bytes(proc.readAllStandardOutput()).decode(
                    "utf-8", errors="replace")))
        proc.readyReadStandardError.connect(
            lambda: self._append_log(
                bytes(proc.readAllStandardError()).decode(
                    "utf-8", errors="replace"), color="#ff6b6b"))
        proc.finished.connect(
            lambda code, _status, b=btn: self._step_finished(b, code))
        proc.start(sys.executable, [path] + args)

    def _set_status_running(self, btn_text):
        """根据正在运行的步骤, 把右侧状态标签设为 Scanning/Locating/Extracting。"""
        if "Scan Images" in btn_text:
            self._scan_status.setText("Scanning")
            self._scan_status.setStyleSheet("color: #f0ad4e;")
        elif "Locate Fin" in btn_text:
            self._fin_status.setText("Locating")
            self._fin_status.setStyleSheet("color: #f0ad4e;")
        elif "Extract Fin Features" in btn_text:
            self._feature_status.setText("Extracting")
            self._feature_status.setStyleSheet("color: #f0ad4e;")

    def _step_finished(self, btn, exit_code):
        btn.setEnabled(True)
        if "Scan Images" in btn.text():
            self._update_scan_status()
            self.scan_finished.emit()
        if "Locate Fin" in btn.text():
            self._update_fin_status()
            self.locate_finished.emit()
        if "Extract Fin Features" in btn.text():
            self._update_feature_status()
        if "Clustering" in btn.text() and exit_code == 0:
            self.cluster_finished.emit()
        self._append_log("%s 已结束 (exit code %d)" % (btn.text(), exit_code),
                         color="#5bc0de" if exit_code == 0 else "#ff6b6b")
        QMessageBox.information(self, "Doris", "%s 已结束 (exit code %d)"
                                % (btn.text(), exit_code))
