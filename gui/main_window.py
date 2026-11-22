# -*- coding: utf-8 -*-
"""主窗口：组装所有标签页、数据集加载与跨页刷新。"""

import os
from collections import deque

import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QMessageBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QMenu, QStackedWidget, QListWidgetItem,
    QInputDialog,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QImage

from doris_dataset import Dataset
from .tabs import DashboardTab, GroupedTab, StatisticsTab
from .dolphin import DolphinTab, MatchPanel, CandidateList
from .common import ImageGrid
from resource_paths import resource_path

DORIS_VERSION = "0.2.5"


class _FeatureWorker(QThread):
    """后台重算深度特征/相似度, 避免 GUI 线程做 ONNX 推理时卡死。"""

    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self._fn()
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.done.emit()

class MainWindow(QMainWindow):
    def __init__(self, root=None):
        super().__init__()
        self.setWindowTitle("Doris v%s — %s" % (DORIS_VERSION, root) if root
                            else "Doris v%s" % DORIS_VERSION)
        self.setWindowIcon(QIcon(resource_path('Delphinus_IAU.png')))
        self.resize(1280, 800)

        self._dashboard = None  # Dashboard 页常驻: 重载数据页时保留终端输出
        self._group_tabs = []   # 未 load 前为空, 供 _on_tab_changed 安全遍历
        self._stats_tab = None
        self._feat_jobs = deque()      # 待执行的特征重算任务 (串行)
        self._feat_worker = None       # 正在执行的后台特征重算线程
        self.tabs = QTabWidget()
        # 切到统计页时惰性重算, 保证任何改组后看到的都是最新数据
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        if root:
            self.load(root)
        else:
            # 未选根目录: 只放 Dashboard, 步骤按钮全禁用, 等用户选目录
            self.root = None
            self._ensure_dashboard()
            self._dashboard.set_root(None)

    def _ensure_dashboard(self):
        """创建并连接 Dashboard 页 (只建一次, 重载数据页时保留终端输出)。"""
        if self._dashboard is not None:
            return
        self._dashboard = DashboardTab()
        self._dashboard.root_changed.connect(self.load)
        self._dashboard.data_changed.connect(lambda: self.load(self.root))
        self.tabs.addTab(self._dashboard, "Dashboard")

    def _on_tab_changed(self, i):
        """切换标签页时, 若目标页标记为脏则重建。"""
        w = self.tabs.widget(i)
        if w is self._stats_tab:
            if getattr(self._stats_tab, "_needs_refresh", False):
                self._rebuild_stats()
            return
        if w is getattr(self, "_low_quality_tab", None):
            if getattr(self._low_quality_tab, "_needs_refresh", False):
                self._rebuild_low_quality()
            return
        for title, tab, fn in self._group_tabs:
            if tab is w:
                if getattr(tab, "_needs_refresh", False):
                    self._rebuild_group_tab(tab, title, fn)
                return
        if w is getattr(self, "_dolphin_page", None):
            cur = self._dolphin_stack.currentWidget()
            if getattr(cur, "_needs_refresh", False):
                if cur is self._dolphin_tab:
                    social_map = self.ds.dolphin_social_map()
                    self._dolphin_tab.show_dolphins(self.ds.dolphins(),
                                                    social=social_map)
                elif cur is self._match_panel:
                    self._match_panel.show_dolphins(self.ds.dolphins())
                elif cur is self._dolphin_review_tab:
                    self._rebuild_dolphin_review()
                cur._needs_refresh = False

    @staticmethod
    def _stop_thumb_loaders(w):
        """停掉一个 tab 页上所有缩略图加载线程 (stop + wait), 供重载/退出前调用。
        findChildren 覆盖所有网格 (含 Fin ID 三栏模式的网格和 Dolphin 页)。"""
        grids = ([w] if isinstance(w, DolphinTab) else []) \
            + w.findChildren(ImageGrid) + w.findChildren(DolphinTab) \
            + w.findChildren(CandidateList)
        for g in grids:
            for loader in getattr(g, "_loaders", []):
                loader.stop()
        for g in grids:
            for loader in getattr(g, "_loaders", []):
                loader.wait(2000)

    def closeEvent(self, event):
        # 退出前停掉所有缩略图加载线程, 否则线程在解释器退出时析构会段错误
        for i in range(self.tabs.count()):
            self._stop_thumb_loaders(self.tabs.widget(i))
        # 等待正在运行的特征重算线程, 避免 .npy 写一半被中断
        if getattr(self, "_feat_worker", None) is not None:
            self._feat_worker.wait(30000)
        super().closeEvent(event)

    def _clear_data_tabs(self):
        """移除并销毁 Dashboard 之外的所有标签页 (停缩略图线程后 deleteLater),
        为重载数据页做准备; Dashboard (及终端输出) 保留。"""
        self._tab_requires = {}
        self._group_tabs = []
        self._dolphin_tab = None
        self._dolphin_page = None
        self._stats_tab = None
        self._low_quality_tab = None
        self._low_quality_grid = None
        for i in range(self.tabs.count() - 1, 0, -1):
            w = self.tabs.widget(i)
            self._stop_thumb_loaders(w)
            self.tabs.removeTab(i)
            w.deleteLater()

    def load(self, root):
        self.root = root
        self.setWindowTitle("Doris %s — %s" % (DORIS_VERSION, root))
        meta_dir = os.path.join(root, "METAINFO")
        self._required_files = {
            "PHOTO_METAINFO.csv": os.path.join(meta_dir, "PHOTO_METAINFO.csv"),
            "FIN_METAINFO.csv": os.path.join(meta_dir, "FIN_METAINFO.csv"),
            "FIN_DEEPFEATURES": os.path.join(meta_dir, "FIN_DEEPFEATURES"),
            # callable 检查 FIN_METAINFO.csv 是否已有对应列 (各步骤产出):
            # Analyse Fin -> Aspect/Clearness; Clustering -> FinID/DolphinID
            "FIN_ANALYSED": lambda: self._fin_has_columns("Aspect", "Clearness"),
            "FIN_CLUSTERED": lambda: self._fin_has_columns("FinID"),
            "FIN_DOLPHIN": lambda: self._fin_has_columns("DolphinID"),
        }
        self._clear_data_tabs()
        self._ensure_dashboard()
        self._dashboard.set_root(root)

        # Dataset 可容忍缺失文件 (空 dataframe), 因此始终创建, 用于初始化各页
        ds = Dataset(root)
        self.ds = ds

        # 1-4. 分组页 (支持拖拽改组: 只更新 dataframe 并写回 CSV, 不动文件)
        shot_boxes = ds.fin_boxes()  # Continuous Shots 页画鳍框
        fin_tabs = {"Fin Aspect", "Fin Quality", "Fin ID"}  # 鳍图页: 中键查看原图, 叠加 shot_id
        alt_tabs = {"Fin Aspect", "Fin Quality"}            # 支持 Alt+N 移入第 N 组
        shot_labels = ds.fin_shot_labels()  # 鳍图路径 -> shot_id 叠加文字
        photo_labels = ds.photo_shot_labels()  # 原图路径 -> shot_id 叠加文字
        best_marks = ds.fin_best_marks()    # Fin ID 页 "每 shot 最佳" 筛选
        clear_marks = ds.fin_clear_marks()  # Fin ID / Dolphin 页 "Clear" 筛选
        for title, fn, mover in [
            ("Continuous Shots", ds.shots, ds.move_shot),
            ("Fin Aspect", ds.aspects, ds.move_aspect),
            ("Fin Quality", ds.blur, ds.move_blur),
            ("Fin ID", ds.fin_ids, ds.move_selected_to_fin_id),
        ]:
            tab = GroupedTab(
                mover=mover,
                full_resolver=ds.orig_of_fin if title in fin_tabs else None,
                full_boxes=shot_boxes if title in fin_tabs else None,
                labels=(photo_labels if title == "Continuous Shots"
                        else (shot_labels if title in fin_tabs else None)),
                orig_labels=photo_labels if (title in fin_tabs
                                             or title == "Continuous Shots")
                else None,
                alt_move=title in alt_tabs,
                on_moved=self._refresh_tabs,
                best_mark=best_marks.__contains__ if title == "Fin ID" else None,
                clear_mark=clear_marks.__contains__ if title == "Fin ID" else None,
                best_marker=ds.mark_best_fins if title == "Fin ID" else None,
                best_in_fin_provider=ds.fin_best_in_fin_marks
                if title == "Fin ID" else None,
                renamer=ds.rename_fin_ids if title == "Fin ID" else None,
                similar_provider=ds.similar_fins if title == "Fin ID" else None,
                track_mover=ds.move_track_to_fin_id if title == "Fin ID" else None,
                track_move_checkbox=True if title == "Fin ID" else False,
                shot_type_setter=ds.set_shot_type
                if title == "Continuous Shots" else None,
                shot_info_provider=ds.shot_info
                if title == "Continuous Shots" else None,
                box_editor=self._edit_box
                if title in fin_tabs or title == "Continuous Shots" else None,
                photo_type_setter=ds.set_photo_type,
                exclude=(ds.split_to_new_fin_id, ds.exclude_fin_id_pair)
                if title == "Fin ID" else None,
                overview_provider=ds.fin_id_previews
                if title == "Fin ID" else None,
                merger=ds.merge_fin_paths if title == "Fin ID" else None,
                similar_mover=ds.move_to_fin_id
                if title == "Fin ID" else None,
                empty_drop_handler=ds.split_to_new_fin_id
                if title == "Fin ID" else None,
                fin_id_remover=ds.remove_fin_id
                if title == "Fin ID" else None,
                aspect_mover=ds.move_aspect
                if title == "Fin ID" else None,
                dolphin_assigner=self._assign_dolphin_to_fin_id
                if title == "Fin ID" else None)
            groups = fn()
            if title == "Fin ID":  # 只显示图片数大于 0 的个体; "0" 未分类组始终保留
                groups = {k: v for k, v in groups.items() if len(v) > 0}
                if groups:
                    groups.setdefault("0", [])
            if groups:
                boxes = shot_boxes if title == "Continuous Shots" else None
                tab.set_groups(groups, boxes=boxes)
            else:
                tab.grid.show_images([])
                QListWidgetItem("(no data)", tab.group_list)
            idx = self.tabs.addTab(tab, title)
            self._group_tabs.append((title, tab, fn))
            # Continuous Shots 依赖原图元数据; Fin Aspect/Fin Quality/Fin ID 还需
            # "4. Analyse Fin" 完成 (FIN_METAINFO 有 Aspect/Clearness 列)
            if title == "Continuous Shots":
                self._tab_requires[idx] = "PHOTO_METAINFO.csv"
            elif title in ("Fin Aspect", "Fin Quality", "Fin ID"):
                self._tab_requires[idx] = "FIN_ANALYSED"
            elif title == "Fin ID":
                self._tab_requires[idx] = "FIN_CLUSTERED" 
            else:
                self._tab_requires[idx] = "FIN_METAINFO.csv"

        # 5. 海豚配对页: 每只海豚一格, 上 DL 下 DR, DR 格可拖拽交换;
        #    把 DR 鳍拖到空白处可创建新海豚; 顶栏 "Mode" 图标按钮弹出菜单
        #    选择三个互斥模式: Overview=海豚格 (DR 可拖拽交换, 默认) /
        #    Match=匹配面板 / Review=左侧 DolphinID 列表 + 右侧全部鳍图
        tab9 = QWidget()
        lay = QVBoxLayout(tab9)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
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
        self._dolphin_mode_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._dolphin_mode_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._dolphin_mode_btn.setMenu(menu)
        top.addWidget(self._dolphin_mode_btn)
        self._btn_rename_dph = QPushButton("Rename ID")
        self._btn_rename_dph.setToolTip(
            "把所有 DolphinID 重新编号为 Dph001..DphN (N=当前海豚总数)")
        self._btn_rename_dph.clicked.connect(self._rename_dolphin_ids)
        top.addWidget(self._btn_rename_dph)
        self._btn_reorder_rel = QPushButton("Reorder by Relationship")
        self._btn_reorder_rel.setToolTip(
            "按社会网络关系 (共同出现次数) 重新排序海豚网格")
        self._btn_reorder_rel.clicked.connect(
            self._reorder_dolphins_by_relationship)
        top.addWidget(self._btn_reorder_rel)
        top.addStretch(1)
        lay.addLayout(top)
        self._dolphin_stack = QStackedWidget()
        self._dolphin_tab = DolphinTab()
        self._dolphin_tab.show_dolphins(ds.dolphins(),
                                        social=ds.dolphin_social_map())
        self._dolphin_tab.dr_swapped.connect(self._swap_dolphin_dr)
        self._dolphin_tab.dr_dropped_on_empty.connect(
            self._create_dolphin_from_dr)
        self._match_panel = MatchPanel(ds.dolphin_candidates)
        self._match_panel.assign_requested.connect(self._match_assign)
        self._match_panel.unmatch_requested.connect(self._match_unmatch)
        self._dolphin_stack.addWidget(self._dolphin_tab)
        self._dolphin_stack.addWidget(self._match_panel)
        # Review 模式: 左侧 DolphinID 列表, 右侧该海豚的全部鳍图 (只读)
        self._dolphin_review_tab = GroupedTab(
            mover=None, full_resolver=ds.orig_of_fin,
            labels=ds.dolphin_fin_labels(),
            orig_labels=photo_labels,
            clear_mark=ds.fin_clear_marks().__contains__,
            best_marker=ds.mark_best_fins,
            best_in_fin_provider=ds.fin_best_in_fin_marks,
            photo_type_setter=ds.set_photo_type,
            on_moved=self._refresh_tabs)
        groups = ds.dolphin_groups()
        if groups:
            self._dolphin_review_tab.set_groups(groups)
        else:
            self._dolphin_review_tab.grid.show_images([])
            QListWidgetItem("(no data)", self._dolphin_review_tab.group_list)
        self._dolphin_stack.addWidget(self._dolphin_review_tab)
        lay.addWidget(self._dolphin_stack)
        idx = self.tabs.addTab(tab9, "Dolphin")
        self._dolphin_page = tab9
        self._dolphin_tab._needs_refresh = False
        self._match_panel._needs_refresh = False
        self._tab_requires[idx] = "FIN_DOLPHIN"  # 需 Clustering 产出 DolphinID

        # 6. 社会结构页 (只读): 整组连拍 (shot_id) 内出现多于一个 DolphinID
        #    的 shot, 同一组 DolphinID 的多个 shot 合并为一组 (NN001...),
        #    组内只收同一张原图含多个海豚鳍的图;
        #    组名写回 IMAGE/FIN_METAINFO.csv 的 SocialStructure 列;
        #    仿 Continuous Shots 页在原图上画鳍框, 标注 FinID 和 DolphinID
        #    (须排在 Dolphin 页后, ds.dolphins() 已分配 DolphinID 列)
        tab_social = GroupedTab(mover=None, on_moved=self._refresh_tabs,
                                box_editor=self._edit_box,
                                labels=photo_labels)
        groups = ds.social()
        if groups:
            tab_social.set_groups(groups, ds.social_tooltips,
                                  boxes=ds.fin_boxes(dolphin_id=True))
        else:
            tab_social.grid.show_images([])
            QListWidgetItem("(no data)", tab_social.group_list)
        idx = self.tabs.addTab(tab_social, "Subgroup")
        self._group_tabs.append(("Social Structure", tab_social, ds.social))
        self._tab_requires[idx] = "FIN_DOLPHIN"  # 需 Clustering 产出 DolphinID

        # 7. 统计页: FinID 鳍图数柱状图 + 部位占比饼图
        self._stats_tab = StatisticsTab()
        self._stats_tab.set_root(root)
        self._stats_tab.show_stats(
            ds.fin_id_counts(), ds.fin_id_shot_counts(), ds.aspect_counts(),
            ds.clearness_values(), ds.dolphin_shot_counts(),
            ds.social_network_data(), ds.traffic_data(), ds.timeline_data())
        idx = self.tabs.addTab(self._stats_tab, "Statistics")
        self._tab_requires[idx] = None  # 统计页常开, 无数据时图表为空

        # 8. 低质量页: 列出无鳍或没有 Clear+已分配 DolphinID 的原始照片
        self._low_quality_tab = QWidget()
        lq_lay = QVBoxLayout(self._low_quality_tab)
        lq_lay.setContentsMargins(0, 0, 0, 0)
        self._low_quality_grid = ImageGrid()
        self._low_quality_grid.set_box_editor(self._edit_box)
        lq_lay.addWidget(self._low_quality_grid)
        idx = self.tabs.addTab(self._low_quality_tab, "Low Quality")
        self._tab_requires[idx] = "FIN_CLUSTERED"  # 需 Clustering 产出 FinID
        self._low_quality_tab._needs_refresh = False
        self._rebuild_low_quality()

        # 根据已有元数据启用/禁用对应标签页; 流水线完成后信号会再次刷新
        self._refresh_tab_enablement()

    def _refresh_tab_enablement(self):
        """根据 METAINFO 下文件是否存在, 启用或禁用各标签页。"""
        available = {k: (v() if callable(v) else os.path.isfile(v))
                     for k, v in getattr(self, "_required_files", {}).items()}
        for idx, req in getattr(self, "_tab_requires", {}).items():
            # req 为 None 表示常开 (统计页)
            enabled = True if req is None else available.get(req, False)
            self.tabs.setTabEnabled(idx, enabled)

    def _fin_has_columns(self, *cols):
        """FIN_METAINFO.csv 是否存在且含指定列 (Analyse Fin 产出 Aspect/Clearness,
        Clustering 产出 FinID/DolphinID; 列缺失说明对应步骤未完成)。"""
        csv = os.path.join(self.root, "METAINFO", "FIN_METAINFO.csv")
        if not os.path.isfile(csv):
            return False
        try:
            header = pd.read_csv(csv, nrows=0).columns
        except Exception:
            return False
        return all(c in header for c in cols)


    def _set_dolphin_mode(self, mode):
        """切换 Dolphin 页模式 (互斥): overview=海豚格 (DR 可拖拽交换, 默认),
        match=匹配面板, review=左侧 DolphinID 列表 + 右侧该海豚全部鳍图。"""
        for k, act in self._dolphin_mode_actions.items():
            act.setChecked(k == mode)
        self._dolphin_mode_btn.setText(
            "Mode: %s" % self._dolphin_mode_actions[mode].text())
        self._btn_reorder_rel.setVisible(mode == "overview")
        if mode == "review":
            self._dolphin_review_tab.set_groups(self.ds.dolphin_groups(),
                                                keep_current=True)
            self._dolphin_stack.setCurrentWidget(self._dolphin_review_tab)
        elif mode == "match":
            self._match_panel.show_dolphins(self.ds.dolphins())
            self._dolphin_stack.setCurrentWidget(self._match_panel)
        else:  # overview
            social_map = self.ds.dolphin_social_map()
            self._dolphin_tab.show_dolphins(self.ds.dolphins(),
                                            social=social_map)
            self._dolphin_stack.setCurrentWidget(self._dolphin_tab)

    def _refresh_dolphin_review(self):
        """DolphinID 有变时刷新 Review 模式左侧列表 (若已创建)。"""
        rt = getattr(self, "_dolphin_review_tab", None)
        if rt is None:
            return
        groups = self.ds.dolphin_groups()
        if groups:
            rt._labels = self.ds.dolphin_fin_labels()
            rt._clear_mark = self.ds.fin_clear_marks().__contains__
            rt.set_groups(groups, keep_current=True)

    def _match_assign(self, name, cls, fid):
        """Match Mode: 候选鳍拖入鳍格, 确认配对后刷新并保持当前海豚。"""
        self.ds.assign_dolphin_fin(name, fid, cls)
        self._match_panel.show_dolphins(self.ds.dolphins(), keep=name)
        self._refresh_dolphin_review()
        self._refresh_stats()

    def _match_unmatch(self, cls, fid):
        """Match Mode: 鳍图拖出格子, 确认不匹配后刷新。"""
        self.ds.unassign_dolphin_fin(fid, cls)
        self._match_panel.show_dolphins(
            self.ds.dolphins(), keep=self._match_panel.current_name())
        self._refresh_dolphin_review()
        self._refresh_stats()

    def _create_dolphin_from_dr(self, src_name):
        """Overview 模式: 把某海豚的 DR 鳍拖到空白处, 创建新海豚并分配该鳍。"""
        dr_fid = None
        for name, _, dr, _, _ in self.ds.dolphins():
            if name == src_name:
                dr_fid = dr
                break
        if dr_fid is None:
            return
        try:
            new_name = self.ds.new_dolphin()
            self.ds.assign_dolphin_fin(new_name, str(dr_fid), "DR")
        except Exception as e:
            QMessageBox.warning(self, "创建新海豚失败", str(e))
            return
        self._refresh_tabs()

    def _rename_dolphin_ids(self):
        """Dolphin 页顶栏 Rename ID: 按当前 Overview 顺序重新编号。
        若 Overview 网格尚无记录, 则按现有 DolphinID 数值顺序重命名。"""
        reply = QMessageBox.question(
            self, "Rename ID",
            "确定把所有 DolphinID 重新编号为 Dph001..DphN 吗?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        order = self._dolphin_tab.current_order()
        try:
            if order:
                self.ds.apply_dolphin_order(order)
            else:
                self.ds.rename_dolphin_ids()
        except Exception as e:
            QMessageBox.warning(self, "重命名失败", str(e))
            return
        self._refresh_tabs()

    def _assign_dolphin_to_fin_id(self, fin_id):
        """Fin ID 页 Review 模式右键菜单: 为指定 FinID 分配 DolphinID。

        弹出对话框列出已有 DolphinID 和 "<New Dolphin>" 选项;
        选择后写回 FIN_METAINFO.csv 并刷新所有 tab。
        """
        existing = sorted({str(d) for d in self.ds.dolphin_groups().keys()})
        items = existing + ["<New Dolphin>"]
        dph, ok = QInputDialog.getItem(
            self, "Assign DolphinID",
            "Assign DolphinID to %s:" % fin_id, items, 0, False)
        if not ok:
            return
        try:
            if dph == "<New Dolphin>":
                dph = self.ds.new_dolphin()
            self.ds.assign_dolphin_to_fin_id(fin_id, dph)
        except Exception as e:
            QMessageBox.warning(self, "Assign DolphinID failed", str(e))
            return
        self._refresh_tabs()

    def _reorder_dolphins_by_relationship(self):
        """按社会网络关系重新排序 Overview 模式的海豚网格 (不改 DolphinID)。"""
        try:
            order = self.ds.relationship_order()
        except Exception as e:
            QMessageBox.warning(self, "排序失败", str(e))
            return
        if not order:
            return
        dolph_map = {d[0]: d for d in self.ds.dolphins()}
        reordered = [dolph_map[name] for name in order if name in dolph_map]
        for d in self.ds.dolphins():
            if d[0] not in dolph_map or d[0] not in order:
                reordered.append(d)
        social_map = self.ds.dolphin_social_map()
        self._dolphin_tab.show_dolphins(reordered, social=social_map)

    def _swap_dolphin_dr(self, src, dst):
        """Dolphin 页拖 DR 格到其它海豚: 交换/移动 DR 配对后刷新本页。"""
        self.ds.swap_dolphin_dr(src, dst)
        social_map = self.ds.dolphin_social_map()
        self._dolphin_tab.show_dolphins(self.ds.dolphins(), social=social_map)
        self._refresh_dolphin_review()
        self._refresh_stats()

    def _refresh_stats(self):
        if self._stats_tab is not None:
            self._stats_tab.show_stats(
                self.ds.fin_id_counts(), self.ds.fin_id_shot_counts(),
                self.ds.aspect_counts(), self.ds.clearness_values(),
                self.ds.dolphin_shot_counts(),
                self.ds.social_network_data(), self.ds.traffic_data(),
                self.ds.timeline_data())

    @staticmethod
    def _qimage_to_pil(qimg):
        """QImage -> PIL RGB。整块拷贝后按行宽剥掉对齐填充 (逐行拷贝太慢)。"""
        from PIL import Image
        import numpy as np
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()  # >= w*3, 行尾有 4 字节对齐填充
        raw = qimg.constBits().asstring(bpl * (h - 1) + w * 3)
        # 前 h-1 行按 bpl 步长, 末行没有填充, 各自剥掉对齐填充后拼接
        out = np.empty((h, w * 3), np.uint8)
        out[:-1] = np.frombuffer(raw[:-w * 3], np.uint8) \
            .reshape(h - 1, bpl)[:, :w * 3]
        out[-1] = np.frombuffer(raw[-w * 3:], np.uint8)
        return Image.fromarray(out.reshape(h, w, 3), "RGB")

    @staticmethod
    def _make_crop_pil(qimg, box):
        """从查看器已解码的 QImage 裁出鳍图块并转 PIL。
        整图转换要搬 ~200MB (数百 ms), 只转裁剪块则近乎免费。"""
        w, h = qimg.width(), qimg.height()
        x0 = max(0, min(w, int(box[0])))
        y0 = max(0, min(h, int(box[1])))
        x1 = max(x0 + 1, min(w, int(box[2])))
        y1 = max(y0 + 1, min(h, int(box[3])))
        return MainWindow._qimage_to_pil(
            qimg.copy(x0, y0, x1 - x0, y1 - y0))

    def _edit_box(self, path, action, index, new_box=None, image=None):
        """全图查看器的鳍框编辑回调: 更新/添加鳍框, 并刷新相关 tab 缩略图。
        image 为查看器已解码的原图 (QImage), 避免数据集重复解码;
        深度特征/相似度重算较贵 (ONNX 推理), 放到后台线程执行。"""
        crop = (self._make_crop_pil(image, new_box)
                if image is not None and new_box is not None else None)
        if action == "update":
            old_boxes = self.ds.fin_boxes().get(path, [])
            if index is None or index >= len(old_boxes):
                return None
            old_box = old_boxes[index]
            if new_box is not None:
                self.ds.edit_fin_box(path, old_box, new_box, crop=crop)
                self._refresh_fin_boxes_on_tabs(path)
                self._enqueue_feature_job(
                    lambda: self.ds.replace_feature_for_box(path, new_box))
                return self.ds.fin_boxes().get(path, [])
            return None
        if action == "add" and new_box is not None:
            new_idx = self.ds.add_fin_box(path, new_box, crop=crop)
            if new_idx is None:
                return None
            shot_id = self.ds.fin_df.at[new_idx, "ShotID"]
            if not pd.isna(shot_id):
                self.ds.assign_track_ids_by_iou(shot_id)
            full = self.ds.fin_df.at[new_idx, "fullpath"]
            ident = self.ds.fin_df.at[new_idx, "identity"]
            # 新鳍先显示 (延迟一帧重建), 特征后台补算
            QTimer.singleShot(0, self._refresh_tabs)
            self._enqueue_feature_job(lambda: self._after_add_fin(full, ident))
            return self.ds.fin_boxes().get(path, [])
        return None

    def _after_add_fin(self, full, ident):
        """新增鳍的后台任务: 提特征并扩展相似度矩阵。"""
        feat = self.ds.extract_feature(full)
        if feat is None:
            return
        # 任务排队期间若又添加了新鳍, 最后一行已变, 扩展会错位, 放弃并提示
        if self.ds.fin_df.empty or \
                self.ds.fin_df["identity"].iloc[-1] != ident:
            raise RuntimeError(
                "连续添加鳍框导致相似度未更新, 请重新运行 Cluster Fin")
        self.ds._expand_similarity_for_new_fin(feat)

    def _enqueue_feature_job(self, fn):
        """排队一个后台特征重算任务 (串行执行, 避免 .npy 写竞争)。"""
        self._feat_jobs.append(fn)
        self._pump_feature_jobs()

    def _pump_feature_jobs(self):
        if self._feat_worker is not None or not self._feat_jobs:
            return
        worker = _FeatureWorker(self._feat_jobs.popleft(), self)
        worker.done.connect(self._on_feature_job_done)
        worker.failed.connect(self._on_feature_job_failed)
        worker.finished.connect(worker.deleteLater)
        self._feat_worker = worker
        worker.start()

    def _on_feature_job_done(self):
        self._feat_worker = None
        self._refresh_tabs()  # 相似度已更新, 刷新各页
        self._pump_feature_jobs()

    def _on_feature_job_failed(self, msg):
        self._feat_worker = None
        if self._dashboard is not None:
            self._dashboard._append_log("特征重算失败: %s" % msg,
                                        color="#ff6b6b")
        self._pump_feature_jobs()

    def _refresh_fin_boxes_on_tabs(self, path):
        """更新所有依赖 fin_boxes 的 tab 上的框。"""
        fin_tabs = {"Fin Aspect", "Fin Quality", "Fin ID", "Continuous Shots"}
        for title, tab, fn in self._group_tabs:
            if title in fin_tabs or title == "Social Structure":
                dolphin_id = (title == "Social Structure")
                tab._boxes[path] = self.ds.fin_boxes(
                    dolphin_id=dolphin_id).get(path, [])
                tab.grid.repaint_path(path)  # 只重绘, 图像文件未变

    def _rebuild_group_tab(self, tab, title, fn):
        """重建一个分组页 (Continuous Shots / Fin Aspect / Fin Quality / Fin ID / Social Structure)。"""
        ds = self.ds
        fin_tabs = {"Fin Aspect", "Fin Quality", "Fin ID"}
        groups = fn()
        if title == "Fin ID":  # 只显示图片数大于 1 的个体; "0" 未分类组始终保留
            groups = {k: v for k, v in groups.items() if len(v) > 1 or k == "0"}
            groups.setdefault("0", [])
            tab._best_mark = ds.fin_best_marks().__contains__
            tab._clear_mark = ds.fin_clear_marks().__contains__
        if not groups:
            tab.grid.show_images([])
            tab.group_list.clear()
            QListWidgetItem("(no data)", tab.group_list)
            tab._needs_refresh = False
            return
        tooltips = getattr(ds, "social_tooltips", None) \
            if title == "Social Structure" else None
        shot_boxes = ds.fin_boxes()
        shot_labels = ds.fin_shot_labels()
        tab._labels = shot_labels if title in fin_tabs else {}
        tab._full_boxes = shot_boxes if title in fin_tabs else {}
        if title == "Continuous Shots":
            boxes = shot_boxes
        elif title == "Social Structure":
            ds.dolphins()  # 保证 DolphinID 已分配
            boxes = ds.fin_boxes(dolphin_id=True)
        else:
            boxes = None
        tab.set_groups(groups, tooltips, keep_current=True, boxes=boxes)
        tab._on_group(None)  # 当前组名没变时信号不发, 强制用新标注重绘网格
        tab._needs_refresh = False

    def _rebuild_dolphin_review(self):
        """重建 Dolphin Review 模式的左侧列表与网格。"""
        rt = self._dolphin_review_tab
        groups = self.ds.dolphin_groups()
        if groups:
            rt._labels = self.ds.dolphin_fin_labels()
            rt._clear_mark = self.ds.fin_clear_marks().__contains__
            rt.set_groups(groups, keep_current=True)
            rt._on_group(None)
        rt._needs_refresh = False

    def _rebuild_stats(self):
        self._refresh_stats()
        if self._stats_tab is not None:
            self._stats_tab._needs_refresh = False

    def _rebuild_low_quality(self):
        """重建 Low Quality 页: 列出所有低质量原图。"""
        if self._low_quality_grid is None:
            return
        paths = self.ds.low_quality_photos()
        self._low_quality_grid.show_images(
            paths, boxes=self.ds.fin_boxes(),
            labels=self.ds.photo_shot_labels())
        if self._low_quality_tab is not None:
            self._low_quality_tab._needs_refresh = False

    def _refresh_tabs(self, origin=None):
        """元数据被拖拽改组更新后, 只刷新可见标签页; 隐藏页标记脏。
        切换标签页时再由 _on_tab_changed 重建。"""
        ds = self.ds
        dolphins = ds.dolphins()
        social_map = ds.dolphin_social_map()
        for title, tab, fn in self._group_tabs:
            if tab.isVisible():
                self._rebuild_group_tab(tab, title, fn)
            else:
                tab._needs_refresh = True
        # Dolphin 页 (overview / match / review)
        dpage = getattr(self, "_dolphin_page", None)
        if dpage is not None:
            if dpage.isVisible():
                cur = self._dolphin_stack.currentWidget()
                if cur is self._dolphin_tab:
                    self._dolphin_tab.show_dolphins(dolphins, social=social_map)
                elif cur is self._match_panel:
                    self._match_panel.show_dolphins(dolphins)
                elif cur is self._dolphin_review_tab:
                    self._rebuild_dolphin_review()
                for w in (self._dolphin_tab, self._match_panel,
                          self._dolphin_review_tab):
                    if w is not None:
                        w._needs_refresh = False
            else:
                for w in (self._dolphin_tab, self._match_panel,
                          self._dolphin_review_tab):
                    if w is not None:
                        w._needs_refresh = True
        # 统计页
        if self._stats_tab is not None:
            if self._stats_tab.isVisible():
                self._rebuild_stats()
            else:
                self._stats_tab._needs_refresh = True
        # 低质量页
        if self._low_quality_tab is not None:
            if self._low_quality_tab.isVisible():
                self._rebuild_low_quality()
            else:
                self._low_quality_tab._needs_refresh = True
