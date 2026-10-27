# -*- coding: utf-8 -*-
"""主窗口：组装所有标签页、数据集加载与跨页刷新。"""

import os

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QMessageBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QMenu, QStackedWidget,QListWidgetItem
)
from PyQt5.QtCore import Qt

from doris_dataset import Dataset
from .tabs import DashboardTab, GroupedTab, StatisticsTab
from .dolphin import DolphinTab, MatchPanel, CandidateList
from .common import ImageGrid

class MainWindow(QMainWindow):
    def __init__(self, root):
        super().__init__()
        self.setWindowTitle("Doris v0.2 — %s" % root)
        #self.resize(1280, 800)

        self.tabs = QTabWidget()
        # 切到统计页时惰性重算, 保证任何改组后看到的都是最新数据
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self.load(root)

    def _on_tab_changed(self, i):
        """切换标签页时, 若目标页标记为脏则重建。"""
        w = self.tabs.widget(i)
        if w is self._stats_tab:
            if getattr(self._stats_tab, "_needs_refresh", False):
                self._rebuild_stats()
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
                    self._dolphin_tab.show_dolphins(self.ds.dolphins())
                elif cur is self._match_panel:
                    self._match_panel.show_dolphins(self.ds.dolphins())
                elif cur is self._dolphin_review_tab:
                    self._rebuild_dolphin_review()
                cur._needs_refresh = False

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
        meta_dir = os.path.join(root, "METAINFO")
        self._required_files = {
            "PHOTO_METAINFO.csv": os.path.join(meta_dir, "PHOTO_METAINFO.csv"),
            "FIN_METAINFO.csv": os.path.join(meta_dir, "FIN_METAINFO.csv"),
            "FIN_DEEPFEATURES": os.path.join(meta_dir, "FIN_DEEPFEATURES"),
        }
        self._tab_requires = {}  # tab index -> prerequisite key
        self._group_tabs = []
        self._dolphin_tab = None
        self._stats_tab = None
        self.tabs.clear()

        # 0. Dashboard 页: 处理流程入口 (参考 dashboard.ui); 选择根目录后重载
        self._dashboard = DashboardTab()
        self._dashboard.root_changed.connect(self.load)
        self._dashboard.scan_finished.connect(lambda: self.load(self.root))
        self._dashboard.locate_finished.connect(lambda: self.load(self.root))
        self._dashboard.cluster_finished.connect(lambda: self.load(self.root))
        self._dashboard.set_root(root)
        self.tabs.addTab(self._dashboard, "Dashboard")

        # Dataset 可容忍缺失文件 (空 dataframe), 因此始终创建, 用于初始化各页
        ds = Dataset(root)
        self.ds = ds

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
            ("Fin ID", ds.fin_ids, ds.move_selected_to_fin_id),
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
            idx = self.tabs.addTab(tab, title)
            self._group_tabs.append((title, tab, fn))
            # Continuous Shots 依赖原图元数据; 其余三个分组页依赖 FIN_METAINFO.csv
            self._tab_requires[idx] = ("PHOTO_METAINFO.csv"
                                       if title == "Continuous Shots"
                                       else "FIN_METAINFO.csv")

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
        self._btn_rename_dph = QPushButton("Rename ID")
        self._btn_rename_dph.setToolTip(
            "把所有 DolphinID 重新编号为 Dph001..DphN (N=当前海豚总数)")
        self._btn_rename_dph.clicked.connect(self._rename_dolphin_ids)
        top.addWidget(self._btn_rename_dph)
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
            clear_mark=ds.fin_clear_marks().__contains__,
            best_marker=ds.mark_best_fins,
            best_in_fin_provider=ds.fin_best_in_fin_marks,
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
        self._tab_requires[idx] = "FIN_METAINFO.csv"

        # 6. 社会结构页 (只读): 整组连拍 (shot_id) 内出现多于一个 DolphinID
        #    的 shot, 同一组 DolphinID 的多个 shot 合并为一组 (NN001...),
        #    组内只收同一张原图含多个海豚鳍的图;
        #    组名写回 IMAGE/FIN_METAINFO.csv 的 SocialStructure 列;
        #    仿 Continuous Shots 页在原图上画鳍框, 标注 FinID 和 DolphinID
        #    (须排在 Dolphin 页后, ds.dolphins() 已分配 DolphinID 列)
        tab_social = GroupedTab(mover=None, on_moved=self._refresh_tabs,
                                box_editor=self._edit_box)
        groups = ds.social()
        if groups:
            tab_social.set_groups(groups, ds.social_tooltips,
                                  boxes=ds.fin_boxes(dolphin_id=True))
        else:
            tab_social.grid.show_images([])
            QListWidgetItem("(no data)", tab_social.group_list)
        idx = self.tabs.addTab(tab_social, "Social Structure")
        self._group_tabs.append(("Social Structure", tab_social, ds.social))
        self._tab_requires[idx] = "FIN_METAINFO.csv"

        # 7. 统计页: FinID 鳍图数柱状图 + 部位占比饼图
        self._stats_tab = StatisticsTab()
        self._stats_tab.set_root(root)
        self._stats_tab.show_stats(
            ds.fin_id_counts(), ds.fin_id_shot_counts(), ds.aspect_counts(),
            ds.clearness_values(), ds.dolphin_shot_counts(),
            ds.social_network_data())
        idx = self.tabs.addTab(self._stats_tab, "Statistics")
        self._tab_requires[idx] = "FIN_METAINFO.csv"

        # 根据已有元数据启用/禁用对应标签页; 流水线完成后信号会再次刷新
        self._refresh_tab_enablement()

    def _refresh_tab_enablement(self):
        """根据 METAINFO 下文件是否存在, 启用或禁用各标签页。"""
        available = {k: os.path.isfile(v)
                     for k, v in getattr(self, "_required_files", {}).items()}
        for idx, req in getattr(self, "_tab_requires", {}).items():
            self.tabs.setTabEnabled(idx, available.get(req, False))


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

    def _new_dolphin(self):
        """新建空海豚并刷新 Dolphin 页。"""
        self.ds.new_dolphin()
        self._dolphin_tab.show_dolphins(self.ds.dolphins())
        self._refresh_dolphin_review()

    def _rename_dolphin_ids(self):
        """Dolphin 页顶栏 Rename ID: 重新编号所有 DolphinID 并刷新界面。"""
        reply = QMessageBox.question(
            self, "Rename ID",
            "确定把所有 DolphinID 重新编号为 Dph001..DphN 吗?")
        if reply != QMessageBox.Yes:
            return
        try:
            self.ds.rename_dolphin_ids()
        except Exception as e:
            QMessageBox.warning(self, "重命名失败", str(e))
            return
        self._dolphin_tab.show_dolphins(self.ds.dolphins())
        self._refresh_dolphin_review()
        self._refresh_stats()
        # Social Structure / Dolphin Review 等脏页标记刷新
        self._refresh_tabs()

    def _swap_dolphin_dr(self, src, dst):
        """Dolphin 页拖 DR 格到其它海豚: 交换/移动 DR 配对后刷新本页。"""
        self.ds.swap_dolphin_dr(src, dst)
        self._dolphin_tab.show_dolphins(self.ds.dolphins())
        self._refresh_dolphin_review()
        self._refresh_stats()

    def _refresh_stats(self):
        if self._stats_tab is not None:
            self._stats_tab.show_stats(
                self.ds.fin_id_counts(), self.ds.fin_id_shot_counts(),
                self.ds.aspect_counts(), self.ds.clearness_values(),
                self.ds.dolphin_shot_counts(),
                self.ds.social_network_data())

    def _edit_box(self, path, action, index, new_box=None):
        """全图查看器的鳍框编辑回调: 更新 CSV 与裁剪图, 并刷新相关 tab 缩略图。"""
        old_boxes = self.ds.fin_boxes().get(path, [])
        if index >= len(old_boxes):
            return None
        old_box = old_boxes[index]
        if action == "update" and new_box is not None:
            self.ds.edit_fin_box(path, old_box, new_box)
            for title, tab, fn in self._group_tabs:
                if title in ("Continuous Shots", "Social Structure"):
                    dolphin_id = (title == "Social Structure")
                    tab._boxes[path] = self.ds.fin_boxes(
                        dolphin_id=dolphin_id).get(path, [])
                    tab.grid.refresh_path(path)
            return self.ds.fin_boxes().get(path, [])
        return None

    def _rebuild_group_tab(self, tab, title, fn):
        """重建一个分组页 (Continuous Shots / Fin Aspect / Blur / Fin ID / Social Structure)。"""
        ds = self.ds
        fin_tabs = {"Fin Aspect", "Blur", "Fin ID"}
        groups = fn()
        if title == "Fin ID":  # 只显示图片数大于 1 的个体
            groups = {k: v for k, v in groups.items() if len(v) > 1}
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

    def _refresh_tabs(self, origin=None):
        """元数据被拖拽改组更新后, 只刷新可见标签页; 隐藏页标记脏。
        切换标签页时再由 _on_tab_changed 重建。"""
        ds = self.ds
        dolphins = ds.dolphins()
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
                    self._dolphin_tab.show_dolphins(dolphins)
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
