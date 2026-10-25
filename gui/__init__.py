# -*- coding: utf-8 -*-
"""Doris GUI 组件包。

入口仍可直接使用 `from gui import MainWindow`。
"""

from .main_window import MainWindow
from .tabs import DashboardTab, GroupedTab, StatisticsTab
from .dolphin import DolphinTab, MatchPanel
from .common import ImageGrid, ThumbLoader, DolphinLoader
