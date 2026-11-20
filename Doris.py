#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DorisGUI — 中华白海豚照片浏览与分组 GUI

用法:
  python DorisGUI.py [root_folder]
不带参数启动时无默认目录, 需在界面内选择 root_folder。
"""

import os
import sys

import PyQt6
from PyQt6.QtWidgets import QApplication, QMessageBox

# PyQt6 wheel bug: Qt derives its plugin path from the library location and
# mangles non-ASCII characters in it, leaving the plugin search path empty
# ("Could not find the Qt platform plugin ... in ''" / wayland shell
# integration not found). Point Qt at the whole bundled plugin tree
# explicitly before QApplication is created.
os.environ.setdefault(
    "QT_PLUGIN_PATH",
    os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins"),
)

from gui import MainWindow


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        root = sys.argv[1]
        if not os.path.isdir(root):
            QMessageBox.critical(None, "Doris", "目录不存在: %s" % root)
            sys.exit(1)
    else:
        root = None  # 未指定根目录: 启动后由用户通过 "Set Root Folder" 选择
    win = MainWindow(root)
    win.show()
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass
    sys.exit(app.exec())
