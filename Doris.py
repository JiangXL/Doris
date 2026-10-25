#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DorisGUI — 中华白海豚照片浏览与分组 GUI

用法:
  python DorisGUI.py [root_folder]
默认 root 为演示目录。
"""

import os
import sys

import PyQt5
from PyQt5.QtWidgets import QApplication, QMessageBox

# PyQt5 wheel bug: Qt derives its plugin path from the library location and
# mangles non-ASCII characters in it, leaving the plugin search path empty
# ("Could not find the Qt platform plugin ... in ''" / wayland shell
# integration not found). Point Qt at the whole bundled plugin tree
# explicitly before QApplication is created.
os.environ.setdefault(
    "QT_PLUGIN_PATH",
    os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins"),
)

from gui import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        root = sys.argv[1]
        if not os.path.isdir(root):
            QMessageBox.critical(None, "Doris", "目录不存在: %s" % root)
            sys.exit(1)
    else:
        root = "demo"
    win = MainWindow(root)
    win.show()
    sys.exit(app.exec_())
