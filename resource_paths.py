# coding: utf-8
"""资源路径解析：打包后从 PyInstaller 的 _MEIPASS 取资源，源码运行时取项目根目录。"""
import os
import sys


def resource_path(relative_path: str) -> str:
    """返回资源文件绝对路径，兼容 PyInstaller 冻结环境与源码运行。"""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base, relative_path)
