# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


# torchvision >= 0.25 的扩展名为 _C_stable.so/.pyd、image_stable.so/.pyd，
# 不匹配 PyInstaller 默认的 lib*.so/*.dll 模式，需显式加宽搜索模式
_torchvision_libs = collect_dynamic_libs(
    'torchvision', search_patterns=['*.so', '*.so.*', '*.dll', '*.dylib', '*.pyd'])

a = Analysis(
    ['Step2_locate_fin.py'],
    pathex=[],
    binaries=collect_dynamic_libs('torch') + _torchvision_libs,
    datas=[
        ('models/fin_yolo_3class_PICWD_best.pt', 'models'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Step2_locate_fin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Step2_locate_fin',
)
