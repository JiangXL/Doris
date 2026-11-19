# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules, get_package_paths)


# 推理经 ONNX Runtime（onnx_infer.py），torch/torchvision/timm/ultralytics 为开发专用依赖，
# 打包时通过 excludes 剔除；onnxruntime 的动态库（含 CUDA EP 库）显式收集。
# Windows 上 ORT 的 CUDA EP 在运行时按 DLL 名动态加载 CUDA 13 运行库（onnx_infer
# 调 onnxruntime.preload_dlls，经 PyInstaller 的 ctypes hook 在 _MEIPASS 包根按名解析），
# 其中 cudart/cublas/cublasLt 为必需，cuDNN 9 可选。CUDA 13 起 NVIDIA 把各组件合并为
# 单个 nvidia/cu13 包（DLL 在 bin/x86_64/ 子目录）。这里统一从 pip 的 nvidia/cu13 收集
# 到包根，并在 Analysis 后剔除 bindepend 从构建机系统 CUDA Toolkit（PATH）捡到的同名
# 副本，使未装 CUDA Toolkit 的机器也能打出自包含的包（Linux 走 LD_LIBRARY_PATH 不需打包）。
CUDA_DLL_NAMES = ('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')

_nv_bins = []
if sys.platform == 'win32':
    _nv_bins += collect_dynamic_libs('nvidia.cudnn')
    _, _cu13_dir = get_package_paths('nvidia.cu13')
    for _name in CUDA_DLL_NAMES:
        _path = os.path.join(_cu13_dir, 'bin', 'x86_64', _name)
        if not os.path.isfile(_path):
            raise RuntimeError(
                f'缺少 CUDA 运行库 {_path}；请确认已安装 requirements-runtime.txt 中的 '
                'onnxruntime-gpu[cuda,cudnn]（含 nvidia-cu13 系列依赖）')
        _nv_bins.append((_path, '.'))

a = Analysis(
        ['Doris.py'],
    pathex=[],
    binaries= collect_dynamic_libs('onnxruntime')
                + collect_dynamic_libs('pyexiv2')
                + _nv_bins,
    datas=[
        ('models/fin_yolo_3class_PICWD_best.onnx', 'models'),
        ('models/fin_quality_detection_resnet101_final.onnx', 'models'),
        ('models/fin_aspect_classification_resnet101_final.onnx', 'models'),
        ('models/megadescriptor_dolphin_ft.onnx', 'models'),
    ],
    hiddenimports=[
        'Step1a_scan_photo_exif',
        'Step1b_analyse_photo_jitter',
        'Step2_locate_fin',
        'Step3_extract_fin_feature',
        'Step5_sort_fin_automatic',
        'Step6a_export_database',
        'Step6b_export_pptx',
        'onnx_infer',
        'check_duplicate_detections',
        'KalmanBoxTracker',
        'resource_paths',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6', 'jupyter', 'notebook', 'pytest',
              'torch', 'torchvision', 'timm', 'ultralytics'],
    noarchive=False,
    optimize=0,
)

# 剔除 bindepend 从系统 CUDA Toolkit(PATH) 解析到的同名 CUDA DLL（与上面 pip 来源的
# 副本重复，且会让未装 CUDA Toolkit 的构建机打出缺库的包）。
if sys.platform == 'win32':
    _cuda_names = {n.lower() for n in CUDA_DLL_NAMES}
    a.binaries = [b for b in a.binaries
                  if os.path.basename(b[0]).lower() not in _cuda_names
                  or 'site-packages' in b[1]]

pyz = PYZ(a.pure)

splash = Splash(
    'Delphinus_IAU.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(40, 1700),
    text_size=48,
    text_default='Booting',
    text_color='white',
    min_time=0,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    exclude_binaries=True,
    name='Doris',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    splash.binaries,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Doris',
)
