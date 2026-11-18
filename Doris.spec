# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


# 推理经 ONNX Runtime（onnx_infer.py），torch/torchvision/timm/ultralytics 为开发专用依赖，
# 打包时通过 excludes 剔除；onnxruntime 的动态库（含 CUDA EP 库）显式收集。
a = Analysis(
        ['Doris.py'],
    pathex=[],
    binaries= collect_dynamic_libs('onnxruntime')
                + collect_dynamic_libs('pyexiv2'),
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
