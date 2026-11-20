# -*- coding: utf-8 -*-
"""Dashboard 流水线步骤的薄封装。

每个函数接收项目根目录（以及 Export PPT 所需的输出路径），实例化对应
Step*.py 中的类并调用入口方法。所有耗时操作在 DashboardTab 的 Worker 线程
中执行，因此这里直接同步调用即可。
"""

import os


def _scan_photos(root):
    from Step1a_scan_photo_exif import ExifScanner
    from Step1b_analyse_photo_jitter import BurstJitterAnalyzer

    scanner = ExifScanner()
    scanner.scan(root)

    analyzer = BurstJitterAnalyzer()
    analyzer.run(root, analyze=True)


def _release_gpu():
    """推理结束后触发垃圾回收（ONNX Runtime 无需手动释放 CUDA 缓存）。"""
    import gc
    gc.collect()


def _locate_fin(root):
    from onnx_infer import is_gpu_available
    from Step2_locate_fin import FinLocator

    cpu_num = os.cpu_count()
    if is_gpu_available():
        # 解码是瓶颈（实测 16 线程才打满），推理/后处理开销小，
        # 二级流水线即可，解码吃满全部核
        decode = cpu_num
        prefetch = 1
        intra_op = None
    else:
        # 收敛线程预算：decode + ORT intra-op ≈ 核数，避免超订
        decode = int(cpu_num * 0.25)
        prefetch = 1
        intra_op = max(1, cpu_num - decode)
    locator = FinLocator(intra_op_threads=intra_op)
    try:
        locator.locate_fin(root, decode_workers=decode,
                           prefetch_batches=prefetch)
    finally:
        del locator
        _release_gpu()


def _extract_features(root):
    from Step3_extract_fin_feature import (
        FinDeepFeatureExtractor, FinQualityAspectAnalyser)

    analyser = FinQualityAspectAnalyser()
    try:
        analyser.analyse(root)
    finally:
        analyser.close()
        del analyser
        _release_gpu()

    extractor = FinDeepFeatureExtractor()
    try:
        extractor.extract(root)
    finally:
        del extractor
        _release_gpu()


def _cluster(root):
    from Step5_sort_fin_automatic import FinSorter

    sorter = FinSorter(root_dir=root)
    sorter.sort()


def _export_database(root):
    from Step6a_export_database import DatabaseExporter

    meta_dir = os.path.join(root, "METAINFO")
    exporter = DatabaseExporter(
        root,
        os.path.join(meta_dir, "FIN_METAINFO.csv"),
        os.path.join(meta_dir, "PHOTO_METAINFO.csv"))
    exporter.export()


def _export_pptx(root, out_path):
    from Step6b_export_pptx import export_dolphin_pptx
    from doris_dataset import Dataset

    ds = Dataset(root)
    ds.dolphins()
    ds.social()
    export_dolphin_pptx(ds.dolphins(), out_path,
                        social=ds.dolphin_social_map())
