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
    """释放 PyTorch CUDA 缓存并触发垃圾回收。"""
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _locate_fin(root):
    import torch
    from Step2_locate_fin import FinLocator

    locator = FinLocator()
    cpu_num = os.cpu_count()
    if torch.cuda.is_available():
        batch = 64
        decode = int(cpu_num * 0.8)
        prefetch = 3 * cpu_num
    else:
        batch = int(cpu_num * 0.5)
        decode = int(cpu_num * 0.5)
        prefetch = int(cpu_num * 0.5)
    try:
        locator.locate_fin(root, batch_size=batch,
                           decode_workers=decode, prefetch_batches=prefetch)
    finally:
        del locator
        _release_gpu()


def _extract_features(root):
    from Step3_extract_fin_feature import FinFeatureExtractorGeneral

    extractor = FinFeatureExtractorGeneral()
    try:
        extractor.extract(root)
    finally:
        del extractor
        _release_gpu()


def _filter_quality(root):
    from Step4_filter_low_quality import FinQualityFilter

    filter_obj = FinQualityFilter(root_dir=root)
    filter_obj.auto_filter()


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
