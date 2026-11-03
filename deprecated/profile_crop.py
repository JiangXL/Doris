#!/usr/bin/env python
# coding: utf-8
"""Step1_crop_fin 分阶段计时 profiling 脚本（不改主代码）。
用法: python profile_crop.py <root_dir> [限制图像数]
"""
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm

from Step1_crop_fin import FinCropper


def main():
    root_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    batch_size = 16

    cropper = FinCropper()

    # --- 与 FinCropper.crop() 相同的数据准备 ---
    root_dir = root_dir.rstrip(os.sep)
    cropper.root_dir = root_dir
    cropper.meta_path = os.path.join(root_dir, cropper.meta_dir)
    os.makedirs(cropper.meta_path, exist_ok=True)
    cropper.fin_save_dir = os.path.join(root_dir, "FIN")
    os.makedirs(cropper.fin_save_dir, exist_ok=True)
    img_info_csv = os.path.join(cropper.meta_path, "IMAGE_METAINFO.csv")
    img_info_df = pd.read_csv(img_info_csv, sep=",", encoding="utf-8")
    jpg_paths = [f"{root_dir}/{name}" for name in img_info_df.orig_img_name]
    if limit:
        jpg_paths = jpg_paths[:limit]

    # --- 计时累加器 ---
    t_wait = 0.0       # 等待预取解码完成
    t_infer = 0.0      # YOLO 调用（含 letterbox 预处理 + GPU 前向 + NMS）
    t_process = 0.0    # 后处理（去重/裁剪/写图）
    t_blur = 0.0       # 批量模糊检测
    n_img = 0
    # ultralytics 内部 speed 统计（ms/张）: preprocess/inference/postprocess
    speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    batches = [jpg_paths[i:i + batch_size] for i in range(0, len(jpg_paths), batch_size)]
    prefetch_batches = 2
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as pool:
        inflight = deque()
        for j in range(min(prefetch_batches + 1, len(batches))):
            inflight.append([pool.submit(cropper._read_image, p) for p in batches[j]])
        for bi in tqdm(range(len(batches)), desc="Profiling"):
            pending = inflight.popleft()
            nxt = bi + prefetch_batches + 1
            if nxt < len(batches):
                inflight.append([pool.submit(cropper._read_image, p) for p in batches[nxt]])
            t0 = time.perf_counter()
            valid_paths, imgs = [], []
            for fut in pending:
                p, img = fut.result()
                if img is None:
                    continue
                valid_paths.append(p)
                imgs.append(img)
            t_wait += time.perf_counter() - t0

            if not imgs:
                continue

            t0 = time.perf_counter()
            results = cropper.fin_detector(imgs, verbose=False)
            t_infer += time.perf_counter() - t0

            for r in results:
                for k in speed:
                    speed[k] += r.speed.get(k, 0.0)
            n_img += len(results)

            t0 = time.perf_counter()
            batch_rows, batch_crops = [], []
            for jpg_path, result in zip(valid_paths, results):
                rows, crops = cropper._process_result(jpg_path, result)
                batch_rows.extend(rows)
                batch_crops.extend(crops)
            t_process += time.perf_counter() - t0

            t0 = time.perf_counter()
            if batch_crops:
                cropper.blur_detector.predict_batch(batch_crops)
            t_blur += time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    print("\n===== Profiling 结果 =====")
    print(f"图像总数: {n_img}, batch_size: {batch_size}")
    print(f"总耗时:            {t_total:8.2f} s  ({t_total/max(n_img,1)*1000:7.1f} ms/张)")
    print(f"等待预取解码:      {t_wait:8.2f} s  ({t_wait/t_total*100:5.1f}%)")
    print(f"YOLO 调用(含预处理):{t_infer:7.2f} s  ({t_infer/t_total*100:5.1f}%)")
    print(f"后处理(去重/裁剪/写图):{t_process:5.2f} s  ({t_process/t_total*100:5.1f}%)")
    print(f"批量模糊检测:      {t_blur:8.2f} s  ({t_blur/t_total*100:5.1f}%)")
    print("\nultralytics speed 分解 (平均每张 ms):")
    for k, v in speed.items():
        print(f"  {k:12s} {v/max(n_img,1):8.2f} ms/张  (累计 {v/1000:.2f} s)")


if __name__ == "__main__":
    main()
