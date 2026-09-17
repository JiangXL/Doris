#!/usr/bin/env python
# coding: utf-8
"""计算连拍序列的帧间抖动（全局运动）

对 IMAGE_METAINFO.csv 中同一 shot_id 的连拍图像，按时间顺序
逐对估计相邻帧的全局运动（平移为主），两种方法的结果追加到 CSV：

1. 相位相关 (cv2.phaseCorrelate)：快速估计平移量
   -> pc_dx, pc_dy, pc_jitter, pc_resp
2. ECC 对齐 (cv2.findTransformECC, 欧氏变换)
   -> ecc_dx, ecc_dy, ecc_jitter

每个 shot 的第一帧没有前一帧，两种结果均为 NaN。
"""
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class BurstJitterAnalyzer:
    """连拍序列帧间抖动分析器"""

    def __init__(self, root_dir, max_dim=1000):
        """
        Args:
            root_dir: 图像集根目录（含 METAINFO/IMAGE_METAINFO.csv）
            max_dim: 估计运动时降采样后的最长边像素数
        """
        self.root_dir = root_dir
        self.max_dim = max_dim
        self.meta_path = os.path.join(root_dir, "METAINFO", "IMAGE_METAINFO.csv")
        self.metainfo = None

        self.ecc_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

    def load_metainfo(self, drop_old_results=True):
        """加载 IMAGE_METAINFO.csv

        Args:
            drop_old_results: 重新计算时丢弃上次运行的旧抖动列；
                仅画图（--plot）时保留
        """
        self.metainfo = pd.read_csv(self.meta_path)
        if drop_old_results:
            jitter_cols = [c for c in self.metainfo.columns
                           if c.split("_")[0] in ("pc", "ecc")]
            self.metainfo = self.metainfo.drop(columns=jitter_cols)

    def _load_gray(self, img_name):
        """读取图像并降采样为灰度图，返回 (gray, scale)"""
        img = cv2.imread(os.path.join(self.root_dir, img_name),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("无法读取图像: %s" % img_name)
        scale = self.max_dim / max(img.shape)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            scale = 1.0
        return img, scale

    def _load_frame(self, img_name):
        """读取并预处理单帧（供预取线程调用，cv2 解码释放 GIL）

        返回 (img_name, frame_dict)，frame_dict 含灰度图、float32 副本、
        Hanning 窗和分辨率换算系数。
        """
        gray, scale = self._load_gray(img_name)
        f32 = np.float32(gray)
        win = cv2.createHanningWindow(f32.shape[::-1], cv2.CV_32F)
        frame = {"name": img_name, "gray": gray, "float": f32,
                 "win": win, "inv_scale": 1.0 / scale}
        return img_name, frame

    # ---- 方法 1：相位相关 ----
    def _phase_correlate(self, prev_f, curr_f, win):
        """返回 (dx, dy, jitter, response)，单位已换算回原图像素"""
        (dx, dy), resp = cv2.phaseCorrelate(prev_f, curr_f, win)
        return dx, dy, float(np.hypot(dx, dy)), resp

    # ---- 方法 2：ECC 对齐 ----
    def _ecc(self, prev_g, curr_g, init_shift=(0.0, 0.0)):
        """ECC 欧氏变换估计，返回 (dx, dy)，失败返回 None

        用相位相关的平移量初始化 warp，避免大位移时不收敛。
        """
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 2], warp[1, 2] = init_shift
        try:
            _, warp = cv2.findTransformECC(
                prev_g.astype(np.float32) / 255.0,
                curr_g.astype(np.float32) / 255.0,
                warp, cv2.MOTION_EUCLIDEAN, self.ecc_criteria)
        except cv2.error:
            return None
        return float(warp[0, 2]), float(warp[1, 2])

    def _analyze_pair(self, prev, curr):
        """分析一对相邻帧，返回两种方法的结果 dict（单位：原图像素）"""
        row = {}
        s = curr["inv_scale"]  # 1/scale，换算回原图分辨率

        dx, dy, jitter, resp = self._phase_correlate(
            prev["float"], curr["float"], curr["win"])
        row.update(pc_dx=dx * s, pc_dy=dy * s,
                   pc_jitter=jitter * s, pc_resp=resp)

        ecc = self._ecc(prev["gray"], curr["gray"],
                        init_shift=(row["pc_dx"] / s, row["pc_dy"] / s))
        if ecc is not None:
            dx, dy = ecc
            row.update(ecc_dx=dx * s, ecc_dy=dy * s,
                       ecc_jitter=float(np.hypot(dx, dy)) * s)
        return row

    def analyze(self, decode_workers=8, analysis_workers=8,
                prefetch_frames=16, inflight_pairs=16):
        """按 shot_id 分组、按时间排序，逐对估计帧间抖动（并行加速）

        参考 Step1_crop_fin.py 的预取模式：
        - 解码线程池：预取窗口内的帧在后台并行解码+预处理
          （cv2.imread/resize 释放 GIL，多线程有效）
        - 分析线程池：相位相关/ECC 均为 OpenCV C++ 实现、释放 GIL，
          相邻帧对两两独立，并行分析

        Args:
            decode_workers: JPEG 解码/预处理线程数
            analysis_workers: 帧对分析线程数
            prefetch_frames: 预取窗口大小（帧数），限制内存占用
            inflight_pairs: 最多同时在飞的分析任务数，限制内存占用
        """
        df = self.metainfo
        ts = pd.to_datetime(df["datetimesec"].str.strip(),
                            format="%Y:%m:%d %H:%M:%S.%f")
        order = df.assign(_ts=ts).sort_values(["shot_id", "_ts"])
        names = order["orig_img_name"].tolist()
        shots = order["shot_id"].tolist()
        # 同 shot 内的相邻帧对（按全局顺序处理，保证预取窗口滑动有效）
        pairs = [(i - 1, i) for i in range(1, len(names))
                 if shots[i] == shots[i - 1]]

        results = {}
        with ThreadPoolExecutor(decode_workers) as load_pool, \
                ThreadPoolExecutor(analysis_workers) as analysis_pool:
            cache = {}           # name -> Future(预处理帧)
            analysis_futs = deque()  # (curr_name, Future(分析结果))
            next_submit = 0      # 下一个待预取的帧下标

            for i_prev, i_curr in tqdm(pairs, desc="Analyzing pairs"):
                # 补充预取窗口：当前对之后的 prefetch_frames 帧在后台解码
                while next_submit <= min(i_curr + prefetch_frames,
                                         len(names) - 1):
                    name = names[next_submit]
                    if name not in cache:
                        cache[name] = load_pool.submit(
                            self._load_frame, name)
                    next_submit += 1

                prev = cache[names[i_prev]].result()[1]
                curr = cache[names[i_curr]].result()[1]
                # 帧 i_prev 只被当前对和前一对使用，此后可从缓存移除
                del cache[names[i_prev]]
                analysis_futs.append((
                    names[i_curr],
                    analysis_pool.submit(self._analyze_pair, prev, curr)))

                # 收取已完成的分析结果，限制在飞任务数
                while len(analysis_futs) >= inflight_pairs:
                    name, fut = analysis_futs.popleft()
                    results[name] = fut.result()
            while analysis_futs:
                name, fut = analysis_futs.popleft()
                results[name] = fut.result()

        result_df = pd.DataFrame(results).T
        self.metainfo = df.merge(
            result_df, left_on="orig_img_name", right_index=True, how="left")

    def save(self):
        self.metainfo.to_csv(self.meta_path, index=False)
        print("结果已追加到 %s" % self.meta_path)

    def run(self, analyze=True):
        self.load_metainfo(drop_old_results=analyze)
        if analyze:
            self.analyze()
            self.save()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
        # --plot: 跳过计算，直接用 CSV 中已有结果画图
        analyze = "--plot" not in sys.argv[2:]
    else:
        print("用法: python3 Step1b_burst_jitter.py <root_dir> [--plot]")
        sys.exit(1)
    analyzer = BurstJitterAnalyzer(root_dir=root_dir)
    analyzer.run(analyze=analyze)
