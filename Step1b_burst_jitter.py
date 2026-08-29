#!/usr/bin/env python
# coding: utf-8
"""计算连拍序列的帧间抖动（全局运动）

对 IMAGE_METAINFO.csv 中同一 shot_id 的连拍图像，按时间顺序
逐对估计相邻帧的全局运动（平移为主），三种方法的结果追加到 CSV：

1. 相位相关 (cv2.phaseCorrelate)：快速估计平移量
   -> pc_dx, pc_dy, pc_jitter, pc_resp
2. ORB 特征匹配 + 相似变换拟合：
   - 最小二乘（无 RANSAC）-> ftls_dx, ftls_dy, ftls_jitter
   - RANSAC（剔除外点）    -> ftrs_dx, ftrs_dy, ftrs_jitter, ftrs_inlier
3. ECC 对齐 (cv2.findTransformECC, 欧氏变换)
   -> ecc_dx, ecc_dy, ecc_jitter

每个 shot 的第一帧没有前一帧，三种结果均为 NaN。
"""
import os
import sys

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

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
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
                           if c.split("_")[0] in ("pc", "ftls", "ftrs", "ecc")]
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

    # ---- 方法 1：相位相关 ----
    def _phase_correlate(self, prev_f, curr_f, win):
        """返回 (dx, dy, jitter, response)，单位已换算回原图像素"""
        (dx, dy), resp = cv2.phaseCorrelate(prev_f, curr_f, win)
        return dx, dy, float(np.hypot(dx, dy)), resp

    # ---- 方法 2：ORB 特征匹配 ----
    def _match_features(self, prev_g, curr_g):
        """ORB 匹配相邻帧，返回匹配点对 (pts_prev, pts_curr)，失败返回 None"""
        kp1, des1 = self.orb.detectAndCompute(prev_g, None)
        kp2, des2 = self.orb.detectAndCompute(curr_g, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return None
        pts_prev = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts_prev, pts_curr

    @staticmethod
    def _fit_similarity_ls(pts_prev, pts_curr):
        """最小二乘拟合相似变换（无 RANSAC），返回 (dx, dy)

        x' = a*x - b*y + tx,  y' = b*x + a*y + ty
        """
        x, y = pts_prev[:, 0], pts_prev[:, 1]
        xp, yp = pts_curr[:, 0], pts_curr[:, 1]
        n = len(x)
        A = np.zeros((2 * n, 4))
        A[0::2] = np.stack([x, -y, np.ones(n), np.zeros(n)], axis=1)
        A[1::2] = np.stack([y, x, np.zeros(n), np.ones(n)], axis=1)
        # b 必须与 A 的行交错顺序一致: x'0, y'0, x'1, y'1, ...
        b = np.stack([xp, yp], axis=1).ravel()
        (a, b_, tx, ty), *_ = np.linalg.lstsq(A, b, rcond=None)
        return tx, ty

    @staticmethod
    def _fit_similarity_ransac(pts_prev, pts_curr):
        """RANSAC 拟合相似变换，返回 (dx, dy, inlier_ratio)，失败返回 None"""
        M, inliers = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return None
        ratio = float(inliers.sum()) / len(inliers) if inliers is not None else 0.0
        return M[0, 2], M[1, 2], ratio

    # ---- 方法 3：ECC 对齐 ----
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
        """分析一对相邻帧，返回三种方法的结果 dict（单位：原图像素）"""
        row = {}
        s = curr["inv_scale"]  # 1/scale，换算回原图分辨率

        dx, dy, jitter, resp = self._phase_correlate(
            prev["float"], curr["float"], curr["win"])
        row.update(pc_dx=dx * s, pc_dy=dy * s,
                   pc_jitter=jitter * s, pc_resp=resp)

        matched = self._match_features(prev["gray"], curr["gray"])
        if matched is not None:
            pts_prev, pts_curr = matched
            dx, dy = self._fit_similarity_ls(pts_prev, pts_curr)
            row.update(ftls_dx=dx * s, ftls_dy=dy * s,
                       ftls_jitter=float(np.hypot(dx, dy)) * s)
            ransac = self._fit_similarity_ransac(pts_prev, pts_curr)
            if ransac is not None:
                dx, dy, ratio = ransac
                row.update(ftrs_dx=dx * s, ftrs_dy=dy * s,
                           ftrs_jitter=float(np.hypot(dx, dy)) * s,
                           ftrs_inlier=ratio)

        ecc = self._ecc(prev["gray"], curr["gray"],
                        init_shift=(row["pc_dx"] / s, row["pc_dy"] / s))
        if ecc is not None:
            dx, dy = ecc
            row.update(ecc_dx=dx * s, ecc_dy=dy * s,
                       ecc_jitter=float(np.hypot(dx, dy)) * s)
        return row

    def analyze(self):
        """按 shot_id 分组、按时间排序，逐对估计帧间抖动"""
        df = self.metainfo
        ts = pd.to_datetime(df["datetimesec"].str.strip(),
                            format="%Y:%m:%d %H:%M:%S.%f")
        results = {}

        for shot_id, group in df.groupby("shot_id"):
            group = group.loc[ts.loc[group.index].sort_values().index]
            if len(group) < 2:
                continue
            frames = []
            for name in group["orig_img_name"]:
                gray, scale = self._load_gray(name)
                f32 = np.float32(gray)
                win = cv2.createHanningWindow(f32.shape[::-1], cv2.CV_32F)
                frames.append({"name": name, "gray": gray, "float": f32,
                               "win": win, "inv_scale": 1.0 / scale})
            for i in tqdm(range(1, len(frames)),
                          desc="shot %d (%d 帧)" % (shot_id, len(frames))):
                results[frames[i]["name"]] = self._analyze_pair(
                    frames[i - 1], frames[i])

        result_df = pd.DataFrame(results).T
        self.metainfo = df.merge(
            result_df, left_on="orig_img_name", right_index=True, how="left")

    def print_summary(self):
        """打印各方法抖动统计，及 RANSAC 与否的差异"""
        df = self.metainfo
        print("\n===== 帧间抖动统计（像素）=====")
        for prefix, label in [("pc", "相位相关"),
                              ("ftls", "特征+最小二乘"),
                              ("ftrs", "特征+RANSAC"),
                              ("ecc", "ECC")]:
            col = prefix + "_jitter"
            if col in df:
                v = df[col].dropna()
                print("%-12s n=%-6d mean=%7.2f  median=%7.2f  max=%8.2f"
                      % (label, len(v), v.mean(), v.median(), v.max()))
        both = df[["ftls_jitter", "ftrs_jitter"]].dropna()
        if len(both):
            diff = (both["ftls_jitter"] - both["ftrs_jitter"]).abs()
            print("\nRANSAC 与否的差异 |ftls - ftrs|："
                  "mean=%.2f  median=%.2f  max=%.2f 像素"
                  % (diff.mean(), diff.median(), diff.max()))
            print("RANSAC 平均内点比例：%.2f" % df["ftrs_inlier"].mean())

    def save(self):
        self.metainfo.to_csv(self.meta_path, index=False)
        print("结果已追加到 %s" % self.meta_path)

    def plot_results(self):
        """可视化抖动结果，四联图保存到 METAINFO/JITTER.png

        1. 各方法的帧间抖动时序曲线（按 shot 分界）
        2. RANSAC 位移矢量场 (dx, dy)，按 shot 着色
        3. 各方法抖动幅度分布直方图
        4. 最小二乘 vs RANSAC 抖动对比（y=x 参考线）
        """
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        df = self.metainfo
        methods = [("pc", "Phase correlation", "tab:blue"),
                   ("ftls", "Feature + LS (no RANSAC)", "tab:orange"),
                   ("ftrs", "Feature + RANSAC", "tab:green"),
                   ("ecc", "ECC", "tab:red")]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # 1. 时序曲线：按 shot_id + 时间排序后顺序排列，竖线标出 shot 分界
        ax = axes[0, 0]
        ts = pd.to_datetime(df["datetimesec"].str.strip(),
                            format="%Y:%m:%d %H:%M:%S.%f")
        order = df.assign(_ts=ts).sort_values(["shot_id", "_ts"])
        x = np.arange(len(order))
        for prefix, label, color in methods:
            ax.plot(x, order[prefix + "_jitter"].values,
                    label=label, color=color, alpha=0.8, lw=1)
        boundaries = np.flatnonzero(np.diff(order["shot_id"].values)) + 0.5
        for b in boundaries:
            ax.axvline(b, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("Frame (grouped by shot)")
        ax.set_ylabel("Inter-frame jitter (px)")
        ax.set_title("Jitter time series (dashed lines = shot boundaries)")
        ax.legend(fontsize=8)

        # 2. 位移矢量场：从原点出发的 (dx, dy)，颜色区分 shot
        ax = axes[0, 1]
        v = order[["ftrs_dx", "ftrs_dy", "shot_id"]].dropna()
        scatter = ax.scatter(v["ftrs_dx"], v["ftrs_dy"],
                             c=v["shot_id"], cmap="tab20", s=15, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("dx (px)")
        ax.set_ylabel("dy (px)")
        ax.set_title("Displacement vectors (RANSAC), colored by shot")
        plt.colorbar(scatter, ax=ax, label="shot_id")

        # 3. 抖动幅度分布直方图
        ax = axes[1, 0]
        for prefix, label, color in methods:
            vals = df[prefix + "_jitter"].dropna()
            if len(vals):
                ax.hist(vals, bins=50, alpha=0.5, label=label, color=color)
        ax.set_xlabel("Inter-frame jitter (px)")
        ax.set_ylabel("Frame count")
        ax.set_title("Jitter distribution")
        ax.legend(fontsize=8)

        # 4. 最小二乘 vs RANSAC：偏离 y=x 的点即 RANSAC 剔除外点起作用的帧
        ax = axes[1, 1]
        both = df[["ftls_jitter", "ftrs_jitter"]].dropna()
        ax.scatter(both["ftrs_jitter"], both["ftls_jitter"], s=15, alpha=0.7)
        if len(both):
            lim = max(both.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
        ax.set_xlabel("Feature + RANSAC jitter (px)")
        ax.set_ylabel("Feature + LS jitter (px)")
        ax.set_title("Effect of RANSAC (points above line = LS pulled by outliers)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(self.root_dir, "METAINFO", "JITTER.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print("可视化已保存到 %s" % out_path)

    def run(self, analyze=True):
        self.load_metainfo(drop_old_results=analyze)
        if analyze:
            self.analyze()
            self.save()
        self.print_summary()
        self.plot_results()


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
