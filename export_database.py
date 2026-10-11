#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DatabaseExporter 接受 FIN_METAINFO.csv / IMAGE_METAINFO.csv 的文件路径
或已加载的 pandas DataFrame 作为源元数据:
  export_dolphin_images:     按 DolphinID 分组 (丢弃缺失行), 把 clear==1 的鳍
                             所在的原始全图 制到 EXPORT/Dphxxx/
  export_low_quality_images: 无鳍 (fin_count==0) 或没有任何 DolphinID 的原图
                             复制到 EXPORT/Quality below 60/
  export_best_images:        每只海豚 DL/DR 各选最佳鳍 (优先 BestInShot 标记中
                             clearness 最高的一张), 其所在原图复制到 EXPORT/
  export_social_structure:   按 SocialStructure 列 (NNxxx 组) 把原图复制到
                             EXPORT/NNxxx/
"""

import os
import shutil
import pandas as pd


class DatabaseExporter:
    """基于 FIN/IMAGE_METAINFO 元数据把原图导出到 root_dir/EXPORT/ 下。

    用法:
        ex = DatabaseExporter(root_dir, fin_meta, image_meta)
        ex.export()

    fin_meta / image_meta 可以是 CSV 文件路径 (str) 或 pandas DataFrame;
    """

    def __init__(self, root_dir, fin_meta, image_meta=None):
        self.root_dir = root_dir
        self.fin_df = self._load(fin_meta, "FIN_METAINFO")
        self.image_df = self._load(image_meta, "IMAGE_METAINFO") \
            if image_meta is not None else None

    @staticmethod
    def _load(src, name):
        """CSV 文件路径或 pandas DataFrame -> DataFrame。"""
        if isinstance(src, pd.DataFrame):
            return src.copy()
        if isinstance(src, str) and os.path.isfile(src):
            return pd.read_csv(src)
        raise ValueError("%s 必须是 CSV 文件路径或 pandas DataFrame: %r"
                         % (name, src))

    def _copy_orig(self, orig_name, folder):
        """把 root_dir 下的原图复制到 folder (按文件名落盘);
        源文件不存在返回 False, 否则返回 True。"""
        src = os.path.join(self.root_dir, str(orig_name))
        if not os.path.isfile(src):
            print("Invild file:", src)
            return False
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(src, os.path.join(folder, os.path.basename(str(orig_name))))
        return True

    def export_dolphin_images(self):
        """执行导出
        out_dir 默认为 root_dir/EXPORT; 已存在的目标文件直接覆盖。
        返回 {DolphinID: {"copied": n, "missing": [原图名...]}}。"""
        out_dir = os.path.join(self.root_dir, "EXPORT")
        df = self.fin_df
        # 忽略无 DolphinID 的行, 再按 clear==1 过滤
        sub = df.dropna(subset=["DolphinID"])
        sub = sub.query("clear==1")
        for dph, g in sub.groupby("DolphinID", sort=True):
            folder = os.path.join(out_dir, str(dph))
            for name in g["orig_img_name"]:
                self._copy_orig(name, folder)


    def export_low_quality_images(self):
        """
        Images("fin_count==0") in IMAGE_METAINFO
        & Images(without any DphID) in FIN_METAINFO
        will copy to "EXPORT/Quality below 60" folder
        """
        folder = os.path.join(self.root_dir, "EXPORT", "Quality below 60")
        low = set()
        img = self.image_df
        # 无鳍的原图 
        if img is not None and not img.empty:
            m = img["fin_count"] == 0
            low |= set(img.loc[m, "orig_img_name"].astype(str))
        else:
            print("Miss valid IAMGE_METAINFO.csv")
        fin = self.fin_df
        if not fin.empty and "orig_img_name" in fin.columns:
            # 有鳍但没有 DolphinID 的原图
            all_orig = set(fin["orig_img_name"].dropna())
            if "DolphinID" in fin.columns:
                with_dph = fin.dropna(subset=["DolphinID"])
                with_dph = set(with_dph["orig_img_name"].dropna())
            else:
                with_dph = set()
            low |= all_orig - with_dph
        for name in sorted(low):
            self._copy_orig(name, folder)

    def export_best_images(self):
        """
        export each dolphin's original image with best left and right fin image to EXPORT if it exists.
        For each aspect, select the one with highest clearness among fin image marked by BestInShot.
        """
        out_dir = os.path.join(self.root_dir, "EXPORT")
        df = self.fin_df
        sub = df.dropna(subset=["DolphinID", "Aspect"])
        for dph, g in sub.groupby("DolphinID", sort=True):
            for aspect in ("DL", "DR"):  # 每侧各选一张最佳鳍
                ga = g[g["Aspect"] == aspect]
                if ga.empty:
                    continue
                if "BestInShot" in ga.columns:  # 优先在 BestInShot 标记里选
                    m = ga["BestInShot"].astype(str).str.lower() \
                        .isin(("true", "1"))
                    if m.any():
                        ga = ga[m]
                if "clearness" in ga.columns and ga["clearness"].notna().any():
                    best = ga.loc[ga["clearness"].idxmax()]  # 最清晰的一张
                else:
                    best = ga.iloc[0]
                self._copy_orig(best["orig_img_name"], out_dir)

    def export_social_structure(self):
        """
        export images based on SocialStructure
        """
        out_dir = os.path.join(self.root_dir, "EXPORT")
        # SocialStructure 列按原图写回 (NNxxx 组名); 优先用 IMAGE_METAINFO
        # (一张原图一行), 缺失时退回 FIN_METAINFO
        df = self.fin_df
        if df is None or df.empty or "SocialStructure" not in df.columns :
            print("Miss valid SocialStructure meta info")
            return
        sub = df.dropna(subset=["SocialStructure"])
        for name, g in sub.groupby("SocialStructure", sort=True):
            folder = os.path.join(out_dir, str(name))
            for o in g["orig_img_name"]:
                self._copy_orig(o, folder)

    def export(self):
        self.export_dolphin_images()
        self.export_low_quality_images()
        self.export_best_images()
        self.export_social_structure()


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    meta_dir = os.path.join(root, "METAINFO")
    ex = DatabaseExporter(
        root,
        os.path.join(meta_dir, "FIN_METAINFO.csv"),
        os.path.join(meta_dir, "IMAGE_METAINFO.csv"))
    ex.export()
