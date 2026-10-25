#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doris_dataset — Doris 的数据层 (pandas / numpy)

从 root 目录的 METAINFO 子目录加载元数据 (PHOTO_METAINFO.csv / FIN_METAINFO.csv),
计算各类分组 (全图 / 连拍 / 部位 / 模糊 / 个体 / 社会结构 / FinID 总览 / 海豚配对),
并在拖拽改组时把改动写回 CSV (自动备份) 或 FIN_SIMILARITY.npy。

纯数据层, 不含任何 PyQt5 / GUI 依赖, 供 DorisGUI.py 使用。
"""

import os
import re
import shutil
from datetime import datetime

import pandas as pd
import numpy as np

ASPECT_LABELS = {"DL": "Left", "DR": "Right", "Others": "Misc", "ND":"Not Dolphin"}


class Dataset:
    """从 root 目录加载全部元数据。"""

    def __init__(self, root):
        self.root = root
        self.meta_dir = os.path.join(self.root, "METAINFO")
        self.photo_df = self._load_meta("PHOTO_METAINFO.csv")
        self.fin_df = self._load_meta("FIN_METAINFO.csv")
        if not self.fin_df.empty:
            self.fin_df["fullpath"] = self.root + "/" + self.fin_df["Path"]
            if "Clear" not in self.fin_df.columns:
                # True=Clear, False=Blur, 空=Mid(未判定)
                self.fin_df["Clear"] = pd.NA
        self._cache = {}

    def _invalidate_cache(self, keys=None):
        """清空只读视图缓存; keys=None 时全部清空。"""
        if keys is None:
            self._cache.clear()
        else:
            for k in keys:
                self._cache.pop(k, None)

    def _load_meta(self, fname):
        p = os.path.join(self.meta_dir, fname)
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()

    def orig_of_fin(self, fin_path):
        """鳍图路径 -> 原始全图路径 (找不到返回 None)。"""
        if self.fin_df.empty or "PhotoName" not in self.fin_df.columns:
            return None
        rows = self.fin_df[self.fin_df["fullpath"] == fin_path]
        if rows.empty:
            return None
        p = os.path.join(self.root, str(rows["PhotoName"].iloc[0]))
        return p if os.path.isfile(p) else None

    #  连拍分组
    def shots(self):
        if self.photo_df.empty or "PhotoName" not in self.photo_df.columns:
            return {}  # 无元数据时返回空组
        groups = {}
        for sid, g in self.photo_df.groupby("ShotID", sort=True):
            groups["shot_%s" % sid] = [os.path.join(self.root, n)
                                       for n in g["PhotoName"]]
        return groups

    def fin_boxes(self, dolphin_id=False):
        """全图绝对路径 -> [(x0, y0, x1, y1, cls, extra), ...] (stored 像素坐标)
        extra 为叠加文字 (FinID 'F3' 和/或 Clearness '0.87'), 都没有时为 None。
        dolphin_id=True 时追加 DolphinID ('Dph001', 仅 DL/DR 鳍有),
        且不再标注 p=/c= 和部位文字 (cls 加 '_' 前缀, 只保留框颜色)。"""
        key = ("fin_boxes", dolphin_id)
        if key in self._cache:
            return self._cache[key]
        boxes = {}
        if self.fin_df.empty or "PhotoName" not in self.fin_df.columns:
            self._cache[key] = boxes
            return boxes
        need = ["XMin", "YMin", "XMax", "YMax"]
        if not all(c in self.fin_df.columns for c in need):
            self._cache[key] = boxes
            return boxes
        has_fid = "FinID" in self.fin_df.columns
        has_cle = "Clearness" in self.fin_df.columns
        has_conf = "CropConf" in self.fin_df.columns
        has_dph = dolphin_id and "DolphinID" in self.fin_df.columns

        def extra(fid, cle, conf, dph):
            parts = []
            if fid is not None and not pd.isna(fid):
                parts.append("F%d" % int(float(fid)))
            if dph is not None and not pd.isna(dph):
                parts.append(str(dph))
            if has_dph:  # dolphin_id 模式只标 DolphinID, 不标 FinID
                return str(dph) if dph is not None and not pd.isna(dph) else None
            if conf is not None and not pd.isna(conf):
                parts.append("p=%.2f" % float(conf))
            if cle is not None and not pd.isna(cle):
                parts.append("c=%.2f" % float(cle))
            return " ".join(parts) or None

        for name, g in self.fin_df.groupby("PhotoName"):
            p = os.path.join(self.root, str(name))
            if not os.path.isfile(p):
                continue
            fids = g["FinID"] if has_fid else [None] * len(g)
            cles = g["Clearness"] if has_cle else [None] * len(g)
            confs = g["CropConf"] if has_conf else [None] * len(g)
            dphs = g["DolphinID"] if has_dph else [None] * len(g)
            boxes[p] = [
                # dolphin_id 模式: cls 加 '_' 前缀, 只取颜色不叠加部位文字
                (int(a), int(b), int(c), int(d),
                 "_" + str(cls) if has_dph else str(cls),
                 extra(fid, cle, conf, dph))
                for a, b, c, d, cls, fid, cle, conf, dph in zip(
                    g["XMin"], g["YMin"], g["XMax"], g["YMax"],
                    g["Aspect"], fids, cles, confs, dphs)]
        self._cache[key] = boxes
        return boxes

    def fin_shot_labels(self):
        """鳍图 -> 'shot_<id> clr=<Clearness> conf=<CropConf>' 叠加标签。
        同时按绝对路径和文件名两种键收录, 以覆盖 FIN/FinIDxxx 里的副本。"""
        key = "fin_shot_labels"
        if key in self._cache:
            return self._cache[key]
        labels = {}
        if self.fin_df.empty or "ShotID" not in self.fin_df.columns:
            self._cache[key] = labels
            return labels
        has_cle = "Clearness" in self.fin_df.columns
        has_conf = "CropConf" in self.fin_df.columns
        cles = self.fin_df["Clearness"] if has_cle else [float("nan")] * len(self.fin_df)
        confs = self.fin_df["CropConf"] if has_conf else [float("nan")] * len(self.fin_df)
        for p, sid, cle, conf in zip(self.fin_df["fullpath"],
                                     self.fin_df["ShotID"], cles, confs):
            if pd.isna(sid):
                continue
            try:  # ShotID 常被读成浮点, 显示为整数
                sid = int(float(sid))
            except (ValueError, TypeError):
                pass
            text = "shot_%s" % sid
            if not pd.isna(cle):
                text += " clr=%.2f" % float(cle)
            if not pd.isna(conf):
                text += " conf=%.2f" % float(conf)
            if isinstance(p, str):
                labels[p] = text
                labels[os.path.basename(p)] = text
        self._cache[key] = labels
        return labels

    def fin_best_marks(self):
        """BestInShot=True 的鳍图路径集合 (FIN_METAINFO.csv 的现成标记)。"""
        key = "fin_best_marks"
        if key in self._cache:
            return self._cache[key]
        if self.fin_df.empty or "BestInShot" not in self.fin_df.columns:
            self._cache[key] = set()
            return set()
        df = self.fin_df
        marks = df["BestInShot"].astype(str).str.lower().isin(("true", "1"))
        out = set(df.loc[marks, "fullpath"].dropna())
        self._cache[key] = out
        return out

    def mark_best_fins(self, paths):
        """把给定鳍图标记为 BestInFin=True (同一 FinID 的其它鳍自动置 False),
        写回 FIN_METAINFO.csv (自动备份); 有改动返回 True。"""
        df = self.fin_df
        mask = df["fullpath"].isin(paths) if "fullpath" in df.columns \
            else pd.Series(False, index=df.index)
        if not mask.any() and "Path" in df.columns:
            names = {os.path.basename(p) for p in paths}
            mask = df["Path"].map(lambda p: os.path.basename(str(p)) in names)
        if not mask.any():
            return False
        if "BestInFin" not in df.columns:
            df["BestInFin"] = False
        # 每个 FinID 只保留一个最佳: 先清掉被标记鳍同 FinID 其它鳍的标记
        fids = set(df.loc[mask, "FinID"].dropna())
        same = df["FinID"].isin(fids)
        df.loc[same, "BestInFin"] = False
        df.loc[mask, "BestInFin"] = True
        self._save_csv(df, "FIN_METAINFO.csv")
        return True

    def fin_best_in_fin_marks(self):
        """BestInFin=True 的鳍图路径集合 (每个 FinID 的最佳鳍)。"""
        key = "fin_best_in_fin_marks"
        if key in self._cache:
            return self._cache[key]
        if self.fin_df.empty or "BestInFin" not in self.fin_df.columns:
            self._cache[key] = set()
            return set()
        df = self.fin_df
        marks = df["BestInFin"].astype(str).str.lower().isin(("true", "1"))
        out = set(df.loc[marks, "fullpath"].dropna())
        self._cache[key] = out
        return out

    def fin_clear_marks(self):
        """Clear==1 (Clear) 的鳍图路径集合; 供 Review 页 "Clear" 筛选复选框。"""
        key = "fin_clear_marks"
        if key in self._cache:
            return self._cache[key]
        if self.fin_df.empty or "Clear" not in self.fin_df.columns:
            self._cache[key] = set()
            return set()
        df = self.fin_df
        marks = df["Clear"].astype(str).str.lower().isin(("true", "1"))
        out = set(df.loc[marks, "fullpath"].dropna())
        self._cache[key] = out
        return out

    #  鳍侧分组
    def aspects(self):
        groups = {label: [] for label in ASPECT_LABELS.values()}  # 空组也列出, 作为拖放目标
        if self.fin_df.empty or "fullpath" not in self.fin_df.columns:
            return groups
        df = self.fin_df.dropna(subset=["fullpath"])
        for cls, g in df.groupby("Aspect"):
            label = ASPECT_LABELS.get(cls, str(cls))
            groups.setdefault(label, []).extend(g["fullpath"])
        return groups

    # 模糊分组: 依据 fin_df["clear"] 列 (True=Clear, False=Blurd)
    def blur(self):
        groups = {"Blur": [], "Clear": []}
        if self.fin_df.empty or "fullpath" not in self.fin_df.columns:
            return groups
        for val, g in self.fin_df.groupby("Clear", dropna=False):
            key = "Clear" if val else "Blur"
            groups[key].extend(g["fullpath"])
        return groups

    #  FinID 分组
    def fin_ids(self):
        groups = {}
        if self.fin_df.empty or "FinID" not in self.fin_df.columns:
            return groups
        for fid, g in self.fin_df.dropna(subset=["FinID"]).groupby("FinID", sort=True):
            groups["FinID%s" % fid] = list(g["fullpath"])
        return groups

    def rename_fin_ids(self):
        """把所有 FinID 重新编号为 1..N (N=非空 FinID 数量), 保持原排序顺序;
        写回 FIN_METAINFO.csv 并返回 old->new 映射。"""
        df = self.fin_df
        if df.empty or "FinID" not in df.columns:
            return {}
        old_ids = [v for v in df["FinID"].dropna().unique() if v != 0]
        if not old_ids:
            return {}
        # 统一按数值排序, 混合字符串/浮点时尽量按数值
        def num_key(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float("inf")
        old_ids_sorted = sorted(old_ids, key=num_key)
        mapping = {old: (i + 1) for i, old in enumerate(old_ids_sorted)}
        df["FinID"] = df["FinID"].map(
            lambda x: mapping.get(x, x)).astype(df["FinID"].dtype)
        self._save_csv(df, "FIN_METAINFO.csv")
        return mapping

    def rename_dolphin_ids(self):
        """把所有 DolphinID 重新编号为 Dph001..DphN (N=当前 DolphinID 数量),
        保持原排序顺序; 写回 FIN_METAINFO.csv 与 DOLPHINS.csv 并返回映射。"""
        df = self.fin_df
        if df.empty or "DolphinID" not in df.columns:
            return {}
        old_names = [str(v) for v in df["DolphinID"].dropna().unique()]
        if not old_names:
            return {}

        def num_key(n):
            m = re.match(r"Dph(\d+)$", n)
            return int(m.group(1)) if m else float("inf")

        old_names_sorted = sorted(old_names, key=num_key)
        mapping = {old: "Dph%03d" % (i + 1)
                   for i, old in enumerate(old_names_sorted)}
        df["DolphinID"] = df["DolphinID"].map(
            lambda x: mapping.get(str(x), x))
        self._save_csv(df, "FIN_METAINFO.csv")
        # 同步 DOLPHINS.csv 中的空海豚记录
        p = os.path.join(self.meta_dir, "DOLPHINS.csv")
        if os.path.isfile(p):
            try:
                extras = pd.read_csv(p)
                if "DolphinID" in extras.columns:
                    extras["DolphinID"] = extras["DolphinID"].astype(str).map(
                        lambda x: mapping.get(x, x))
                    extras.to_csv(p, index=False)
            except Exception:
                pass
        return mapping

    # 社会结构: 整组连拍 (按 ShotID) 内出现多于一个 DolphinID 的 shot, 若多个
    #    shot 出现同一组 DolphinID (集合完全相同) 则并入同一 NN 组;
    #    组名 NN001...; 组内只收同一张原图含多个 DolphinID 鳍的图;
    #    组名写回 PHOTO_METAINFO.csv & FIN_METAINFO.csv 的 SocialStructure 列
    #    (须在 ds.dolphins() 分配 DolphinID 列之后调用)
    def social(self):
        key = "social"
        if key in self._cache:
            return self._cache[key]
        df = self.fin_df
        if df.empty or "DolphinID" not in df.columns \
                or "PhotoName" not in df.columns \
                or "ShotID" not in df.columns:
            self._save_social({})  # 无海豚配对时清空已保存的 SocialStructure 列
            self._cache[key] = {}
            return {}
        sub = df.dropna(subset=["DolphinID", "PhotoName", "ShotID"])

        def shot_num(sid):
            try:
                return float(sid)
            except (TypeError, ValueError):
                return float("inf")

        # ShotID -> 整组连拍内出现的 DolphinID 集合 (只留多海豚的连拍)
        members = {}  # ShotID -> set(DolphinID)
        for sid, g in sub.groupby("ShotID"):
            ids = set(g["DolphinID"].astype(str))
            if len(ids) >= 2:
                members[sid] = ids
        if not members:
            self._save_social({})
            self._cache[key] = {}
            return {}
        # 同一 DolphinID 集合的多个连拍并入同一 NN 组
        by_ids = {}  # frozenset(DolphinID) -> [ShotID]
        for sid, ids in members.items():
            by_ids.setdefault(frozenset(ids), []).append(sid)
        # 组排序: DolphinID 数降序, 同数按最小 ShotID 升序
        def group_key(item):
            ids, sids = item
            return (-len(ids), min(shot_num(s) for s in sids))

        groups = {}
        self.social_tooltips = {}
        img_group = {}  # PhotoName -> NN 组名
        formed = []  # (DolphinID 集合, shot 列表, 组内原图)
        for ids, sids in sorted(by_ids.items(), key=group_key):
            # 组内仍只收同一张原图里含多个海豚 (>=2 个 DolphinID) 鳍的图
            multi = []
            for sid in sorted(sids, key=shot_num):
                g = sub[sub["ShotID"] == sid]
                per_img = g.groupby("PhotoName")["DolphinID"].nunique()
                multi.extend(per_img[per_img >= 2].index)
            if multi:
                formed.append((ids, sids, multi))
        for n, (ids, sids, multi) in enumerate(formed, 1):
            name = "NN%03d" % n
            sid_disp = []
            for s in sorted(sids, key=shot_num):  # ShotID 常读成浮点, 显示整数
                try:
                    sid_disp.append(str(int(float(s))))
                except (ValueError, TypeError):
                    sid_disp.append(str(s))
            self.social_tooltips[name] = "shots %s 海豚: %s" % (
                ",".join(sid_disp), ",".join(sorted(ids)))
            img_group.update({o: name for o in multi})
            groups[name] = [os.path.join(self.root, o) for o in multi
                            if os.path.isfile(os.path.join(self.root, o))]
        self._save_social(img_group)
        self._cache[key] = groups
        return groups

    def _save_social(self, img_group):
        """把 NN 组名写回 PHOTO_METAINFO.csv 和 FIN_METAINFO.csv 的
        SocialStructure 列 (鳍图按所属原图取组名; 不在任何组的为空);
        内容不变时不写盘。"""
        if not self.photo_df.empty and "PhotoName" in self.photo_df.columns:
            col = self.photo_df["PhotoName"].map(img_group)
            old = self.photo_df["SocialStructure"] \
                if "SocialStructure" in self.photo_df.columns else None
            if old is None or not old.fillna("").astype(str) \
                    .equals(col.fillna("").astype(str)):
                self.photo_df["SocialStructure"] = col
                self._save_csv(self.photo_df, "PHOTO_METAINFO.csv")
        if not self.fin_df.empty and "PhotoName" in self.fin_df.columns:
            col = self.fin_df["PhotoName"].map(img_group)
            old = self.fin_df["SocialStructure"] \
                if "SocialStructure" in self.fin_df.columns else None
            if old is None or not old.fillna("").astype(str) \
                    .equals(col.fillna("").astype(str)):
                self.fin_df["SocialStructure"] = col
                self._save_csv(self.fin_df, "FIN_METAINFO.csv")

    def _similarity_matrix(self):
        """懒加载并缓存 FIN_SIMILARITY.npy (与 fin_df 行对齐)。"""
        if getattr(self, "_sim_loaded", False):
            return self._sim
        self._sim_loaded = True
        self._sim = None
        sim_file = os.path.join(self.meta_dir, "FIN_SIMILARITY.npy")
        if os.path.isfile(sim_file):
            try:
                sim = np.load(sim_file)
                if sim.shape[0] == len(self.fin_df):
                    self._sim = sim
            except Exception:
                pass
        return self._sim

    def similar_fins(self, group, threshold=0.4):
        """与当前 FinID 两两相似度 > threshold 的其它个体鳍图 (同 aspect)。
        只比较、只返回 BestInShot 标记的鳍图。
        返回 (paths, labels): 按相似度降序, labels = 'FinIDx sim=0.83'。"""
        key = ("similar_fins", group, threshold)
        if key in self._cache:
            return self._cache[key]
        df = self.fin_df
        m = re.match(r"FinID(.+)$", group or "")
        sim = self._similarity_matrix()
        if not m or df.empty or "FinID" not in df.columns or sim is None:
            self._cache[key] = ([], {})
            return [], {}
        try:
            fid = int(float(m.group(1)))
        except ValueError:
            fid = m.group(1)
        if "BestInShot" in df.columns:
            marked = df["BestInShot"].astype(str).str.lower() \
                .isin(("true", "1")).to_numpy()
        else:
            marked = np.ones(len(df), dtype=bool)  # 无标记列时不过滤
        fids = df["FinID"].to_numpy()
        cur = np.where((fids == fid) & marked)[0]
        if not len(cur):
            self._cache[key] = ([], {})
            return [], {}
        # 每个鳍与当前组最佳鳍的最大相似度 (只比较同 aspect 的行)
        best = np.full(len(df), -1.0)
        cls = df["Aspect"].to_numpy() if "Aspect" in df.columns else None
        for i in cur:
            row = sim[i]
            if cls is not None:
                row = np.where(cls == cls[i], row, -1.0)
            best = np.maximum(best, row)
        paths, labels = [], {}
        for j in np.argsort(-best):
            if best[j] <= threshold:
                break
            if not marked[j]:  # 只显示 BestInShot 标记的鳍图
                continue
            ofid = fids[j]
            if ofid == fid or pd.isna(ofid) or ofid == 0:
                continue
            p = df["fullpath"].iloc[j]
            if not isinstance(p, str):
                continue
            paths.append(p)
            labels[p] = "FinID%s sim=%.2f" % (
                int(ofid) if float(ofid).is_integer() else ofid, best[j])
        self._cache[key] = (paths, labels)
        return paths, labels

    #  FinID 总览: 每个个体一张代表鳍图 (最清晰的)
    def fin_id_previews(self):
        """返回 (paths, labels): 每个 FinID 一张代表图 + 'FinIDx (n)' 角标。
        代表图优先取 BestInShot 标记中 Clearness 最高的一张 (与 Dolphin 页同规则)。"""
        df = self.fin_df
        if df.empty or "FinID" not in df.columns \
                or "fullpath" not in df.columns:
            return [], {}
        paths, labels = [], {}
        for fid, g in df.dropna(subset=["FinID"]).groupby("FinID", sort=True):
            if fid == 0:
                continue  # 0 = 未分类
            n = len(g)  # 角标数量按全部鳍图计, 不受 BestInShot 筛选影响
            if "BestInShot" in g.columns:  # 优先在 BestInShot 标记里选
                m = g["BestInShot"].astype(str).str.lower().isin(("true", "1"))
                if m.any():
                    g = g[m]
            if "Clearness" in g.columns and g["Clearness"].notna().any():
                rep = g.loc[g["Clearness"].idxmax()]  # 最清晰的一张
            else:
                rep = g.iloc[0]
            p = rep["fullpath"]
            paths.append(p)
            labels[p] = "FinID%s (%d)" % (fid, n)
        return paths, labels

    #  统计: FinID 鳍图数分布 / 部位占比
    def fin_id_counts(self):
        """[(label, count)]: 每个 FinID 的鳍图数, 按数量降序 (同数按编号升序);
        不含 FinID0 (未分类, 非真实个体)。"""
        if self.fin_df.empty or "FinID" not in self.fin_df.columns:
            return []
        counts = self.fin_df.dropna(subset=["FinID"]).groupby("FinID").size()
        counts = counts[[fid for fid in counts.index if fid != 0]]
        def num(fid):
            try:
                return float(fid)
            except (TypeError, ValueError):
                return float("inf")
        return [("FinID%s" % fid, int(counts[fid]))
                for fid in sorted(counts.index,
                                  key=lambda f: (-counts[f], num(f)))]

    def fin_id_shot_counts(self):
        """[(label, count)]: 每个 FinID 出现的不同 shot 数 (按 ShotID 去重),
        排序与 fin_id_counts 一致; 不含 FinID0。"""
        if self.fin_df.empty or "FinID" not in self.fin_df.columns \
                or "ShotID" not in self.fin_df.columns:
            return []
        df = self.fin_df.dropna(subset=["FinID", "ShotID"])
        counts = df.groupby("FinID")["ShotID"].nunique()
        counts = counts[[fid for fid in counts.index if fid != 0]]
        def num(fid):
            try:
                return float(fid)
            except (TypeError, ValueError):
                return float("inf")
        return [("FinID%s" % fid, int(counts[fid]))
                for fid in sorted(counts.index,
                                  key=lambda f: (-counts[f], num(f)))]

    def aspect_counts(self):
        """[(label, count)]: 各部位类别的鳍图数。"""
        if self.fin_df.empty or "Aspect" not in self.fin_df.columns:
            return []
        return [(ASPECT_LABELS.get(c, str(c)), int(n))
                for c, n in self.fin_df.groupby("Aspect").size().items()]

    def clearness_values(self):
        """返回所有非空 Clearness 数值列表 (供统计页绘制直方图)。"""
        if self.fin_df.empty or "Clearness" not in self.fin_df.columns:
            return []
        return self.fin_df["Clearness"].dropna().astype(float).tolist()

    def dolphin_shot_counts(self):
        """[(DolphinID, dl_shots, dr_shots)]: 每只海豚在 DL/DR 上出现的
        不同 shot 数, 按 DL+DR 总数降序; 不含空海豚。"""
        if self.fin_df.empty or "DolphinID" not in self.fin_df.columns \
                or "ShotID" not in self.fin_df.columns \
                or "Aspect" not in self.fin_df.columns:
            return []
        df = self.fin_df.dropna(subset=["DolphinID", "ShotID", "Aspect"])
        df = df[df["Aspect"].isin(["DL", "DR"])]
        counts = df.groupby(["DolphinID", "Aspect"])["ShotID"].nunique().unstack(
            fill_value=0)
        counts = counts.reindex(columns=["DL", "DR"], fill_value=0)
        counts["total"] = counts["DL"] + counts["DR"]
        counts = counts.sort_values("total", ascending=False).drop(
            columns="total")
        return [(str(name), int(row["DL"]), int(row["DR"]))
                for name, row in counts.iterrows()]

    def social_network_data(self):
        """返回社会结构网络图数据。

        Returns
        -------
        (nodes, edges, node_counts) :
            nodes   : list[str] — DolphinID 列表（按出现 shot 数降序）
            edges   : list[(u, v, weight)] — 两海豚共同出现的不同 ShotID 数量
            node_counts : dict[DolphinID, int] — 每只海豚出现的不同 ShotID 数
        """
        if self.fin_df.empty or "DolphinID" not in self.fin_df.columns \
                or "ShotID" not in self.fin_df.columns:
            return [], [], {}
        df = self.fin_df.dropna(subset=["DolphinID", "ShotID"])
        # 每 ShotID 出现的 DolphinID 集合
        shot_dphs = df.groupby("ShotID")["DolphinID"].apply(
            lambda s: set(s.astype(str)))
        # 节点出现次数
        node_counts = df.groupby("DolphinID")["ShotID"].nunique().to_dict()
        node_counts = {str(k): int(v) for k, v in node_counts.items()}
        # 共同出现计数
        pair_counts = {}
        for dphs in shot_dphs:
            dphs = sorted(dphs)
            if len(dphs) < 2:
                continue
            for i in range(len(dphs)):
                for j in range(i + 1, len(dphs)):
                    pair_counts[(dphs[i], dphs[j])] = \
                        pair_counts.get((dphs[i], dphs[j]), 0) + 1
        edges = [(u, v, w) for (u, v), w in pair_counts.items()]
        nodes = sorted(node_counts.keys(),
                       key=lambda d: (-node_counts[d], d))
        return nodes, edges, node_counts

    # 9. 海豚配对: DL/DR 鳍配对
    def _extra_dolphins(self):
        """DOLPHINS.csv 中用户手工创建的空海豚名列表。"""
        p = os.path.join(self.meta_dir, "DOLPHINS.csv")
        if os.path.isfile(p):
            try:
                return list(pd.read_csv(p)["DolphinID"].dropna().astype(str))
            except Exception:
                pass
        return []

    def new_dolphin(self):
        """创建空海豚 (名字顺延, 记入 DOLPHINS.csv 持久化), 返回名字。"""
        names = {d[0] for d in self.dolphins()} | set(self._extra_dolphins())
        nums = [int(m.group(1)) for n in names
                for m in [re.match(r"Dph(\d+)$", n)] if m]
        name = "Dph%03d" % (max(nums) + 1 if nums else 1)
        extras = self._extra_dolphins()
        extras.append(name)
        pd.DataFrame({"DolphinID": extras}).to_csv(
            os.path.join(self.meta_dir, "DOLPHINS.csv"), index=False)
        self._invalidate_cache(["dolphins", "dolphin_groups"])
        return name

    def dolphins(self, match_threshold=0.5):
        """[(name, dl_fid, dr_fid, dl_path, dr_path)]。
        配对关系存于 fin_df 的 DolphinID 列并写回 CSV:
        每个 DL FinID 一只海豚 (Dph001...); DR FinID 按与 DL 的最高
        相似度贪心配对 (FIN_SIMILARITY.npy, 每只 DR 最多配一只海豚),
        低于阈值的 DR FinID 也单独成海豚 (上格 DL 留空), 不丢任何 DR;
        另有 DOLPHINS.csv 里用户手工创建的空海豚。
        鳍图取该 FinID 该侧 BestInShot 标记中 Clearness 最高的一张
        (无标记时回退到该侧全部鳍图)。"""
        key = ("dolphins", match_threshold)
        if key in self._cache:
            return self._cache[key]
        df = self.fin_df
        if df.empty or "FinID" not in df.columns \
                or "Aspect" not in df.columns:
            self._cache[key] = []
            return []
        changed = False
        if "DolphinID" not in df.columns:
            df["DolphinID"] = pd.NA
            changed = True
        sub = df.dropna(subset=["FinID"])
        sub = sub[sub["FinID"] != 0]  # 0 = 未分类

        def dph_of(fid, cls):
            v = sub[(sub["FinID"] == fid) & (sub["Aspect"] == cls)][
                "DolphinID"].dropna().unique()
            return str(v[0]) if len(v) else None

        pairs = {}  # name -> {"dl": fid|None, "dr": fid|None}
        for fid in sub[sub["Aspect"] == "DL"]["FinID"].unique():
            n = dph_of(fid, "DL")
            if n:
                pairs.setdefault(n, {})["dl"] = fid
        for fid in sub[sub["Aspect"] == "DR"]["FinID"].unique():
            n = dph_of(fid, "DR")
            if n:
                pairs.setdefault(n, {})["dr"] = fid
        for n in self._extra_dolphins():  # 用户创建的空海豚
            pairs.setdefault(n, {})
        # 新名字顺延编号 (含已有和用户创建的名字)
        nums = [int(m.group(1)) for n in pairs
                for m in [re.match(r"Dph(\d+)$", n)] if m]
        nxt = max(nums) + 1 if nums else 1
        for fid in sorted(sub[sub["Aspect"] == "DL"]["FinID"].unique()):
            if dph_of(fid, "DL") is None:
                name = "Dph%03d" % nxt
                nxt += 1
                df.loc[(df["FinID"] == fid) & (df["Aspect"] == "DL"),
                       "DolphinID"] = name
                pairs[name] = {"dl": fid, "dr": None}
                changed = True
        # 未配对的 DR FinID: 先按与 DL 的最高相似度贪心配对
        unmatched_dr = [f for f in sub[sub["Aspect"] == "DR"]["FinID"].unique()
                        if dph_of(f, "DR") is None]
        mismatch = self._mismatch_pairs()  # 用户确认过的不匹配对, 不再自动重配
        sim = self._similarity_matrix()
        used_dr = set()
        if sim is not None and unmatched_dr:
            fids = df["FinID"].to_numpy()
            clss = df["Aspect"].to_numpy()

            def rows_of(fid, cls):
                return np.where((fids == fid) & (clss == cls))[0]

            cands = []
            for name, d in pairs.items():
                if d.get("dl") is None or d.get("dr") is not None:
                    continue
                ia = rows_of(d["dl"], "DL")
                if not len(ia):
                    continue
                dl_key = self._fid_key(d["dl"])
                for dr in unmatched_dr:
                    if (dl_key, self._fid_key(dr)) in mismatch:
                        continue
                    ib = rows_of(dr, "DR")
                    if len(ib):
                        cands.append((float(sim[np.ix_(ia, ib)].max()),
                                      name, dr))
            cands.sort(key=lambda c: -c[0])
            used_dl = set()
            for s, name, dr in cands:
                if s <= match_threshold or dr in used_dr or name in used_dl:
                    continue
                df.loc[(df["FinID"] == dr) & (df["Aspect"] == "DR"),
                       "DolphinID"] = name
                pairs[name]["dr"] = dr
                used_dr.add(dr)
                used_dl.add(name)
                changed = True
        # 剩下的 DR FinID (低于阈值/无 DL 可配): 各自单独成海豚, DL 留空
        for dr in unmatched_dr:
            if dr in used_dr:
                continue
            name = "Dph%03d" % nxt
            nxt += 1
            df.loc[(df["FinID"] == dr) & (df["Aspect"] == "DR"),
                   "DolphinID"] = name
            pairs[name] = {"dl": None, "dr": dr}
            changed = True
        if changed:
            self._save_csv(df, "FIN_METAINFO.csv")

        def best(fid, cls):
            g = df[(df["FinID"] == fid) & (df["Aspect"] == cls)] \
                .dropna(subset=["fullpath"])
            if g.empty:
                return None
            if "BestInFin" in g.columns:  # 优先在 BestInFin 标记里选
                m = g["BestInFin"].astype(str).str.lower().isin(("true", "1"))
                if m.any():
                    g = g[m]
                else:
                    if "BestInShot" in g.columns:  # 其次在 BestInShot 标记里选
                        m = g["BestInShot"].astype(str).str.lower().isin(("true", "1"))
                        if m.any():
                            g = g[m]
            elif( "BestInShot" in g.columns):  # 其次在 BestInShot 标记里选
                m = g["BestInShot"].astype(str).str.lower().isin(("true", "1"))
                if m.any():
                    g = g[m]
            if "Clearness" in g.columns and g["Clearness"].notna().any():
                return g.loc[g["Clearness"].idxmax(), "fullpath"]
            return g["fullpath"].iloc[0]

        out = [(name, d.get("dl"), d.get("dr"),
                best(d["dl"], "DL") if d.get("dl") is not None else None,
                best(d["dr"], "DR") if d.get("dr") is not None else None)
               for name, d in sorted(pairs.items())]
        self._cache[key] = out
        return out

    def dolphin_groups(self):
        """DolphinID -> 该海豚全部鳍图路径 (按 DolphinID 排序)。

        包含 FIN_METAINFO.csv 中有鳍图的海豚, 也包含 DOLPHINS.csv 中
        用户手工创建的空海豚 (方便 Review 模式列出全部可编辑对象)。
        """
        key = "dolphin_groups"
        if key in self._cache:
            return self._cache[key]
        groups = {}
        df = self.fin_df
        if not df.empty and "DolphinID" in df.columns \
                and "fullpath" in df.columns:
            sub = df.dropna(subset=["DolphinID", "fullpath"])
            for name, g in sub.groupby("DolphinID", sort=True):
                groups[str(name)] = [p for p in g["fullpath"]
                                     if isinstance(p, str)]
        # 补上空海豚 (无鳍图)
        for name in self._extra_dolphins():
            groups.setdefault(name, [])
        self._cache[key] = groups
        return groups

    def dolphin_fin_labels(self):
        """鳍图路径 -> 'DL F1' / 'DR F5' 标注 (Dolphin Review Mode 用)。"""
        key = "dolphin_fin_labels"
        if key in self._cache:
            return self._cache[key]
        labels = {}
        df = self.fin_df
        if df.empty or "fullpath" not in df.columns:
            self._cache[key] = labels
            return labels
        for _, r in df.iterrows():
            p = r["fullpath"]
            if not isinstance(p, str):
                continue
            parts = []
            if isinstance(r.get("Aspect"), str):
                parts.append(r["Aspect"])
            fid = r.get("FinID")
            if fid is not None and not pd.isna(fid):
                try:
                    parts.append("F%d" % int(float(fid)))
                except (TypeError, ValueError):
                    parts.append(str(fid))
            if parts:
                labels[p] = " ".join(parts)
        self._cache[key] = labels
        return labels

    def swap_dolphin_dr(self, src_name, dst_name):
        """交换两只海豚的 DR FinID (目标无 DR 时为移动), 写回 CSV。"""
        df = self.fin_df
        if df.empty or "DolphinID" not in df.columns \
                or src_name == dst_name:
            return

        def dr_of(name):
            v = df[(df["DolphinID"] == name) & (df["Aspect"] == "DR")][
                "FinID"].dropna().unique()
            return v[0] if len(v) else None

        s, d = dr_of(src_name), dr_of(dst_name)
        if s is None:
            return  # 源海豚没有 DR 可交换
        df.loc[(df["FinID"] == s) & (df["Aspect"] == "DR"),
               "DolphinID"] = dst_name
        if d is not None:
            df.loc[(df["FinID"] == d) & (df["Aspect"] == "DR"),
                   "DolphinID"] = src_name
        self._save_csv(df, "FIN_METAINFO.csv")

    # ------------------------------------------------------------------
    # Match Mode: 手工确认配对/不匹配
    # ------------------------------------------------------------------
    @staticmethod
    def _fid_key(fid):
        """FinID 归一化键: 浮点读法 (12.0) 与整数/字符串读法视为同一个。"""
        try:
            return str(int(float(fid)))
        except (TypeError, ValueError):
            return str(fid)

    def _mismatch_path(self):
        return os.path.join(self.meta_dir, "DOLPHIN_MISMATCH.csv")

    def _mismatch_pairs(self):
        """用户在 Match Mode 确认不匹配的 (dl_fid, dr_fid) 键集合。"""
        key = "_mismatch_pairs"
        if key in self._cache:
            return self._cache[key]
        pairs = set()
        p = self._mismatch_path()
        if os.path.isfile(p):
            try:
                for _, r in pd.read_csv(p).iterrows():
                    pairs.add((self._fid_key(r["dl_fid"]),
                               self._fid_key(r["dr_fid"])))
            except Exception:
                pass
        self._cache[key] = pairs
        return pairs

    def _save_mismatch(self, pairs):
        pd.DataFrame(sorted(pairs), columns=["dl_fid", "dr_fid"]).to_csv(
            self._mismatch_path(), index=False)
        self._invalidate_cache(["_mismatch_pairs"])

    def assign_dolphin_fin(self, name, fid, cls):
        """确认匹配: 把 FinID 划入 DphID, 写回 CSV;
        同时清除涉及该鳍的不匹配记录。"""
        df = self.fin_df
        if df.empty or "DolphinID" not in df.columns:
            return
        fidk = self._fid_key(fid)
        mask = (df["FinID"].map(self._fid_key) == fidk) \
            & (df["Aspect"] == cls)
        if not mask.any():
            return
        df.loc[mask, "DolphinID"] = name
        pairs = self._mismatch_pairs()
        side = 1 if cls == "DR" else 0
        kept = {p for p in pairs if p[side] != fidk}
        if kept != pairs:
            self._save_mismatch(kept)
        self._save_csv(df, "FIN_METAINFO.csv")

    def unassign_dolphin_fin(self, fid, cls):
        """确认不匹配: 把某 FinID 的某侧鳍移出当前海豚, 并记录 (dl, dr)
        不匹配对 (DOLPHIN_MISMATCH.csv), 防止 dolphins() 的贪心配对再把
        它们配回去。被移出的鳍下次 dolphins() 单独成豚。"""
        df = self.fin_df
        if df.empty or "DolphinID" not in df.columns:
            return
        fidk = self._fid_key(fid)
        mask = (df["FinID"].map(self._fid_key) == fidk) \
            & (df["Aspect"] == cls)
        if not mask.any():
            return
        names = df.loc[mask, "DolphinID"].dropna().unique()
        if not len(names):
            return
        name = str(names[0])
        other = "DL" if cls == "DR" else "DR"
        ofids = df[(df["DolphinID"] == name) & (df["Aspect"] == other)][
            "FinID"].dropna().unique()
        if len(ofids):  # 有对侧鳍才构成一对不匹配
            ofidk = self._fid_key(ofids[0])
            pairs = self._mismatch_pairs()
            pairs.add((ofidk, fidk) if cls == "DR" else (fidk, ofidk))
            self._save_mismatch(pairs)
        df.loc[mask, "DolphinID"] = pd.NA
        self._save_csv(df, "FIN_METAINFO.csv")

    def dolphin_candidates(self, fid, cls, threshold=0.3, limit=30):
        """与 (fid, cls) 鳍最相似的异侧候选鳍 (Match Mode 候选栏)。
        返回 (fids, paths, labels): 每个候选 FinID 取 BestInShot 中最清晰
        一张代表图, labels = 'FinIDx sim=0.83', 按相似度降序。"""
        df = self.fin_df
        sim = self._similarity_matrix()
        if df.empty or sim is None or "FinID" not in df.columns \
                or "Aspect" not in df.columns:
            return [], [], {}
        other = "DR" if cls == "DL" else "DL"
        fids = df["FinID"].to_numpy()
        clss = df["Aspect"].to_numpy()
        keys = pd.Series(fids).map(self._fid_key).to_numpy()
        cur = np.where((keys == self._fid_key(fid)) & (clss == cls))[0]
        if not len(cur):
            return [], [], {}
        # 每个候选鳍与当前 FinID 各鳍的最大相似度 (只看异侧的行)
        best = np.full(len(df), -1.0)
        for i in cur:
            best = np.maximum(best, np.where(clss == other, sim[i], -1.0))
        scores = {}  # 候选 FinID 键 -> 最高相似度 (每 FinID 只取一次)
        for j in np.argsort(-best):
            if best[j] <= threshold:
                break
            if clss[j] != other or pd.isna(fids[j]) or fids[j] == 0:
                continue
            k = keys[j]
            if k not in scores:
                scores[k] = float(best[j])
            if len(scores) >= limit:
                break
        out_fids, paths, labels = [], [], {}
        for k, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            g = df[(keys == k) & (clss == other)] \
                .dropna(subset=["fullpath"])
            if g.empty:
                continue
            if "BestInShot" in g.columns:  # 优先在 BestInShot 标记里选
                m = g["BestInShot"].astype(str).str.lower() \
                    .isin(("true", "1"))
                if m.any():
                    g = g[m]
            if "Clearness" in g.columns and g["Clearness"].notna().any():
                p = g.loc[g["Clearness"].idxmax(), "fullpath"]
            else:
                p = g["fullpath"].iloc[0]
            out_fids.append(k)
            paths.append(p)
            labels[p] = "FinID%s sim=%.2f" % (k, s)
        return out_fids, paths, labels

    # ------------------------------------------------------------------
    # 拖拽改组: 只更新 dataframe 并写回 CSV, 不移动/重命名任何文件
    # ------------------------------------------------------------------
    def _save_csv(self, df, fname):
        path = os.path.join(self.meta_dir, fname)
        if os.path.isfile(path):
            bdir = os.path.join(self.meta_dir, "backup")
            os.makedirs(bdir, exist_ok=True)
            shutil.copy2(path, os.path.join(
                bdir, fname + "." +
                datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"))
        df.drop(columns=["fullpath"], errors="ignore") \
            .to_csv(path, index=False)
        self._invalidate_cache()

    def move_shot(self, paths, group):
        """改连拍分组: 更新 photo_df 和 fin_df 的 ShotID。"""
        m = re.match(r"shot_(\d+)$", group)
        if not m:
            return {}
        sid = int(m.group(1))
        names = {os.path.basename(p) for p in paths}
        moved = {p: p for p in paths}
        if not self.photo_df.empty:
            mask = self.photo_df["PhotoName"].isin(names)
            if mask.any():
                self.photo_df.loc[mask, "ShotID"] = sid
                self._save_csv(self.photo_df, "PHOTO_METAINFO.csv")
        if not self.fin_df.empty and "PhotoName" in self.fin_df.columns:
            mask = self.fin_df["PhotoName"].isin(names)
            if mask.any():
                self.fin_df.loc[mask, "ShotID"] = sid
                self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return moved

    def move_aspect(self, paths, group):
        """改部位类别: 只更新 fin_df 的 aspect 列。"""
        cls = {v: k for k, v in ASPECT_LABELS.items()}.get(group, group)
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        self.fin_df.loc[mask, "Aspect"] = cls
        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return {p: p for p in paths}

    def move_blur(self, paths, group):
        """改质量分组: 更新 Clear / clear 列 (True=Clear, False=Blur, 空=Mid)。"""
        val = {"Clear": True, "Blur": False}.get(group, pd.NA)
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        self.fin_df.loc[mask, "Clear"] = val
        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        return {p: p for p in paths}

    def move_to_fin_id(self, paths, group):
        """改个体分组: 更新 fin_df 的 FinID 列, 同时统一该 FinID 的 DolphinID。
        被拖鳍所属 FinID 的全部鳍一起移动 (即整个个体并入目标组)。"""
        m = re.match(r"FinID(.+)$", group)
        if not m:
            return {}
        fid = m.group(1)
        try:  # FinID 为数值列时保持数值类型
            fid = int(float(fid))
        except ValueError:
            pass
        if "FinID" not in self.fin_df.columns:
            self.fin_df["FinID"] = pd.NA
        if "DolphinID" not in self.fin_df.columns:
            self.fin_df["DolphinID"] = pd.NA
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        # 扩展到与被拖鳍同 FinID 的所有鳍 (忽略未分类的 0/空)
        src_ids = [v for v in self.fin_df.loc[mask, "FinID"].dropna().unique()
                   if v != 0]
        if src_ids:
            mask = mask | self.fin_df["FinID"].isin(src_ids)
        self.fin_df.loc[mask, "FinID"] = fid

        # 统一该 FinID 的 DolphinID: 优先用目标组已有的 DolphinID,
        # 否则保留源组的 DolphinID, 并把它传播给该 FinID 的全部行
        fin_mask = self.fin_df["FinID"] == fid
        dph_values = self.fin_df.loc[fin_mask, "DolphinID"].dropna().unique()
        if len(dph_values) > 0:
            # 目标行 (非被拖入的行) 的 DolphinID 优先
            target_mask = fin_mask & ~mask
            target_dph = self.fin_df.loc[target_mask, "DolphinID"].dropna().unique()
            chosen = str(target_dph[0]) if len(target_dph) else str(dph_values[0])
            self.fin_df.loc[fin_mask, "DolphinID"] = chosen

        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        # 返回所有实际被移动的鳍, 让界面把它们一起换组
        moved_paths = self.fin_df.loc[mask, "fullpath"].dropna()
        return {p: p for p in moved_paths}

    def merge_fin_paths(self, src_paths, target_path):
        """总览模式拖放合并: 被拖鳍所在 FinID 并入目标鳍所在 FinID
        (对应 DolphinID 一并合并)。返回被合并的源 FinID 键列表。"""
        df = self.fin_df
        if df.empty or "FinID" not in df.columns \
                or "fullpath" not in df.columns:
            return []
        keys = df["FinID"].map(self._fid_key).to_numpy()
        paths = df["fullpath"].to_numpy()

        def fid_of(p):
            hit = np.where(paths == p)[0]
            return keys[hit[0]] if len(hit) else None

        dst = fid_of(target_path)
        if dst is None or dst in ("0", "nan", "None"):
            return []
        merged = []
        for p in src_paths:
            src = fid_of(p)
            if src and src != dst and src not in ("0", "nan", "None"):
                self.merge_fin_ids(src, dst)
                merged.append(src)
        return merged

    def merge_fin_ids(self, src_key, dst_key):
        """把 FinID=src_key 的所有鳍并入 dst_key, 对应 DolphinID 一并合并
        (目标已有 DolphinID 时用目标名, 否则沿用源名; 源 DolphinID 名下的
        其它鳍如对侧 DR 也归入目标海豚), 写回 CSV。"""
        df = self.fin_df
        if df.empty or "FinID" not in df.columns:
            return
        keys = df["FinID"].map(self._fid_key)
        src_mask = (keys == src_key).to_numpy()
        dst_mask = (keys == dst_key).to_numpy()
        if not src_mask.any() or not dst_mask.any():
            return
        if "DolphinID" not in df.columns:
            df["DolphinID"] = pd.NA
        dph_dst = df.loc[dst_mask, "DolphinID"].dropna().unique()
        dph_src = df.loc[src_mask, "DolphinID"].dropna().unique()
        df.loc[src_mask, "FinID"] = df.loc[dst_mask, "FinID"].iloc[0]  # 保持目标列类型
        name = str(dph_dst[0]) if len(dph_dst) else \
            (str(dph_src[0]) if len(dph_src) else None)
        if name is not None:
            if len(dph_src) and str(dph_src[0]) != name:
                df.loc[df["DolphinID"] == dph_src[0], "DolphinID"] = name
            df.loc[df["FinID"].map(self._fid_key) == dst_key,
                   "DolphinID"] = name
        self._save_csv(df, "FIN_METAINFO.csv")

    def move_track_to_fin_id(self, paths, group):
        """改个体分组 (勾选最佳鳍图时的侧边拖放): 只更新同 track
        (ShotID + TrackIDInShot) 的鳍的 FinID, 不动整个个体。"""
        m = re.match(r"FinID(.+)$", group)
        if not m:
            return {}
        fid = m.group(1)
        try:  # FinID 为数值列时保持数值类型
            fid = int(float(fid))
        except ValueError:
            pass
        if "FinID" not in self.fin_df.columns:
            self.fin_df["FinID"] = pd.NA
        mask = self.fin_df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        # 扩展到同 track 的所有鳍 (TrackIDInShot 只在 shot 内唯一, 需带 ShotID)
        if "TrackIDInShot" in self.fin_df.columns \
                and "ShotID" in self.fin_df.columns:
            key_rows = self.fin_df.loc[
                mask, ["ShotID", "TrackIDInShot"]].dropna()
            pairs = set(zip(key_rows["ShotID"], key_rows["TrackIDInShot"]))
            if pairs:
                row_keys = pd.Series(
                    list(zip(self.fin_df["ShotID"],
                             self.fin_df["TrackIDInShot"])),
                    index=self.fin_df.index)
                mask = mask | row_keys.isin(pairs)
        self.fin_df.loc[mask, "FinID"] = fid
        self._save_csv(self.fin_df, "FIN_METAINFO.csv")
        # 返回所有实际被移动的鳍, 让界面把它们一起换组
        moved_paths = self.fin_df.loc[mask, "fullpath"].dropna()
        return {p: p for p in moved_paths}

    # ------------------------------------------------------------------
    # 三栏模式排除区: 相似度置 0 并持久化到 FIN_SIMILARITY.npy
    # ------------------------------------------------------------------
    def _zero_sim_between(self, ids_a, ids_b):
        """把两组 FinID 之间的两两相似度置 0, 写回 FIN_SIMILARITY.npy。"""
        sim = self._similarity_matrix()
        if sim is None or "FinID" not in self.fin_df.columns:
            return
        fids = self.fin_df["FinID"].to_numpy()
        ia = np.where(np.isin(fids, [float(i) for i in ids_a]))[0]
        ib = np.where(np.isin(fids, [float(i) for i in ids_b]))[0]
        if not len(ia) or not len(ib):
            return
        sim[np.ix_(ia, ib)] = 0
        sim[np.ix_(ib, ia)] = 0
        sim_file = os.path.join(self.meta_dir, "FIN_SIMILARITY.npy")
        if os.path.isfile(sim_file):
            bdir = os.path.join(self.meta_dir, "backup")
            os.makedirs(bdir, exist_ok=True)
            shutil.copy2(sim_file, os.path.join(
                bdir, "FIN_SIMILARITY.npy." +
                datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"))
            np.save(sim_file, sim)
        self._invalidate_cache()

    def exclude_to_new_fin_id(self, paths, cur_group):
        """当前组的鳍拖入排除区: 同 track 的鳍赋予新 FinID (现有最大 +1),
        新旧 FinID 间相似度置 0, 写回 CSV 和 npy。返回 {path: path}。"""
        m = re.match(r"FinID(.+)$", cur_group or "")
        if not m:
            return {}
        try:
            cur_fid = int(float(m.group(1)))
        except ValueError:
            return {}
        df = self.fin_df
        mask = df["fullpath"].isin(paths)
        if not mask.any():
            return {}
        # 同 track 扩展 (TrackIDInShot 只在 shot 内唯一, 需带 ShotID)
        if "TrackIDInShot" in df.columns and "ShotID" in df.columns:
            key_rows = df.loc[mask, ["ShotID", "TrackIDInShot"]].dropna()
            pairs = set(zip(key_rows["ShotID"], key_rows["TrackIDInShot"]))
            if pairs:
                row_keys = pd.Series(
                    list(zip(df["ShotID"], df["TrackIDInShot"])),
                    index=df.index)
                mask = mask | row_keys.isin(pairs)
        nums = pd.to_numeric(df["FinID"], errors="coerce")
        new_fid = int(nums.max()) + 1 if nums.notna().any() else 1
        df.loc[mask, "FinID"] = new_fid
        self._save_csv(df, "FIN_METAINFO.csv")
        self._zero_sim_between({new_fid}, {cur_fid})
        moved = df.loc[mask, "fullpath"].dropna()
        return {p: p for p in moved}

    def exclude_fin_id_pair(self, paths, cur_group):
        """左下相似鳍拖入排除区: 被拖鳍的 FinID 与当前 FinID 的
        两两相似度置 0 (不改 FinID), 写回 npy。"""
        m = re.match(r"FinID(.+)$", cur_group or "")
        if not m:
            return
        try:
            cur_fid = int(float(m.group(1)))
        except ValueError:
            return
        mask = self.fin_df["fullpath"].isin(paths)
        src = [v for v in self.fin_df.loc[mask, "FinID"].dropna().unique()
               if v != 0]
        if src:
            self._zero_sim_between(src, {cur_fid})
