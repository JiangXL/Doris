"""归档: 深度特征重算相关方法 (原 doris_dataset.Dataset 的方法)。

2026-11-26 从 doris_dataset.py 移出, 已无任何调用方:
- replace_feature_for_box: 鳍框编辑后重算该鳍深度特征并更新相似度矩阵,
  原供后台特征重算任务调用; 该任务已移除 (改为占位矩阵 + 用户直接指定 FinID)。
- extract_feature: 单张裁剪鳍图提特征, 仅 replace_feature_for_box 调用。
- _replace_similarity_row: 替换特征向量并重算矩阵行列, 仅 replace_feature_for_box 调用。

保留作历史参考; 如需恢复, 把下面方法贴回 Dataset 类内即可
(依赖 os/threading/numpy 及 Dataset._find_fin_row / _ensure_matrices /
 _save_csv / _invalidate_cache, 这些仍在 doris_dataset.py 中)。
不要 import 本模块。
"""


class Dataset:
    """方法归档容器 (不可实例化使用, 仅保存代码)。"""

    def extract_feature(self, crop_path):
        """对单个裁剪鳍图提取深度特征向量 (numpy 1-D array)。"""
        if not os.path.isfile(crop_path):
            return None
        if not hasattr(self, "_feat_lock"):
            self._feat_lock = threading.Lock()
        with self._feat_lock:  # 首次加载 ONNX 模型较慢, 防止并发初始化
            if getattr(self, "_deep_feat_extractor", None) is None:
                from Step3_extract_fin_feature import FinDeepFeatureExtractor
                self._deep_feat_extractor = \
                    FinDeepFeatureExtractor().feature_extractor
        extractor = self._deep_feat_extractor
        feat = extractor.extract([crop_path], batch_size=1)
        return feat[0] if feat is not None and len(feat) else None


    def replace_feature_for_box(self, path, box):
        """鳍框被编辑后, 重算该鳍的深度特征并更新相似度矩阵 (供后台任务调用)。
        优先按 box[6] (identity) 定位行, 否则按 PhotoName+坐标。"""
        if self.fin_df.empty or "PhotoName" not in self.fin_df.columns:
            return
        idx = self._find_fin_row(path, box)
        if idx is None:
            return
        full = self.fin_df.at[idx, "fullpath"] \
            if "fullpath" in self.fin_df.columns \
            else os.path.join(self.root, str(self.fin_df.at[idx, "Path"]))
        feat = self.extract_feature(full)
        if feat is None:
            return
        self._replace_similarity_row(idx, np.asarray(feat).reshape(1, -1))


    def _replace_similarity_row(self, idx, f_new):
        """替换第 idx 个鳍的特征向量, 重算相似度/关联矩阵的第 idx 行/列
        (就地更新常驻成员 similarity_matrix / connection_matrix 并持久化)。"""
        feat_file = os.path.join(self.meta_dir, "FIN_DEEPFEATURES.npy")
        sim_file = os.path.join(self.meta_dir, "FIN_SIMILARITY.npy")
        conn_file = os.path.join(self.meta_dir, "FIN_CONNECTION.npy")
        if not os.path.isfile(feat_file):
            return
        if not hasattr(self, "_feat_lock"):
            self._feat_lock = threading.Lock()
        with self._feat_lock:
            if self._ensure_matrices() is None \
                    or self.similarity_matrix is None:
                return
            features = np.load(feat_file)
            sim = self.similarity_matrix
            conn = self.connection_matrix
            N = len(features)
            if not (0 <= idx < N) or sim.shape != (N, N):
                return
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_new = float(np.linalg.norm(f_new))
            if norm_new == 0:
                norm_new = 1.0
            cross = np.dot(f_new / norm_new,
                           (features / norms).T).reshape(-1)
            features[idx] = f_new
            sim[idx, :] = cross
            sim[:, idx] = cross
            sim[idx, idx] = 1.0
            conn[idx, :] = cross
            conn[:, idx] = cross
            conn[idx, idx] = 1.0
            # same-photo 置 0 (编辑后的鳍与同一原图的其它鳍)
            name = str(self.fin_df["PhotoName"].iloc[idx])
            same = np.flatnonzero(
                self.fin_df["PhotoName"].to_numpy() == name)
            same = same[same != idx]
            if len(same):
                sim[idx, same] = 0.0
                sim[same, idx] = 0.0
            # CONNECTION: same-track 置 1
            if "TrackIDInShot" in self.fin_df.columns:
                track = self.fin_df["TrackIDInShot"].iloc[idx]
                shot = str(self.fin_df["ShotID"].iloc[idx])
                if not pd.isna(track):
                    mask = ((self.fin_df["ShotID"].astype(str) == shot)
                            & (self.fin_df["TrackIDInShot"] == track)
                            ).to_numpy().copy()  # CoW 下 to_numpy 可能只读
                    mask[idx] = False
                    tidx = np.flatnonzero(mask)
                    conn[idx, tidx] = 1.0
                    conn[tidx, idx] = 1.0
            np.save(feat_file, features)
            np.save(sim_file, sim)
            np.save(conn_file, conn)
        self._invalidate_cache()
