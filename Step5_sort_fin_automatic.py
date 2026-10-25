# coding: utf-8
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from KalmanBoxTracker import KalmanBoxTracker

class FinSorter:
    """Automatically cluster fin features based on cosine similarity and shot group."""

    DEFAULT_THRESHOLD = 0.70
    DEFAULT_SIMILARITY_MATCH = 1.0
    DEFAULT_SIMILARITY_EXCLUDE = 0.0
    DEFAULT_GATE_DIST = 300  # pixels; gating for Hungarian assignment
    DEFAULT_MAX_AGE = 6  # frames a track may go unmatched before termination

    def __init__(
        self,
        root_dir,
        metainfo_csv='METAINFO/FIN_METAINFO.csv',
        deepfeatures='METAINFO/FIN_DEEPFEATURES.npy',
        similarity_npy='METAINFO/FIN_SIMILARITY.npy',
        threshold=DEFAULT_THRESHOLD,
    ):
        """
        Args:
            root_dir: Root directory of the project.
            metainfo_csv: Relative path to the metadata CSV.
            similarity_npy: Relative path for the output similarity matrix.
            threshold: Cosine similarity threshold for clustering.
        """
        self.root_dir = root_dir
        self.metainfo_csv = os.path.join(root_dir, metainfo_csv)
        self.deepfeatures = os.path.join(root_dir, deepfeatures)
        self.similarity_npy = os.path.join(root_dir, similarity_npy)
        self.threshold = threshold

        self.metainfo = None
        self.features = None
        self.similarity = None
        self.fin_id_list = None

    def load_data(self):
        """Load metadata and deep features.
        """
        self.metainfo = pd.read_csv(self.metainfo_csv)  # , index_col=0)

        npy_path = self.deepfeatures
        if not os.path.isfile(npy_path):
            raise FileNotFoundError("Feature file not found: %s" % npy_path)
        self.features = np.load(npy_path)

    def filter_fin_by_clearness(self, clearness_threshold, crop_conf_threshold):
        """Compute statistics after applying thresholds."""
        self.metainfo["Select"] = (
          (self.metainfo["Clearness"] > clearness_threshold)
          * (self.metainfo["CropConf"] > crop_conf_threshold)
        )
        self.metainfo["Clear"] = (
            self.metainfo["Clearness"] > clearness_threshold
        )

    def compute_similarity(self):
        """Compute cosine similarity matrix between all features."""
        feats = self.features
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        normalized = feats / norms
        self.similarity = np.dot(normalized, normalized.T)

    def correct_fin_class_by_shot(self, class_col="AssumedAspect",
                                  out_col="ConfirmedAspect"):
        """
        Correct DL/DR misclassifications: all fins within one shot (burst)
        should share the same side class. Each shot votes, weighted by
        crop_conf; fins disagreeing with the winning class are corrected.
        The original column is kept; corrections go to `out_col`.
        'Others' (unknown side) neither votes nor gets corrected.
        """
        metadata = self.metainfo
        if class_col not in metadata.columns:
            print("No %s column found, skip class correction" % class_col)
            return
        if "ShotID" not in metadata.columns:
            print("No shot_id column found, skip class correction")
            return
        corrected = metadata[class_col].copy()
        flip_count = 0
        mixed_shot_count = 0
        for shot_id, shot in metadata.groupby("ShotID"):
            votes = {}
            for _, row in shot.iterrows():
                cls = row[class_col]
                if cls not in ("DL", "DR"):
                    continue
                votes[cls] = votes.get(cls, 0.0) + float(row["CropConf"])
            if len(votes) < 2:
                continue  # consistent shot or no DL/DR at all
            mixed_shot_count += 1
            winner = max(votes, key=votes.get)
            for idx, row in shot.iterrows():
                if row[class_col] in ("DL", "DR") and row[class_col] != winner:
                    corrected[idx] = winner
                    flip_count += 1
        metadata[out_col] = corrected
        print("Class correction: %d fins flipped in %d mixed shots "
              "(see column %s)" % (flip_count, mixed_shot_count, out_col))

    @staticmethod
    def _compute_center_dist(box_a, box_b):
        """Compute pixel distance between centers of two boxes
        given as (x_min, y_min, x_max, y_max)."""
        cx_a = (box_a[0] + box_a[2]) / 2
        cy_a = (box_a[1] + box_a[3]) / 2
        cx_b = (box_b[0] + box_b[2]) / 2
        cy_b = (box_b[1] + box_b[3]) / 2
        return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

    def _load_cumulative_offsets(self):
        """
        Load per-frame global (whole-image) motion from
        METAINFO/PHOTO_METAINFO.csv (written by Step1b_analyse_photo_jitter.py )
        and cumulate it within each shot relative to the shot's first
        frame. ecc offsets are preferred; pc is the fallback. Offsets map
        previous-frame coordinates to current-frame coordinates, so a
        detection box in frame k is moved into the shot's first-frame
        reference by subtracting its cumulative offset.

        Returns:
            dict {photo_name: (cum_dx, cum_dy)}, or None when no
            jitter data is available (tracking then runs uncompensated).
        """
        path = os.path.join(self.root_dir, "METAINFO", "PHOTO_METAINFO.csv")
        if not os.path.exists(path):
            print("No PHOTO_METAINFO.csv, track without motion compensation")
            return None
        df = pd.read_csv(path)
        if "ecc_dx" in df.columns:
            dx, dy = df["ecc_dx"].copy(), df["ecc_dy"].copy()
            if "pc_dx" in df.columns:
                dx = dx.fillna(df["pc_dx"])
                dy = dy.fillna(df["pc_dy"])
        elif "pc_dx" in df.columns:
            dx, dy = df["pc_dx"], df["pc_dy"]
        else:
            print("No jitter columns, track without motion compensation")
            return None
        df["_dx"] = dx.fillna(0.0)
        df["_dy"] = dy.fillna(0.0)
        offsets = {}
        # filenames are sequential in time within a burst
        for _, group in df.groupby("ShotID"):
            group = group.sort_values("PhotoName")
            cum_dx = group["_dx"].cumsum()
            cum_dy = group["_dy"].cumsum()
            for name, cx, cy in zip(group["PhotoName"], cum_dx, cum_dy):
                offsets[name] = (float(cx), float(cy))
        return offsets

    def automatic_link_fin_by_shot_group(self,
                                        gate_dist=DEFAULT_GATE_DIST,
                                        max_age=DEFAULT_MAX_AGE,
                                        same_class_only=False):
        """
        Must-link: track fins frame-by-frame within each burst (shot_id)
        with a SORT-style tracker (constant-velocity Kalman filter +
        Hungarian assignment). Whole-image motion (camera shake, boat
        motion) is compensated first with the per-frame jitter offsets
        from PHOTO_METAINFO.csv (see _load_cumulative_offsets), so the
        Kalman filter only has to model dolphin motion; Hungarian
        enforces one-to-one matching so one fin cannot be linked to two
        different dolphins. All members of a track get pairwise
        similarity 1 so clustering always links them.
        Args:
            gate_dist: Max pixel distance between the predicted center and
                a detection center for a match candidate.
            max_age: Frames a track may go unmatched before termination.
            same_class_only: If True and a class column exists (DL/DR),
                only link fins of the same class. Uses 'class_corrected'
                when available (see correct_fin_class_by_shot), otherwise
                the raw 'class' column.
        """
        print("Linking fin based on shot group with Kalman + Hungarian. ")
        class_col = "Aspect"
        img_col = "PhotoName"
        if "ShotID" not in self.metainfo.columns:
            print("No ShotID column found, skip must-link")
            return
        offsets = self._load_cumulative_offsets()

        track_id_list = np.zeros(len(self.metainfo), dtype=np.int32)
        track_count = 0
        for shot_id in sorted(self.metainfo["ShotID"].unique()):
            shot = self.metainfo[self.metainfo["ShotID"] == shot_id]
            tracks = self._track_shot(shot, img_col, class_col,
                                      gate_dist, max_age, same_class_only,
                                      offsets)
            for track in tracks:
                track_count += 1
                for fin in track:
                    track_id_list[fin] = track_count
                if len(track) < 2:
                    continue
                for i in range(len(track)):
                    for j in range(i + 1, len(track)):
                        self.similarity[track[i], track[j]] = 1
                        self.similarity[track[j], track[i]] = 1
        self.metainfo["TrackIDInShot"] = track_id_list
        print("Total %d tracks in %d shot"
              % (track_count, len(self.metainfo["ShotID"].unique())))


    def _track_shot(self, shot, photo_col, class_col,
                    gate_dist, max_age, same_class_only, offsets=None):
        """Run the Kalman + Hungarian tracker over one burst.
        When offsets (see _load_cumulative_offsets) are given, detection
        boxes are moved into the shot's first-frame reference first, so
        the tracker sees dolphin motion with whole-image jitter removed.
        Returns a list of tracks; each track is a list of fin indices."""
        # photo filenames are sequential in time within a burst
        photo_names = sorted(shot[photo_col].unique())
        active = []    # list of (tracker, member_indices, cls)
        finished = []  # member_indices of terminated tracks
        for photo in photo_names:
            dets = shot[shot[photo_col] == photo]
            det_idx = list(dets.index)
            ox, oy = offsets.get(photo, (0.0, 0.0)) if offsets else (0.0, 0.0)
            det_boxes = [(row["XMin"] - ox, row["YMin"] - oy,
                          row["XMax"] - ox, row["YMax"] - oy)
                         for _, row in dets.iterrows()]
            pred_boxes = [trk.predict() for trk, _, _ in active]
            matched, un_dets = self._assign(
                pred_boxes, det_boxes, active, dets, class_col,
                gate_dist, same_class_only)
            for trk_i, det_i in matched:
                active[trk_i][0].update(det_boxes[det_i])
                active[trk_i][1].append(det_idx[det_i])
            for det_i in un_dets:
                cls = (dets.iloc[det_i][class_col]
                       if class_col in dets.columns else None)
                active.append([KalmanBoxTracker(det_boxes[det_i]),
                               [det_idx[det_i]], cls])
            still_active = []
            for entry in active:
                if entry[0].time_since_update > max_age:
                    finished.append(entry[1])
                else:
                    still_active.append(entry)
            active = still_active
        finished.extend(members for _, members, _ in active)
        return finished

    def _assign(self, pred_boxes, det_boxes, active, dets, class_col,
                gate_dist, same_class_only):
        """Hungarian one-to-one assignment between predicted boxes and
        detections, gated by predicted-center distance (and optionally by
        class). Returns (matched_pairs, unmatched_det_indices)."""
        n_trk, n_det = len(pred_boxes), len(det_boxes)
        if n_det == 0:
            return [], []
        if n_trk == 0:
            return [], list(range(n_det))
        # large finite dummy cost: Hungarian requires a feasible matrix;
        # dummy matches are filtered out afterwards
        cost = np.full((n_trk, n_det), 1e6)
        for i, pred in enumerate(pred_boxes):
            for j, det in enumerate(det_boxes):
                if (same_class_only and class_col in dets.columns
                        and active[i][2] is not None
                        and dets.iloc[j][class_col] != active[i][2]):
                    continue
                dist = self._compute_center_dist(pred, det)
                if dist < gate_dist:
                    cost[i, j] = dist
        rows, cols = linear_sum_assignment(cost)
        matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e6]
        matched_dets = {c for _, c in matched}
        return matched, [j for j in range(n_det) if j not in matched_dets]

    def mark_best_in_shot(self, quality_col="Clearness"):
        """
        find the best quality fin for each track inside the same shot
        group. Quality score = clearness x crop_conf. Appends a boolean
        column "BestInShot": True for the highest-scoring fin.
        """
        best = pd.Series(False, index=self.metainfo.index)
        for shot_id, shot in self.metainfo.groupby("ShotID"):
            for track_id, track in shot.groupby("TrackIDInShot"):
                # what about find the highest clearnes if CropConf higher than xxx
                score = track[quality_col] * track["CropConf"]
                if score.isna().all():
                    continue
                best[score.idxmax()] = True
        self.metainfo["BestInShot"] = best
        print("Best in shot: %d marked out of %d fins"
              % (int(best.sum()), len(self.metainfo)))

    def automatic_link_fin_by_similarity(self, index):
        """
        Link a feature at `index` to similar features based on cosine similarity.
        update current fin ID assignments (modified in-place).
        Args:
            index: Feature index to process.
        """
        high_similar_fin_list = []
        
        # compare similarity between current index and others
        for i in range(0, len(self.similarity)):
            if self.similarity[index, i] > self.threshold:
                high_similar_fin_list.append(i)
        if len(high_similar_fin_list) == 1:  # if only itself
            if self.fin_id_list[ high_similar_fin_list[0]] != 0:
                # if this fin already has fin_id, do nothing
                return 

        # find all assigned id in high_similar_fin_list
        assigned_id = []
        for i in high_similar_fin_list:
            fin_id = self.fin_id_list[i]
            if not (fin_id == 0):
                if not (fin_id in assigned_id):
                    assigned_id.append(fin_id)

        # label fin
        if len(assigned_id) == 0: # all the fin image haven't fin id
            prev_fin_id = np.max(self.fin_id_list)
            for i in range(1, prev_fin_id + 2):
                if not (i in self.fin_id_list):
                    # find the fin id without being assigned
                    cur_fin_id = i
                    break
        elif len(assigned_id) == 1: # only one unique fin id was recorded
            cur_fin_id = assigned_id[0]
        else: # more than one unique fin id was recorded, renew with minial id
            cur_fin_id = np.min(assigned_id)
            # replace old fin id even fin image don't show on high_similar_fin_list
            assigned_id.remove(cur_fin_id)
            for fin_id in assigned_id:
                self.fin_id_list[self.fin_id_list == fin_id] = cur_fin_id
        for i in high_similar_fin_list:
            # assign fin id to new found high similar fin with/without fin id
            self.fin_id_list[i] = cur_fin_id 

    def cluster(self):
        """Run automatic clustering.

        When BestInShot/TrackIDInShot columns are available, only the
        best fin of each track is compared with the best fins of other
        tracks; when two representatives exceed the similarity threshold,
        every fin sharing their TrackIDInShot gets the same FinID.
        Otherwise falls back to linking every fin image.
        """
        self.fin_id_list = np.zeros(len(self.features), dtype=np.int32)
        self._cluster_by_best_in_shot()
        print("Unclassified /Total fin image:", 
            "%d/%d."%(np.sum(self.fin_id_list == 0), len(self.features)))
        print("Unique FinID number:", len(np.unique(self.fin_id_list)) - 1)

    def _cluster_by_best_in_shot(self):
        """Cluster tracks through their BestInShot representatives.

        Only fins marked BestInShot are compared (cosine similarity);
        merging is transitive (union-find), and a merged pair shares its
        FinID with all fins that have the same TrackIDInShot.
        Only fins which are DL/DR and Clear participate, and merging
        only happens between fins of the same aspect (DL with DL, DR with
        DR); the FinID is propagated only to same-aspect fins in the
        merged tracks. Everything else is left unclassified (FinID 0).
        """
        best_mask = (self.metainfo["BestInShot"].astype(str).str.lower()
                     .isin(["true", "1"]))
        aspect_mask = self.metainfo["Aspect"].isin(["DL", "DR"]).to_numpy()
        clear_mask = self.metainfo["Clear"].isin([1]).to_numpy()
        candidate_mask = best_mask & aspect_mask & clear_mask

        candidate_idx = np.flatnonzero(candidate_mask.to_numpy())
        classes = self.metainfo["Aspect"].to_numpy()
        parent = {idx: idx for idx in candidate_idx}

        def find(idx):
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        for idx in candidate_idx:
            similar = candidate_idx[self.similarity[idx, candidate_idx] > self.threshold]
            for other in similar:
                if classes[idx] != classes[other]:
                    continue  # 只允许同 aspect (DL-DL / DR-DR) 之间合并
                parent[find(idx)] = find(other)

        groups = {}
        for idx in candidate_idx:
            groups.setdefault(find(idx), []).append(idx)
        track_col = self.metainfo["TrackIDInShot"]
        fin_id = 0
        for members in groups.values():
            fin_id += 1
            cls = classes[members[0]]  # 同组鳍的 aspect 必然相同
            track_ids = track_col.iloc[members]
            # 只把 FinID 传播给同 track 且同 aspect 的鳍
            member_mask = (track_col.isin(track_ids).to_numpy()
                           & (classes == cls))
            self.fin_id_list[member_mask] = fin_id
    
    def normalize_same_fin_similarity(self):
        """Set similarity of same-fin pairs to 1."""
        for fin_id in np.unique(self.fin_id_list[self.fin_id_list != 0]):
            fin_has_same_id_list = []
            for fin_idx in range(len(self.fin_id_list)):
                if self.fin_id_list[fin_idx] == fin_id:
                    fin_has_same_id_list.append(fin_idx)
            for fin in fin_has_same_id_list:
                self.similarity[fin, fin_has_same_id_list] = 1
        #print("Connected node number:", np.sum(self.similarity == 1))

    def exclude_same_photo_duplicates(self):
        """
        Exclude multiple fins detected from the same original photo
        by zeroing their cross-similarity.
        """
        photo_list = self.metainfo["PhotoName"].unique()
        occurred_number = 0
        for photo in photo_list:
            fin_idx_list = self.metainfo.index[
                self.metainfo["PhotoName"] == photo
            ].values
            fin_number = len(fin_idx_list)
            if fin_number > 1:
                occurred_number = occurred_number + 1
                for i in range(fin_number):
                    fin_idx_i = fin_idx_list[i]
                    for j in range(i + 1, fin_number):
                        fin_idx_j = fin_idx_list[j]
                        self.similarity[fin_idx_i, fin_idx_j] = 0
                        self.similarity[fin_idx_j, fin_idx_i] = 0
        print("Found %s photos have multiple fins" % (occurred_number))

    def save_results(self):
        """Save FinID to metainfo CSV."""
        print("Update FIN_METAINFO.csv")
        self.metainfo["FinID"] = self.fin_id_list
        self.metainfo.to_csv(self.metainfo_csv, index=False)
        print("Save FIN_SIMILARITY.npy")
        np.save(self.similarity_npy, self.similarity)

    def sort(self):
        """Execute the full sorting pipeline."""
        self.load_data()
        #self.filter_fin_by_clearness(
        #        clearness_threshold=0.15, crop_conf_threshold=0.4)
        self.compute_similarity()
        self.exclude_same_photo_duplicates()
        #self.correct_fin_class_by_shot()
        self.automatic_link_fin_by_shot_group()
        self.mark_best_in_shot()
        self.cluster()
        self.normalize_same_fin_similarity()
        self.save_results()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
    sorter = FinSorter(root_dir=root_dir)
    sorter.sort()
