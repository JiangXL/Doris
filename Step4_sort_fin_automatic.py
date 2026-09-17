#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.data import FeatureDataset
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
        deepfeatures_dir='METAINFO/FIN_DEEPFEATURES',
        output_csv='METAINFO/FIN_METAINFO.csv',
        similarity_npy='METAINFO/FIN_SIMILARITY.npy',
        threshold=DEFAULT_THRESHOLD,
    ):
        """
        Args:
            root_dir: Root directory of the project.
            metainfo_csv: Relative path to the metadata CSV.
            deepfeatures_dir: Relative path to the deep features directory.
            output_csv: Relative path for the output metadata CSV; new
                columns (e.g. FinID) are merged back into the full metadata.
            similarity_npy: Relative path for the output similarity matrix.
            threshold: Cosine similarity threshold for clustering.
        """
        self.root_dir = root_dir
        self.metainfo_csv = os.path.join(root_dir, metainfo_csv)
        self.deepfeatures_dir = os.path.join(root_dir, deepfeatures_dir)
        self.output_csv = os.path.join(root_dir, output_csv)
        self.similarity_npy = os.path.join(root_dir, similarity_npy)
        self.threshold = threshold

        self.metainfo = None
        self.features = None
        self.similarity = None
        self.fin_id_list = None

    def load_data(self):
        """Load metadata and deep features."""
        self.metainfo = pd.read_csv(self.metainfo_csv)#, index_col=0)
        self.features = FeatureDataset.from_file(self.deepfeatures_dir)

    def compute_similarity(self):
        """Compute cosine similarity matrix between all features."""
        matcher = CosineSimilarity()
        self.similarity = matcher(self.features, self.features)

    def correct_fin_class_by_shot(self, class_col="class",
                                  out_col="class_corrected"):
        """
        Correct DL/DR misclassifications: all fins within one shot (burst)
        should share the same side class. Each shot votes, weighted by
        crop_conf; fins disagreeing with the winning class are corrected.
        The original column is kept; corrections go to `out_col`.
        'Others' (unknown side) neither votes nor gets corrected.
        """
        metadata = self.features.metadata
        if class_col not in metadata.columns:
            print("No %s column found, skip class correction" % class_col)
            return
        if "shot_id" not in metadata.columns:
            print("No shot_id column found, skip class correction")
            return
        corrected = metadata[class_col].copy()
        flip_count = 0
        mixed_shot_count = 0
        for shot_id, shot in metadata.groupby("shot_id"):
            votes = {}
            for _, row in shot.iterrows():
                cls = row[class_col]
                if cls not in ("DL", "DR"):
                    continue
                votes[cls] = votes.get(cls, 0.0) + float(row["crop_conf"])
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
        METAINFO/IMAGE_METAINFO.csv (written by Step1b_burst_jitter.py)
        and cumulate it within each shot relative to the shot's first
        frame. ecc offsets are preferred; pc is the fallback. Offsets map
        previous-frame coordinates to current-frame coordinates, so a
        detection box in frame k is moved into the shot's first-frame
        reference by subtracting its cumulative offset.

        Returns:
            dict {orig_img_name: (cum_dx, cum_dy)}, or None when no
            jitter data is available (tracking then runs uncompensated).
        """
        path = os.path.join(self.root_dir, "METAINFO", "IMAGE_METAINFO.csv")
        if not os.path.exists(path):
            print("No IMAGE_METAINFO.csv, track without motion compensation")
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
        for _, group in df.groupby("shot_id"):
            group = group.sort_values("orig_img_name")
            cum_dx = group["_dx"].cumsum()
            cum_dy = group["_dy"].cumsum()
            for name, cx, cy in zip(group["orig_img_name"], cum_dx, cum_dy):
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
        from IMAGE_METAINFO.csv (see _load_cumulative_offsets), so the
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
        print("linking fin based on shot group with Kalman + Hungarian")
        class_col = "class"
        img_col = "orig_img_name"
        if "shot_id" not in self.metainfo.columns:
            print("No shot_id column found, skip must-link")
            return
        offsets = self._load_cumulative_offsets()

        track_id_list = np.zeros(len(self.metainfo), dtype=np.int32)
        track_count = 0
        for shot_id in sorted(self.metainfo["shot_id"].unique()):
            shot = self.metainfo[self.metainfo["shot_id"] == shot_id]
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
              % (track_count, len(self.metainfo["shot_id"].unique())))


    def _track_shot(self, shot, img_col, class_col,
                    gate_dist, max_age, same_class_only, offsets=None):
        """Run the Kalman + Hungarian tracker over one burst.
        When offsets (see _load_cumulative_offsets) are given, detection
        boxes are moved into the shot's first-frame reference first, so
        the tracker sees dolphin motion with whole-image jitter removed.
        Returns a list of tracks; each track is a list of fin indices."""
        # image filenames are sequential in time within a burst
        img_names = sorted(shot[img_col].unique()) #TODO: maybe sort by timestamp
        active = []    # list of (tracker, member_indices, cls)
        finished = []  # member_indices of terminated tracks
        for img in img_names:
            dets = shot[shot[img_col] == img]
            det_idx = list(dets.index)
            ox, oy = offsets.get(img, (0.0, 0.0)) if offsets else (0.0, 0.0)
            det_boxes = [(row["x_min"] - ox, row["y_min"] - oy,
                          row["x_max"] - ox, row["y_max"] - oy)
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

    def mark_best_in_shot(self, quality_col="clearness"):
        """
        find the best quality fin for each track inside the same shot
        group. Quality score = clearness x crop_conf. Appends a boolean
        column "BestInShot": True for the highest-scoring fin in each
        (shot_id, FinID) group; False for everything else.
        """
        best = pd.Series(False, index=self.metainfo.index)
        for shot_id, shot in self.metainfo.groupby("shot_id"):
            for track_id, track in shot.groupby("TrackIDInShot"):
                score = track[quality_col] * track["crop_conf"]
                if score.isna().all():
                    continue
                best[score.idxmax()] = True
        self.metainfo["BestInShot"] = best
        print("Best in shot: %d marked out of %d fins"
              % (int(best.sum()), len(self.metainfo)))

    def automatic_link_fin(self, index):
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
        """Run automatic clustering over all features."""
        self.fin_id_list = np.zeros(len(self.features), dtype=np.int32)
        for i in range(len(self.similarity)):
            self.automatic_link_fin(i)
        print("Unclassified /Total fin image:", 
            "%d/%d."%(np.sum(self.fin_id_list == 0), len(self.features)))
        print("Unique FinID number:", len(np.unique(self.fin_id_list)) - 1)
    
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

    def exclude_same_image_duplicates(self):
        """
        Exclude multiple fins detected from the same original image
        by zeroing their cross-similarity.
        """
        ori_image_list = self.features.metadata.orig_img_name.unique()
        occurred_number = 0
        for image in ori_image_list:
            fin_idx_list = self.features.metadata.index[
                self.features.metadata.orig_img_name == image
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
        print("Found %s images have multiple fins" % (occurred_number))

    def save_results(self):
        """Save FinID to metainfo CSV."""
        print("Update FIN_METAINFO.csv")
        self.metainfo["FinID"] = self.fin_id_list
        self.metainfo.to_csv(self.output_csv, index=False)
        print("Save FIN_SIMILARITY.npy")
        np.save(self.similarity_npy, self.similarity)

    def sort(self):
        """Execute the full sorting pipeline."""
        self.load_data()
        self.compute_similarity()
        self.exclude_same_image_duplicates()
        #self.correct_fin_class_by_shot()
        self.automatic_link_fin_by_shot_group()
        self.mark_best_in_shot()
        self.cluster()
        self.normalize_same_fin_similarity()
        self.save_results()

if __name__ == '__main__':
    #root_dir = r'/media/filming/2025-白海豚/20240825-JM_02-3/'
    import sys
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
    sorter = FinSorter(root_dir=root_dir)
    sorter.sort()
