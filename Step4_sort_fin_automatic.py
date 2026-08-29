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


class _KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking one fin box (SORT-style).

    State: [cx, cy, s, r, vx, vy, vs] where (cx, cy) is the box center,
    s the box area and r the aspect ratio w/h. The velocity terms absorb
    consistent frame-to-frame displacement from camera shake and dolphin
    motion.
    """

    def __init__(self, box):
        self.x = np.zeros((7, 1))
        self.x[:4] = self._box_to_z(box)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self.Q = np.eye(7) * 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.time_since_update = 0

    @staticmethod
    def _box_to_z(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return np.array([box[0] + w / 2, box[1] + h / 2,
                         w * h, w / float(h)]).reshape((4, 1))

    @staticmethod
    def _x_to_box(x):
        cx, cy = x[0, 0], x[1, 0]
        s = max(x[2, 0], 1.0)
        w = np.sqrt(s * x[3, 0])
        h = s / w
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._x_to_box(self.x)

    def update(self, box):
        self.time_since_update = 0
        y = self._box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class FinSorter:
    """Automatically cluster fin features based on cosine similarity and shot group."""

    DEFAULT_THRESHOLD = 0.85
    DEFAULT_SIMILARITY_MATCH = 1.0
    DEFAULT_SIMILARITY_EXCLUDE = 0.0
    DEFAULT_CENTER_DIST_THRESHOLD = 100  # pixels between adjacent frames
    DEFAULT_GATE_DIST = 200  # pixels; gating for Hungarian assignment
    DEFAULT_MAX_AGE = 1  # frames a track may go unmatched before termination

    def __init__(
        self,
        root_dir,
        metainfo_csv='METAINFO/FIN_METAINFO.csv',
        deepfeatures_dir='METAINFO/FIN_DEEPFEATURES',
        output_csv='METAINFO/FIN_METAINFO_SELECTED.csv',
        similarity_npy='METAINFO/FIN_SIMILARITY.npy',
        threshold=DEFAULT_THRESHOLD,
    ):
        """
        Args:
            root_dir: Root directory of the project.
            metainfo_csv: Relative path to the metadata CSV.
            deepfeatures_dir: Relative path to the deep features directory.
            output_csv: Relative path for the output selected metadata CSV.
            similarity_npy: Relative path for the output similarity matrix.
            threshold: Cosine similarity threshold for clustering.
        """
        self.root_dir = root_dir
        self.metainfo_csv = os.path.join(root_dir, metainfo_csv)
        self.deepfeatures_dir = os.path.join(root_dir, deepfeatures_dir)
        self.output_csv = os.path.join(root_dir, output_csv)
        self.similarity_npy = os.path.join(root_dir, similarity_npy)
        self.threshold = threshold

        self.full_metainfo = None
        self.features = None
        self.similarity = None
        self.fin_id_list = None

    def load_data(self):
        """Load metadata and deep features."""
        self.full_metainfo = pd.read_csv(self.metainfo_csv, index_col=0)
        self.features = FeatureDataset.from_file(self.deepfeatures_dir)

    def compute_similarity(self):
        """Compute cosine similarity matrix between all features."""
        matcher = CosineSimilarity()
        self.similarity = matcher(self.features, self.features)

    @staticmethod
    def stats_fin_id(_fin_id_list):
        """Count occurrences of each fin ID."""
        max_id = int(np.max(_fin_id_list))
        fin_counter = np.zeros(max_id + 1, dtype=np.int16)
        # fin_id count from 1, 0 means unclassified fins
        for i in range(len(_fin_id_list)):
            fin_id = int(_fin_id_list[i])
            fin_counter[fin_id] = fin_counter[fin_id] + 1
        return fin_counter

    def print_fin_distribution(self, _fin_id_list=None):
        """Print fin ID distribution to stdout."""
        if _fin_id_list is None:
            _fin_id_list = self.fin_id_list
        fin_stats = self.stats_fin_id(_fin_id_list)
        stats_text = "unclassified fin: %d\n" % (fin_stats[0])
        for i in range(1, len(fin_stats)):
            stats_text = stats_text + "fin %d: %d\n" % (i, fin_stats[i])
        print(stats_text)

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

    def automatic_link_fin_by_shot_grup(self,
                                        gate_dist=DEFAULT_GATE_DIST,
                                        max_age=DEFAULT_MAX_AGE,
                                        same_class_only=False):
        """
        Must-link: track fins frame-by-frame within each burst (shot_id)
        with a SORT-style tracker (constant-velocity Kalman filter +
        Hungarian assignment). The Kalman prediction absorbs camera shake
        and dolphin motion; Hungarian enforces one-to-one matching so one
        fin cannot be linked to two different dolphins. All members of a
        track get pairwise similarity 1 so clustering always links them.
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
        metadata = self.features.metadata
        if "class_corrected" in metadata.columns:
            class_col = "class_corrected"
        else:
            class_col = "class"
        # image name column differs between metainfo versions
        if "orig_img_name" in metadata.columns:
            img_col = "orig_img_name"
        else:
            print("No orig image column found, skip must-link")
            return
        if "shot_id" not in metadata.columns:
            print("No shot_id column found, skip must-link")
            return

        track_count = 0
        link_pair_count = 0
        for shot_id in sorted(metadata["shot_id"].unique()):
            shot = metadata[metadata["shot_id"] == shot_id]
            tracks = self._track_shot(shot, img_col, class_col,
                                      gate_dist, max_age, same_class_only)
            for members in tracks:
                if len(members) < 2:
                    continue
                track_count += 1
                link_pair_count += len(members) * (len(members) - 1) // 2
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        self.similarity[members[i], members[j]] = 1
                        self.similarity[members[j], members[i]] = 1
        print("Must-link: %d pairs in %d tracks"
              % (link_pair_count, track_count))

    def _track_shot(self, shot, img_col, class_col,
                    gate_dist, max_age, same_class_only):
        """Run the Kalman + Hungarian tracker over one burst.
        Returns a list of tracks; each track is a list of fin indices."""
        # image filenames are sequential in time within a burst
        img_names = sorted(shot[img_col].unique())
        active = []    # list of (tracker, member_indices, cls)
        finished = []  # member_indices of terminated tracks
        for img in img_names:
            dets = shot[shot[img_col] == img]
            det_idx = list(dets.index)
            det_boxes = [(row["x_min"], row["y_min"],
                          row["x_max"], row["y_max"])
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
                active.append([_KalmanBoxTracker(det_boxes[det_i]),
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
        """Save FinID back to features and export metadata CSV."""
        print("Save FIN_DEEPFEATURES and FIN_METAINFO_SELECTED.csv")
        self.features.metadata["FinID"] = self.fin_id_list
        self.features.save(self.deepfeatures_dir)
        self.features.metadata.to_csv(self.output_csv)
        print("Save FIN_SIMILARITY.npy")
        np.save(self.similarity_npy, self.similarity)

    def sort(self):
        """Execute the full sorting pipeline."""
        self.load_data()
        self.compute_similarity()
        self.correct_fin_class_by_shot()
        self.automatic_link_fin_by_shot_grup()
        self.cluster()
        self.normalize_same_fin_similarity()
        self.exclude_same_image_duplicates()
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
