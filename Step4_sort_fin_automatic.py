#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.data import ImageDataset
from wildlife_tools.data import FeatureDataset

#TODO: Read the focus position from exif 

class FinFeatureSorter:
    """Automatically sort/cluster fin features based on cosine similarity."""

    DEFAULT_THRESHOLD = 0.80
    DEFAULT_SIMILARITY_MATCH = 1.0
    DEFAULT_SIMILARITY_EXCLUDE = 0.0

    def __init__(
        self,
        root_dir,
        metainfo_csv='METAINFO/FIN_METAINFO.csv',
        deepfeatures_dir='METAINFO/FIN_DEEPFEATURES',
        output_csv='METAINFO/FIN_METAINFO_SELECTED.csv',
        similarity_npy='METAINFO/FIN_SIMILARITY.npy',
        threshold=DEFAULT_THRESHOLD,
        device='cuda',
    ):
        """
        Args:
            root_dir: Root directory of the project.
            metainfo_csv: Relative path to the metadata CSV.
            deepfeatures_dir: Relative path to the deep features directory.
            output_csv: Relative path for the output selected metadata CSV.
            similarity_npy: Relative path for the output similarity matrix.
            threshold: Cosine similarity threshold for clustering.
            device: Device for similarity computation.
        """
        self.root_dir = root_dir
        self.metainfo_csv = os.path.join(root_dir, metainfo_csv)
        self.deepfeatures_dir = os.path.join(root_dir, deepfeatures_dir)
        self.output_csv = os.path.join(root_dir, output_csv)
        self.similarity_npy = os.path.join(root_dir, similarity_npy)
        self.threshold = threshold
        self.device = device

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

    def automatic_link_fin(self, index, fin_id_list=None, threshold=None):
        """
        Link a feature at `index` to similar features based on cosine similarity.

        Args:
            index: Feature index to process.
            fin_id_list: Current fin ID assignments (modified in-place).
            threshold: Similarity threshold.

        Returns:
            Updated fin_id_list.
        """
        if fin_id_list is None:
            fin_id_list = self.fin_id_list
        if threshold is None:
            threshold = self.threshold

        high_similar_fin_list = []
        similarity = self.similarity

        for i in range(0, len(similarity)):
            if similarity[index, i] > threshold:
                high_similar_fin_list.append(i)
        if len(high_similar_fin_list) == 1:
            return fin_id_list

        existed_id = []
        for i in high_similar_fin_list:
            fin_id = fin_id_list[i]
            if not (fin_id == 0):
                if not (fin_id in existed_id):
                    existed_id.append(fin_id)

        # label fin
        if len(existed_id) == 0:
            prev_fin_id = np.max(fin_id_list)
            for i in range(1, prev_fin_id + 2):
                if not (i in fin_id_list):
                    cur_fin_id = i
                    break
        elif len(existed_id) == 1:
            cur_fin_id = existed_id[0]
        else:
            # replace old id
            cur_fin_id = np.min(existed_id)
            existed_id.remove(cur_fin_id)
            for fin_id in existed_id:
                fin_id_list[fin_id_list == fin_id] = cur_fin_id
        for i in high_similar_fin_list:
            fin_id_list[i] = cur_fin_id
        return fin_id_list

    def cluster(self):
        """Run automatic clustering over all features."""
        self.fin_id_list = np.zeros(len(self.features), dtype=np.int32)
        for i in range(len(self.similarity)):
            self.fin_id_list = self.automatic_link_fin(i)

        print("Unclassified /Total fin image:", 
            "%d/%d."%(np.sum(self.fin_id_list == 0), len(self.features)))
        print("Unique FinID number:", len(np.unique(self.fin_id_list)) - 1)

    def save_results(self):
        """Save FinID back to features and export metadata CSV."""
        self.features.metadata["FinID"] = self.fin_id_list
        self.features.save(self.deepfeatures_dir)
        self.features.metadata.to_csv(self.output_csv)
        print("Save FIN_DEEPFEATURES and FIN_METAINFO_SELECTED.csv")

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
        ori_image_list = self.features.metadata.orig_img.unique()
        occurred_number = 0
        for image in ori_image_list:
            fin_idx_list = self.features.metadata.index[
                self.features.metadata.orig_img == image
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

    def save_similarity(self):
        """Save the similarity matrix to disk."""
        print("Save FIN_SIMILARITY.npy")
        np.save(self.similarity_npy, self.similarity)

    def run(self):
        """Execute the full sorting pipeline."""
        self.load_data()
        self.compute_similarity()
        self.cluster()
        self.save_results()
        self.normalize_same_fin_similarity()
        self.exclude_same_image_duplicates()
        self.save_similarity()


if __name__ == '__main__':
    #root_dir = r'/media/filming/2025-白海豚/20240825-JM_02-3/'
    import sys
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
    sorter = FinFeatureSorter(root_dir=root_dir)
    sorter.run()
