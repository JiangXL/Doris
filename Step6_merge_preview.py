#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import multiprocessing.connection
from pathlib import Path
from matplotlib import pyplot as plt
from wildlife_tools.data import FeatureDataset


class FinMergePreview:
    """
    Interactive fin merge preview server.

    Builds a preview of the clearest image per FinID, communicates with
    a GUI over a multiprocessing pipe for merge decisions, then
    relabels, saves results, and organizes output folders.
    """

    DEFAULT_HOST = 'localhost'
    DEFAULT_PORT = 1126
    DEFAULT_AUTHKEY = b'dolphin'

    def __init__(
        self,
        root_dir,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        authkey=DEFAULT_AUTHKEY,
        deepfeatures_path='METAINFO/FIN_DEEPFEATURES',
        similarity_path='METAINFO/FIN_SIMILARITY.npy',
        output_features_path='METAINFO/FIN_DEEPFEATURES_MERGED',
        output_csv_path='METAINFO/FIN_METAINFO_SELECTED_MERGED.csv',
        output_plot_path='METAINFO/FinID_statistics.png',
        fin_output_dir='FIN',
    ):
        """
        Args:
            root_dir: Root project directory.
            host: Listener host address.
            port: Listener port.
            authkey: Authentication key for the pipe.
            deepfeatures_path: Relative path to deep features.
            similarity_path: Relative path to similarity .npy file.
            output_features_path: Relative path for merged deep features.
            output_csv_path: Relative path for merged metadata CSV.
            output_plot_path: Relative path for FinID histogram plot.
            fin_output_dir: Relative directory for organized fin folders.
        """
        self.root_dir = root_dir
        self.host = host
        self.port = port
        self.authkey = authkey
        self.deepfeatures_path = os.path.join(root_dir, deepfeatures_path)
        self.similarity_path = os.path.join(root_dir, similarity_path)
        self.output_features_path = os.path.join(root_dir, output_features_path)
        self.output_csv_path = os.path.join(root_dir, output_csv_path)
        self.output_plot_path = os.path.join(root_dir, output_plot_path)
        self.fin_output_dir = os.path.join(root_dir, fin_output_dir)

        self.features = None
        self.similarity = None
        self.preview_fin = None
        self.updated_fin_id_list = None
        self.receiver = None

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
            _fin_id_list = self.updated_fin_id_list
        fin_stats = self.stats_fin_id(_fin_id_list)
        stats_text = "unclassified fin: %d\n" % (fin_stats[0])
        for i in range(1, len(fin_stats)):
            stats_text = stats_text + "fin %d: %d\n" % (i, fin_stats[i])
        print(stats_text)

    def load_data(self):
        """Load deep features and similarity matrix."""
        self.features = FeatureDataset.from_file(self.deepfeatures_path)
        print("Total FinID count:", len(self.features.metadata.FinID.unique()))
        self.updated_fin_id_list = self.features.metadata.FinID.values.copy()
        self.similarity = np.load(self.similarity_path)

    def build_preview(self):
        """Build preview DataFrame with clearest image per FinID."""
        uniqued_fin_id = self.features.metadata.FinID[
            self.features.metadata.FinID != 0
        ].unique()
        preview_fin = pd.DataFrame(columns=["idx", "path", "annotation"])
        for fin in uniqued_fin_id:
            fin_group = self.features.metadata.query("FinID==%d" % fin)
            name = fin_group.loc[fin_group['clearness'].idxmax(), 'path']
            path = os.path.join(self.root_dir, name)
            row = {
                "idx": fin_group['clearness'].idxmax(),
                "path": path,
                "annotation": "FinID%d count: %s\n%s" % (fin, len(fin_group), name),
            }
            preview_fin.loc[len(preview_fin)] = row
        self.preview_fin = preview_fin

    def sort_preview_similarity(self):
        """Sort preview list so that the most similar fins are adjacent."""
        preview_fin = self.preview_fin
        preview_num = len(preview_fin)
        idx_values = preview_fin['idx'].values
        similarity = self.similarity
        for cur_i in range(0, preview_num - 1):
            fin_idx_similarity = similarity[
                idx_values[cur_i], idx_values[cur_i + 1:preview_num]
            ]
            found_i_with_max_similarity = (
                np.argmax(fin_idx_similarity) + cur_i + 1
            )
            # place most similar fin next to current fin
            if found_i_with_max_similarity != (cur_i + 1):
                preview_fin.iloc[[cur_i + 1, found_i_with_max_similarity]] = (
                    preview_fin.iloc[[found_i_with_max_similarity, cur_i + 1]].values
                )

    def sort_preview_by_count(self, ascending=False):
        """
        Sort preview list by how many images belong to each FinID.

        Args:
            ascending: If True, sort from fewest to most occurrences.
                       If False (default), sort from most to fewest.
        """
        preview_fin = self.preview_fin.copy()
        updated_fin_id_list = self.updated_fin_id_list

        # Count images per FinID using the current (merged) labels.
        fin_id_counts = pd.Series(updated_fin_id_list).value_counts()

        # Map each preview row's FinID to its count
        preview_fin['count'] = preview_fin['idx'].map(
            lambda idx: int(fin_id_counts.get(updated_fin_id_list[idx], 0))
        )

        # Sort preview rows by that count
        preview_fin.sort_values(
            by='count', ascending=ascending, inplace=True, kind='mergesort'
        )

        # Update annotation with current FinID and count
        preview_fin['annotation'] = preview_fin['idx'].apply(
            lambda idx: "FinID%d count: %s\n%s" % (
                updated_fin_id_list[idx],
                int(fin_id_counts.get(updated_fin_id_list[idx], 0)),
                self.features.metadata.at[idx, 'path'],
            )
        )

        preview_fin.drop(columns='count', inplace=True)
        preview_fin.reset_index(drop=True, inplace=True)
        self.preview_fin = preview_fin

    def start_client(self, timeout=30):
        """Start the multiprocessing pipe client, retrying until ready."""
        import time
        deadline = time.time() + timeout
        last_exc = None
        while time.time() < deadline:
            try:
                self.receiver = multiprocessing.connection.Client(
                    (self.host, self.port), authkey=self.authkey
                )
                return
            except ConnectionRefusedError as exc:
                last_exc = exc
                time.sleep(0.5)
        raise RuntimeError(
            "Could not connect to ImageGridViewer GUI at %s:%d within %ds. "
            "Please start it first with: python ImageGridViewer.py"
            % (self.host, self.port, timeout)
        ) from last_exc

    def merge_loop(self):
        """
        Main interactive loop: send preview to GUI, receive merge
        decisions, and update fin IDs until the GUI disconnects.
        """
        features = self.features
        preview_fin = self.preview_fin
        updated_fin_id_list = self.updated_fin_id_list
        receiver = self.receiver

        try:
            while True:
                self.sort_preview_by_count()
                preview_fin = self.preview_fin
                preview_fin_list = preview_fin['idx'].tolist()
                preview_fin_path_list = preview_fin['path'].tolist()
                preview_fin_annotation_list = preview_fin['annotation'].tolist()

                receiver.send({
                    "idx": preview_fin_list,
                    "path": preview_fin_path_list,
                    "annotation": preview_fin_annotation_list,
                })

                user_merged_fin_idx_list = receiver.recv()
                #print(user_merged_fin_idx_list)

                if len(user_merged_fin_idx_list) > 1:
                    kept_fin_idx_list = user_merged_fin_idx_list
                    kept_fin_clearness_list = features.metadata['clearness'][kept_fin_idx_list]
                    kept_fin_idx = kept_fin_clearness_list.idxmax()
                    kept_fin_id = updated_fin_id_list[kept_fin_idx]

                    merged_fin_id_list = updated_fin_id_list[user_merged_fin_idx_list]
                    drop_mask = (
                        preview_fin['idx'].isin(user_merged_fin_idx_list)
                        & (preview_fin['idx'] != kept_fin_idx)
                    )
                    preview_fin.drop(preview_fin[drop_mask].index, inplace=True)
                    preview_fin.reset_index(drop=True, inplace=True)

                    for old_id in merged_fin_id_list:
                        if old_id != kept_fin_id:
                            updated_fin_id_list[updated_fin_id_list == old_id] = kept_fin_id
                self.preview_fin = preview_fin  
                fin_id_count = len(set(updated_fin_id_list))
                print("Cur Fin Count: %d"%fin_id_count)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            print("GUI disconnected. Exiting merge loop.")
        finally:
            receiver.close()

    def relabel_fin_ids(self):
        """
        Relabel fin IDs in ascending order by occurrence count.

        FinID 1 is assigned to the fin with the most images, FinID 2 to
        the next most frequent, and so on. Unlabeled fins keep FinID 0.
        """
        fin_id_counts = pd.Series(self.updated_fin_id_list).value_counts()
        # Drop the unlabeled FinID==0 entry if present.
        if 0 in fin_id_counts.index:
            fin_id_counts = fin_id_counts.drop(0)

        # Sort by count descending -> new ID 1 is the largest group.
        sorted_by_count = fin_id_counts.sort_values(ascending=False)

        for new_id, old_id in enumerate(sorted_by_count.index, start=1):
            self.updated_fin_id_list[
                self.updated_fin_id_list == old_id
            ] = new_id

        self.features.metadata.FinID = self.updated_fin_id_list

    def plot_statistics(self):
        """Plot and save FinID histogram."""
        plt.plot(self.features.metadata['FinID'].value_counts(), "*")
        plt.title("Fin ID Histogram")
        plt.xlabel("Fin ID")
        plt.ylabel("Count")
        plt.savefig(self.output_plot_path)
        plt.show()

    def save_results(self):
        """Save merged features and metadata CSV."""
        self.features.save(self.output_features_path)
        self.features.metadata.to_csv(self.output_csv_path)

    @staticmethod
    def creat_empty_folder(folder):
        """Create an empty folder, removing existing contents if necessary."""
        folder_path = Path(folder)
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"Deleted: {folder_path}")
        os.mkdir(folder)

    def organize_folders(self):
        """Organize fin images into folders by FinID using symlinks."""
        max_FinID = self.features.metadata["FinID"].max()
        for i in range(max_FinID + 1):
            paths = self.features.metadata.query("FinID==%d" % i)['path']
            if i == 0:
                dest_dir = os.path.join(self.fin_output_dir, "Fin_NoLabel")
            else:
                dest_dir = os.path.join(self.fin_output_dir, "FinID%03d" % i)
            self.creat_empty_folder(dest_dir)
            for fin in paths:
                fin_file = os.path.basename(fin)
                src = os.path.join(self.root_dir, fin)
                dest = os.path.join(dest_dir, fin_file)
                os.symlink(src, dest)

    def __call__(self):
        """Run the full merge-preview pipeline."""
        self.load_data()
        self.build_preview()
        self.start_client()
        self.merge_loop()
        self.relabel_fin_ids()
        self.print_fin_distribution()
        self.plot_statistics()
        self.save_results()
        self.organize_folders()


if __name__ == '__main__':
    import sys
    #root_dir = "/media/filming/2025-白海豚/20240825-JM_02-3"
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else: 
        print("No root directory is provided")
    merger = FinMergePreview(root_dir=root_dir)
    merger()
