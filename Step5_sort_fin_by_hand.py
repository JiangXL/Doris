#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing.connection

from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset
from wildlife_tools.data import FeatureDataset


class FinInteractiveLabeler:
    """
    Interactive fin labeling server.

    Iterates over all features, sends similar candidates to a GUI for
    human confirmation, and updates FinID assignments and similarity
    constraints based on user feedback.
    """
    DEFAULT_HOST = 'localhost'
    DEFAULT_PORT = 1126
    DEFAULT_AUTHKEY = b'dolphin'
    DEFAULT_THRESHOLD = 0.5
    def __init__(
        self,
        root_dir,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        authkey=DEFAULT_AUTHKEY,
        threshold=DEFAULT_THRESHOLD,
        deepfeatures_path='METAINFO/FIN_DEEPFEATURES',
        similarity_path='METAINFO/FIN_SIMILARITY.npy',
        output_csv='METAINFO/FIN_METAINFO_SELECTED.csv',
    ):
        self.root_dir = root_dir
        self.host = host
        self.port = port
        self.authkey = authkey
        self.threshold = threshold
        self.deepfeatures_path = os.path.join(root_dir, deepfeatures_path)
        self.similarity_path = os.path.join(root_dir, similarity_path)
        self.output_csv = os.path.join(root_dir, output_csv)

        self.features = None
        self.metainfo = None
        self.similarity = None
        self.fin_id_list = None
        self.client = None

    def load_data(self):
        """Load deep features and similarity matrix."""
        self.features = FeatureDataset.from_file(self.deepfeatures_path)
        self.metainfo = self.features.metadata
        self.similarity = np.load(self.similarity_path)
        self.fin_id_list = self.features.metadata.FinID.values.copy()

    def start_client(self, timeout=30):
        """Start the multiprocessing pipe client, retrying until ready."""
        import time
        deadline = time.time() + timeout
        last_exc = None
        while time.time() < deadline:
            try:
                self.client = multiprocessing.connection.Client(
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

    @staticmethod
    def stats_fin_id(_fin_id_list):
        max_id = int(np.max(_fin_id_list))
        fin_counter = np.zeros(max_id + 1, dtype=np.int16)
        for i in range(len(_fin_id_list)):
            fin_id = int(_fin_id_list[i])
            fin_counter[fin_id] = fin_counter[fin_id] + 1
        return fin_counter

    def print_fin_distribution(self, _fin_id_list=None):
        if _fin_id_list is None:
            _fin_id_list = self.fin_id_list
        fin_stats = self.stats_fin_id(_fin_id_list)
        stats_text = "unclassified fin: %d\n" % (fin_stats[0])
        for i in range(1, len(fin_stats)):
            stats_text = stats_text + "fin %d: %d\n" % (i, fin_stats[i])
        print(stats_text)

    def _save_progress(self, i=None):
        """Save current state to disk."""
        self.features.metadata["FinID"] = self.fin_id_list
        self.features.save(self.deepfeatures_path)
        self.features.metadata.to_csv(self.output_csv)
        np.save(self.similarity_path, self.similarity)
        if i is not None:
            print("\nProcessing %d/%d" % (i, len(self.features)))
        print("Labeled fin: %d/%d" % (np.sum(self.fin_id_list > 0), len(self.fin_id_list)))
        print("FinID count:", len(np.unique(self.fin_id_list)) - 1)

    def interactive_label_fin(self, index):
        """
        Send similar candidates for feature `index` to GUI and receive
        user confirmation.

        Returns:
            True if FinID was updated, False if skipped, or raises
            connection errors if GUI disconnects.
        """
        metainfo = self.metainfo
        similarity = self.similarity
        fin_id_list = self.fin_id_list
        client = self.client
        threshold = self.threshold

        similar_fin_list = []
        wait_user_check_fin_list = []
        wait_user_check_fin_path_list = []
        wait_user_check_fin_annotation_list = []
        labeled_fin_list = []
        unlabeled_fin_list = []
        ref_fin_id = set()

        sorted_indices = np.flip(np.argsort(similarity[index, :]))

        for i in range(0, len(similarity)):
            query_index = sorted_indices[i]
            if similarity[index, query_index] > threshold:
                similar_fin_list.append(query_index)

        for fin in similar_fin_list:
            if fin != index:
                if fin_id_list[fin] == 0:
                    unlabeled_fin_list.append(fin)
                else:
                    labeled_fin_list.append(fin)

        labeled_fin_ref_list = []
        for fin in labeled_fin_list:
            fin_id = fin_id_list[fin]
            if fin_id not in ref_fin_id:
                ref_fin_id.add(fin_id)
                labeled_fin_ref_list.append(fin)

        wait_user_check_fin_list.append(index)
        wait_user_check_fin_annotation_list.append(
            "Reference Image: " + metainfo.path[index][4:]
        )
        wait_user_check_fin_path_list.append(
            os.path.join(self.root_dir, metainfo.path[index])
        )

        for fin in labeled_fin_ref_list:
            fin_id = fin_id_list[fin]
            if fin_id != fin_id_list[index]:
                wait_user_check_fin_list.append(fin)
                wait_user_check_fin_annotation_list.append(
                    "Labeled Image: %s\nSimilarity: %0.3f"
                    % (metainfo.path[fin][4:], similarity[index, fin])
                )
                wait_user_check_fin_path_list.append(
                    os.path.join(self.root_dir, metainfo.path[fin])
                )

        for fin in unlabeled_fin_list:
            wait_user_check_fin_list.append(fin)
            wait_user_check_fin_annotation_list.append(
                "Unlabeled Image: %s\nSimilarity: %0.3f"
                % (metainfo.path[fin][4:], similarity[index, fin])
            )
            wait_user_check_fin_path_list.append(
                os.path.join(self.root_dir, metainfo.path[fin])
            )

        # Skip checking if no new candidate
        if len(wait_user_check_fin_list) < 2:
            return False
        elif len(wait_user_check_fin_list) == 2:
            if fin_id_list[index] in ref_fin_id:
                return False

        # Send list to GUI
        client.send({
            "idx": wait_user_check_fin_list,
            "path": wait_user_check_fin_path_list,
            "annotation": wait_user_check_fin_annotation_list,
        })
        #TODO: add shoot group to annotation

        # Receive result from GUI
        user_confirmed_fin_list = client.recv()
        print("labeled fin idx:", labeled_fin_ref_list)
        print("user confirmed: %d/%d" % (len(user_confirmed_fin_list), len(wait_user_check_fin_list)))

        same_fin_list = user_confirmed_fin_list

        # Assign new fin label
        existed_id = []
        for fin in same_fin_list:
            fin_id = fin_id_list[fin]
            if fin_id != 0 and fin_id not in existed_id:
                existed_id.append(fin_id)

        if len(existed_id) == 0:
            prev_fin_id = np.max(fin_id_list)
            for i in range(1, prev_fin_id + 2):
                if i not in fin_id_list:
                    cur_fin_id = i
                    break
        elif len(existed_id) == 1:
            cur_fin_id = existed_id[0]
        else:
            cur_fin_id = np.min(existed_id)
            existed_id.remove(cur_fin_id)
            for fin_id in existed_id:
                fin_id_list[fin_id_list == fin_id] = cur_fin_id

        for fin in same_fin_list:
            fin_id_list[fin] = cur_fin_id

        # Propagate yes/no in similarity matrix
        fin_has_same_cur_id_list = [
            fin for fin in range(len(fin_id_list))
            if fin_id_list[fin] == cur_fin_id
        ]
        for extend_fin in fin_has_same_cur_id_list:
            similarity[extend_fin, fin_has_same_cur_id_list] = 1

        for waiting_fin in wait_user_check_fin_list:
            if waiting_fin not in same_fin_list:
                excluded_fin = waiting_fin
                excluded_fin_id = fin_id_list[excluded_fin]
                if excluded_fin_id == 0:
                    fin_has_same_excluded_id_list = [excluded_fin]
                else:
                    fin_has_same_excluded_id_list = [
                        fin_ for fin_ in range(len(fin_id_list))
                        if fin_id_list[fin_] == excluded_fin_id
                    ]
                for extend_excluded_fin in fin_has_same_excluded_id_list:
                    for extend_fin in fin_has_same_cur_id_list:
                        similarity[extend_fin, extend_excluded_fin] = 0
                        similarity[extend_excluded_fin, extend_fin] = 0

        return True

    def run(self):
        """Execute the full interactive labeling pipeline."""
        self.load_data()
        self.start_client()

        try:
            print("Please select images with same fin of reference image")
            for i in range(len(self.features)):
                try:
                    updated = self.interactive_label_fin(i)
                    if updated:
                        self._save_progress(i)
                except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                    print("\nGUI disconnected (%s). Saving progress and exiting." % type(e).__name__)
                    self._save_progress(i if i > 0 else None)
                    break
        finally:
            if self.client is not None:
                self.client.close()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        print("Usage: python Step5_sort_fin_by_hand.py <root_dir>")
        sys.exit(1)

    labeler = FinInteractiveLabeler(root_dir=root_dir)
    labeler.run()
