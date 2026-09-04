#!/usr/bin/env python
# coding: utf-8
# 背鳍图片质量控制
import os
import glob
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

class FinQualityFilter:
    """Filter low-quality fin images based on clearness and crop confidence."""
    def __init__(self, root_dir, clearness_threshold=0.15, crop_conf_threshold=0.4):
        self.root_dir = root_dir
        self.clearness_threshold = clearness_threshold
        self.crop_conf_threshold = crop_conf_threshold
        self.metainfo_csv = os.path.join(root_dir, "METAINFO", "FIN_METAINFO.csv")
        self.metainfo = None

    def load_metainfo(self):
        """Load the FIN_METAINFO.csv file."""
        self.metainfo = pd.read_csv(self.metainfo_csv, index_col=0)

    def filter_fin_and_save_info(self):
        """Compute statistics after applying thresholds."""
        self.metainfo["select"] = (
            (self.metainfo.clearness > self.clearness_threshold)
            * (self.metainfo.crop_conf > self.crop_conf_threshold)
        )
        self.metainfo["quality"] = self.metainfo.clearness > self.clearness_threshold
        # Save the updated metainfo DataFrame to CSV
        self.metainfo.to_csv(self.metainfo_csv)
 
    def plot_statistics_result(self):
        """Plot and save the filter statistics result.
        TODO: show on GUI and save on local
        """
        
    def auto_filter(self):
        """Execute the full filtering pipeline."""
        self.load_metainfo()
        self.filter_fin_and_save_info()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
        filter_obj = FinQualityFilter(root_dir=root_dir)
        filter_obj.auto_filter()
    else:
        print("No root directory is provided")
