#!/usr/bin/env python
# coding: utf-8
# 移除低质量的背鳍图片
import os
import glob
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

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
        # Save the updated metainfo DataFrame to CSV
        self.metainfo.to_csv(self.metainfo_csv)

    def plot_statistics_result(self):
        """Plot and save the filter statistics result."""
        total_fin_num = len(self.metainfo)
        selected_clearness_num = int(np.sum(self.metainfo.clearness > self.clearness_threshold))
        selected_crop_conf_num = int(np.sum(self.metainfo.crop_conf > self.crop_conf_threshold))

        clearness_annotation = "Clearness > %0.3f \n %d/%d" % (
            self.clearness_threshold,
            selected_clearness_num,
            total_fin_num,
        )
        crop_conf_annotation = "Crop confidence > %0.3f \n %d/%d" % (
            self.crop_conf_threshold,
            selected_crop_conf_num,
            total_fin_num,
        )
        final_selected_num = int(np.sum(self.metainfo.select))
        final_selected_annotation = "Final selected: %d/%d" % (
            final_selected_num,
            total_fin_num,
        )

        plt.subplot(211)
        plt.hist(self.metainfo.clearness, bins=128)
        plt.xlabel("Fin Image Clearness")
        plt.ylabel("Fin Image Number")
        plt.plot([self.clearness_threshold, self.clearness_threshold], [500, 0])
        plt.annotate(clearness_annotation, [self.clearness_threshold, 250])
        plt.subplot(212)
        plt.hist(self.metainfo.crop_conf, bins=128)
        plt.xlabel("Fin Crop Confidence")
        plt.xlim(0, 1)
        plt.annotate(crop_conf_annotation, [0.5, 100])
        plt.annotate(final_selected_annotation, [0.1, 50])
        plt.plot([self.crop_conf_threshold, self.crop_conf_threshold], [150, 0])
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_dir, "METAINFO", "FIN_Filter.png"))
        #plt.show()

    @staticmethod
    def create_empty_folder(folder):
        """Create an empty folder, removing existing contents if necessary."""
        folder_path = Path(folder)
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"Deleted: {folder_path}")
        os.mkdir(folder)

    def _symlink_fins(self, dest_dir, query_str):
        """Helper to create symlinks for fins matching a query."""
        self.create_empty_folder(dest_dir)
        for fin in self.metainfo.query(query_str)["path"]:
            src = os.path.join(self.root_dir, fin)
            dest = os.path.join(dest_dir, fin[4:])
            os.symlink(src, dest)

    def create_category_folders(self):
        """Create categorized folders (BLUR, LOW_CROP_CONFIDENCE, SELECTED) with symlinks."""
        # Soft link fin files
        dest_dir = os.path.join(self.root_dir, "FIN", "BLUR")
        self._symlink_fins(dest_dir, "clearness<%f" % self.clearness_threshold)

        dest_dir = os.path.join(self.root_dir, "FIN", "LOW_CROP_CONFIDENCE")
        self._symlink_fins(dest_dir, "crop_conf<%f" % self.crop_conf_threshold)

        dest_dir = os.path.join(self.root_dir, "FIN", "SELECTED")
        self._symlink_fins(dest_dir, "select==True")

    def scan_user_confirmation(self):
        """Scan the SELECTED folder to receive user confirmation (manual moves)."""
        scan_dir = os.path.join(self.root_dir, "FIN", "SELECTED")
        scan_list = sorted(os.listdir(scan_dir))
        scanned_img_list = glob.glob(os.path.join(scan_dir, "*.JPG"))
        scanned_img_list = [os.path.basename(path) for path in scanned_img_list]

        # Confirm fin images moved into SELECTED folder
        for img_name in scanned_img_list:
            current_select = self.metainfo.loc[self.metainfo["path"] == "FIN/" + img_name, "select"]
            if current_select.values[0] == False:
                print("User move", img_name, "into SELECTED folder")
                self.metainfo.loc[self.metainfo["path"] == "FIN/" + img_name, "select"] = True

        # Confirm fin images moved out of SELECTED folder
        for path in self.metainfo.loc[self.metainfo["select"] == True, "path"]:
            img_name = os.path.basename(path)
            if img_name not in scanned_img_list:
                print("User move", img_name, "out of SELECTED folder")
                self.metainfo.loc[self.metainfo["path"] == path, "select"] = False

        self.metainfo.to_csv(self.metainfo_csv)

    def link_low_quality_originals(self):
        """Create symlinks for original images that have no selected fins."""
        lowquality_img = [
            i
            for i in self.metainfo.orig_img.unique()
            if i not in self.metainfo.loc[self.metainfo["select"] == True, "orig_img"].unique()
        ]
        print("Total low quality images number:", len(lowquality_img))
        for img in tqdm(lowquality_img):
            src = os.path.join(self.root_dir, img)
            dest_dir = os.path.join(self.root_dir, "Quality below 60")
            dest = os.path.join(dest_dir, img)
            if not Path(dest_dir).exists():
                os.mkdir(dest_dir)
            if not os.path.exists(dest):
                os.symlink(src, dest)

    def auto_filter(self):
        """Execute the full filtering pipeline."""
        self.load_metainfo()
        self.filter_fin_and_save_info()
        self.plot_statistics_result()
        self.create_category_folders()

    def confirm_filter(self):
        self.scan_user_confirmation()
        self.link_low_quality_originals()

if __name__ == "__main__":
    import sys
    #root_dir = "/media/filming/2025-白海豚/20240825-JM_02-3//"
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
    filter_obj = FinQualityFilter(root_dir=root_dir)
    filter_obj.auto_filter()
    filter_obj.confirm_filter()
