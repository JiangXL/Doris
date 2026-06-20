#!/usr/bin/env python
# coding: utf-8
# Match Left side and Right side of fin
# 在制作 preview slide 时，匹配并移动 Fin
# 根据 Fin 像素大小排序，选像素最高者

import argparse
import cv2
import glob
import numpy as np
import os
import shutil
from pathlib import Path
from matplotlib import pyplot as plt

from wildlife_tools.data import FeatureDataset


def create_empty_folder(folder):
    folder_path = Path(folder)
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"Deleted: {folder_path}")
    os.mkdir(folder)


def pair_fin(root_dir):
    """Pair fins, assign dolphin IDs, and organize social relationship folders."""
    root_dir = str(root_dir)
    features = FeatureDataset.from_file(
        os.path.join(root_dir, "METAINFO", "FIN_DEEPFEATURES_MERGED")
    )

    # Scan the dolphin id from folder structure DphID->FinID->FinImageID
    print("Scanning dolphin id from folder")
    features.metadata["DphID"] = 0
    features.metadata["FinID2"] = 0
    dolphin_path = os.path.join(glob.escape(root_dir), "FIN", "DphID*")
    dolphin_list = glob.glob(dolphin_path)
    dolphin_list.sort()
    for dolphin in dolphin_list:
        dolphin_id = int(os.path.basename(dolphin)[5:])  # DphIDxxx
        fin_list = glob.glob(os.path.join(glob.escape(dolphin), "FinID*"))
        for fin in fin_list:
            fin_id = int(os.path.basename(fin)[5:])  # FinIDxxx
            fin_image_list = glob.glob(os.path.join(glob.escape(fin), "*.JPG"))
            for fin_image in fin_image_list:
                fin_image_name = "FIN/" + os.path.basename(fin_image)
                features.metadata.loc[
                    features.metadata["path"] == fin_image_name, "FinID2"
                ] = fin_id
                features.metadata.loc[
                    features.metadata["path"] == fin_image_name, "DphID"
                ] = dolphin_id

    features.metadata.to_csv(
        os.path.join(root_dir, "METAINFO", "FIN_METAINFO_SELECTED_MERGED_PAIRED.csv")
    )
    features.save(
        os.path.join(root_dir, "METAINFO", "FIN_DEEPFEATURES_SELECTED_MERGED_PAIRED")
    )

    # Link original image to subfolder
    print("Linking orignal image to dolphin folder")
    DphID_list = features.metadata["DphID"].unique()
    for DphID in DphID_list:
        paths = features.metadata.query("DphID==%d" % DphID)["orig_img"]
        if DphID != 0:
            dest_dir = os.path.join(root_dir, "DphID%03d" % DphID)
            create_empty_folder(dest_dir)
        else:
            dest_dir = os.path.join(root_dir, "Quality below 60")
            if not Path(dest_dir).exists():
                os.mkdir(dest_dir)
        for dolphin in paths:
            src = os.path.join(root_dir, dolphin)
            dest = os.path.join(dest_dir, dolphin)
            if not os.path.exists(dest):
                os.symlink(src, dest)
    
    ## Find NN
    # Link all nearby dolphins to NN folder automatic
    print("Finding All Near Neighbours")
    dolphin_count_in_img = features.metadata.orig_img[
        features.metadata.DphID != 0
    ].value_counts()
    # exclude unclassied dolphin
    create_empty_folder(os.path.join(root_dir, "NN"))
    create_empty_folder(os.path.join(root_dir, "MCP"))
    create_empty_folder(os.path.join(root_dir, "SYN"))
    for i in range(len(dolphin_count_in_img)):
        img_name = dolphin_count_in_img.index[i]
        dolphin_count = dolphin_count_in_img[img_name]
        if dolphin_count > 1:
            dolphins_in_same_img = features.metadata.query(
                "orig_img=='%s'" % (img_name)
            )
            NNGroupName = ""
            dolphin_id_list = dolphins_in_same_img.DphID.sort_values().values
            for dolphin_id in dolphin_id_list:
                NNGroupName = NNGroupName + "_DphID%03d" % dolphin_id
            src = os.path.join(root_dir, img_name)
            dest_dir = os.path.join(root_dir, "NN", NNGroupName[1:])
            dest = os.path.join(dest_dir, img_name)
            if not Path(dest_dir).exists():
                os.mkdir(dest_dir)
            if not os.path.exists(dest):
                os.symlink(src, dest)

    # generate automatic NN ID
    features.metadata["automatic_group"] = ""
    dolphin_count_in_img = features.metadata.orig_img[
        features.metadata.DphID != 0
    ].value_counts()
    # exclude unclassied dolphin
    NN_group_dict = dict()
    for i in range(len(dolphin_count_in_img)):
        img_name = dolphin_count_in_img.index[i]
        dolphin_count = dolphin_count_in_img[img_name]
        if dolphin_count > 1:
            dolphins_in_same_img = features.metadata.query(
                "orig_img=='%s'" % (img_name)
            )
            NNGroupName = ""
            dolphin_id_list = dolphins_in_same_img.DphID.sort_values().values
            for dolphin_id in dolphin_id_list:
                NNGroupName = NNGroupName + "_DphID%03d" % dolphin_id
            NNGroupName = NNGroupName[1:]
            if NNGroupName not in NN_group_dict:
                NN_group_dict[NNGroupName] = []
            NN_group_dict[NNGroupName].append(img_name)

    group_idx = 1
    for key in NN_group_dict.keys():
        group_name = "NN%02d" % group_idx
        for img in NN_group_dict[key]:
            features.metadata.loc[
                features.metadata["orig_img"] == img, "automatic_group"
            ] = group_name
        group_idx = group_idx + 1

    # Scan folder structure to obtain confirmed social relationship
    ret = input("Please move MCP, SYN outof NN folder, type Y to continue: ")
    features.metadata["confirmed_group"] = ""

    # find the dolphin id from folder structure
    MCP_dir = os.path.join(root_dir, "MCP")
    if os.path.isdir(MCP_dir):
        MCP_list = os.listdir(MCP_dir)
        MCP_list.sort()
        MCP_idx = 1
        for MCP in MCP_list:
            img_list = glob.glob(os.path.join(MCP_dir, MCP, "*.JPG"))
            MCP_name = "MCP%02d" % (MCP_idx)
            for img in img_list:
                img_name = os.path.basename(img)
                features.metadata.loc[
                    features.metadata["orig_img"] == img_name, "confirmed_group"
                ] = MCP_name
            MCP_idx = MCP_idx + 1

    features.metadata.to_csv(
        os.path.join(
            root_dir, "METAINFO", "FIN_METAINFO_SELECTED_MERGED_PAIRED_SOCIAL.csv"
        )
    )
    features.save(
        os.path.join(
            root_dir, "METAINFO", "FIN_DEEPPFEATUES_SELECTED_MERGED_PAIRED_SOCIAL"
        )
    )

    # find the dolphin id from folder structure
    NN_dir = os.path.join(root_dir, "NN")
    if os.path.isdir(NN_dir):
        NN_list = os.listdir(NN_dir)
        NN_list.sort()
        NN_idx = 1
        for NN in NN_list:
            img_list = glob.glob(os.path.join(NN_dir, NN, "*.JPG"))
            NN_name = "NN%02d" % (NN_idx)
            for img in img_list:
                img_name = os.path.basename(img)
                features.metadata.loc[
                    features.metadata["orig_img"] == img_name, "confirmed_group"
                ] = NN_name
            NN_idx = NN_idx + 1

    features.metadata.to_csv(
        os.path.join(
            root_dir, "METAINFO", "FIN_METAINFO_SELECTED_MERGED_PAIRED_SOCIAL.csv"
        )
    )
    features.save(
        os.path.join(
            root_dir, "METAINFO", "FIN_DEEPPFEATUES_SELECTED_MERGED_PAIRED_SOCIAL"
        )
    )

    # find the dolphin id from folder structure
    SYN_dir = os.path.join(root_dir, "SYN")
    if os.path.isdir(SYN_dir):
        SYN_list = os.listdir(SYN_dir)
        SYN_list.sort()
        SYN_idx = 1
        for SYN in SYN_list:
            img_list = glob.glob(os.path.join(SYN_dir, SYN, "*.JPG"))
            SYN_name = "SYN%02d" % (SYN_idx)
            for img in img_list:
                img_name = os.path.basename(img)
                features.metadata.loc[
                    features.metadata["orig_img"] == img_name, "confirmed_group"
                ] = SYN_name
            SYN_idx = SYN_idx + 1

    features.metadata.to_csv(
        os.path.join(
            root_dir, "METAINFO", "FIN_METAINFO_SELECTED_MERGED_PAIRED_SOCIAL.csv"
        )
    )
    features.save(
        os.path.join(
            root_dir, "METAINFO", "FIN_DEEPPFEATUES_SELECTED_MERGED_PAIRED_SOCIAL"
        )
    )

    unclassied_img_list = []
    orig_img_has_DphID_list = features.metadata.loc[
        features.metadata["DphID"] != 0, "orig_img"
    ].unique()
    for i in features.metadata.orig_img.unique():
        if i not in orig_img_has_DphID_list:
            unclassied_img_list.append(i)

    ret = features.metadata.loc[features.metadata["DphID"] != 0, "orig_img"].unique()

    # Move unclassied original image to Quality below 60
    unclassied_img_list = [
        i
        for i in features.metadata.orig_img.unique()
        if i not in features.metadata.loc[features.metadata["DphID"] != 0, "orig_img"].unique()
    ]
    for img in unclassied_img_list:
        src = os.path.join(root_dir, img)
        dest_dir = os.path.join(root_dir, "Quality below 60")
        dest = os.path.join(dest_dir, img)
        if not Path(dest_dir).exists():
            os.mkdir(dest_dir)
        if not os.path.exists(dest):
            os.symlink(src, dest)

    # Convert softlink to regular file
    # ```bash
    # for f in $(find MCP/ -type l);do cp --remove-destination $(readlink $f) $f;done;
    # ```
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair fin features and organize social relationship folders."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="/media/filming/2025-白海豚/20240825-JM_02-3/",
        help="Root directory of the dataset",
    )
    args = parser.parse_args()
    pair_fin(args.root_dir)
