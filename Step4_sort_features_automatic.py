#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing.connection

from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.data import ImageDataset
from wildlife_tools.data import FeatureDataset

root_dir = r'/media/filming/2025-白海豚/20240825-JM_02-3/'
METAINFO_csv = root_dir + "/METAINFO/FIN_METAINFO.csv"
FIN_DEEPFEATURES = root_dir + "/METAINFO/FIN_DEEPFEATURES"
full_metainfo = pd.read_csv(METAINFO_csv, index_col=0)

features = FeatureDataset.from_file(FIN_DEEPFEATURES) 
matcher = CosineSimilarity()
similarity = matcher(features, features)

def stats_fin_id(_fin_id_list):
    max_id = int(np.max(_fin_id_list))
    fin_counter = np.zeros(max_id+1, dtype=np.int16)
    # fin_id count from 1, 0 maens unclassfied fins
    for i in range(len(_fin_id_list)):
        fin_id = int(_fin_id_list[i])
        fin_counter[fin_id ]  = fin_counter[fin_id] + 1
    return fin_counter

def imshow_fin_image_from_label(label):
    indices = np.where(fin_id_list==label)[0]
    file_list = []
    annotation_list = []
    for index in indices:
        file_list.append(features.metadata.loc[index, 'path'])
        annotation_list.append(features.metadata.loc[index, 'identity'] 
                                       + ' ' + "%d"%index
                                       + ' ' + features.metadata.loc[index, 'path'][-14:])
    image_grid.image_grid(file_list, top=100, text=annotation_list)


def plot_fin_distribution(_fin_id_list):
    plt.figure()
    plt.plot(fin_id_list, ".")
    fin_stats = stats_fin_id(_fin_id_list)
    stats_text = "unclassified fin: %d\n"%(fin_stats[0])
    for i in range(1, len(fin_stats)):
        stats_text = stats_text + "fin %d: %d\n"%(i, fin_stats[i])
    plt.text(34000, 1, stats_text)

def print_fin_distribution(_fin_id_list):
    fin_stats = stats_fin_id(_fin_id_list)
    stats_text = "unclassified fin: %d\n"%(fin_stats[0])
    for i in range(1, len(fin_stats)):
        stats_text = stats_text + "fin %d: %d\n"%(i, fin_stats[i])
    print(stats_text)

# fin id/label count from 1
def automatic_link_fin(similarity, index, fin_id_list, threshold=0.9):
    high_similar_fin_list = []
    labeled_fin_list = []
    unlabeled_fin_list = []

    for i in range(0, len(similarity)):
        if  similarity[index, i] > threshold:
            high_similar_fin_list.append(i)
    if len(high_similar_fin_list) == 1:
        return fin_id_list

    existed_id = []
    for i in high_similar_fin_list:
        fin_id = fin_id_list[i]
        if not (fin_id == 0):
            if not (fin_id in existed_id):
                existed_id.append(fin_id)
    #print("existed_id", existed_id)
    # label fin
    if len(existed_id) == 0 :
        prev_fin_id = np.max(fin_id_list)
        for i in range(1, prev_fin_id+2):
            if not (i in fin_id_list):
                cur_fin_id = i
                break;
    elif(len(existed_id) == 1):
        cur_fin_id = existed_id[0]
    else:
        # replace old id
        cur_fin_id = np.min(existed_id)
        existed_id.remove(cur_fin_id)
        for fin_id in existed_id:
            fin_id_list[fin_id_list == fin_id] = cur_fin_id
    for i in high_similar_fin_list :
        fin_id_list[i] = cur_fin_id
    #print("cur_fin_id: ", cur_fin_id)
    #print("labeled fin: %d/%d"%(np.sum(fin_id_list >0), len(fin_id_list)))
    return fin_id_list

fin_id_list = np.zeros(len(features), dtype=np.int32)
for i in range(len(similarity)):
     fin_id_list = automatic_link_fin(similarity, i, fin_id_list, 0.80)
print("Unclassified fin number:", np.sum(fin_id_list == 0), "Total fin image number: ", len(features))
print("FinID number:", len(np.unique(fin_id_list))-1)


features.metadata["FinID"] = fin_id_list
features.save(FIN_DEEPFEATURES)
features.metadata.to_csv(root_dir + "/METAINFO/FIN_METAINFO_SELECTED.csv")

# calculate the cosine similarity and set similarity of same fin to 1
for fin_id in np.unique(fin_id_list[fin_id_list!=0]):
    fin_has_same_id_list = []
    for fin_idx in range(len(fin_id_list)):
        if fin_id_list[fin_idx] == fin_id:
            fin_has_same_id_list.append(fin_idx)
    for fin in fin_has_same_id_list:
        similarity[fin, fin_has_same_id_list] = 1
print("Connected node number:",np.sum(similarity == 1))

# 排除同一原始图里同一个背鳍出现多次
ori_image_list = features.metadata.orig_img.unique()
occurred_number = 0
for image in ori_image_list:
    fin_idx_list = features.metadata.index[features.metadata.orig_img == image].values
    fin_number = len(fin_idx_list) 
    if fin_number > 1:
        #print("Found %s has %d fins"%(image, fin_number))
        occurred_number = occurred_number + 1
        for i in range(fin_number):
            fin_idx_i = fin_id_list[i]
            for j in range(i+1, fin_number):
                fin_idx_j = fin_id_list[j]
                similarity[fin_idx_i, fin_idx_j] = 0
                similarity[fin_idx_j, fin_idx_i] = 0
print("Found %s images have multiple fins"%(occurred_number))


np.save(root_dir+"/METAINFO/FIN_SIMILARITY.npy", similarity)
#similarity = np.load(root_dir + "FIN_SIMILARITY.npy")

