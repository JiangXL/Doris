import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing.connection

from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.data import ImageDataset
from wildlife_tools.data import FeatureDataset

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
def interactive_label_fin_2(features, similarity, index, threshold, fin_id_list):
    similar_fin_list = []
    wait_user_check_fin_list = []
    wait_user_check_fin_path_list = []
    labeled_fin_list = []
    unlabeled_fin_list = []
    user_confirmed_fin_list  = []
    same_fin_list = []
    file_list = []
    annotation_list = []

    sorted_indices = np.flip(np.argsort(similarity[index, :]))
    
    for i in range(0, len(similarity)):
        query_index = sorted_indices[i]
        if  similarity[index, query_index] > threshold:
            similar_fin_list.append(query_index)
    
    # Search whether fin has been labeled
    wait_user_check_fin_list.append(index) # add the current index as reference
    for i in similar_fin_list :
        if not (i==index) :
            if fin_id_list[i] == 0 :
                unlabeled_fin_list.append(i)
            else :
                labeled_fin_list.append(i) 

    # append fin image have been labeled, todo: the most similari one
    labeled_fin_ref_list = []
    ref_fin_id = set()
    for i in labeled_fin_list:
        fin_id = fin_id_list[i]
        if not (fin_id in ref_fin_id):
            ref_fin_id.add(fin_id)
            labeled_fin_ref_list.append(i)
    print("labeled ref: ", labeled_fin_ref_list)
    wait_user_check_fin_list = wait_user_check_fin_list + labeled_fin_ref_list + unlabeled_fin_list
    for i in wait_user_check_fin_list:
        wait_user_check_fin_path_list.append(features.metadata.loc[i, 'path'])
    # Send to GUI
    if len(wait_user_check_fin_list) < 2:
        # no new candidate
        print("no new candidate")
        return 
    elif len(wait_user_check_fin_list) == 2:
        # if current fin id is same with ref fin id
        if fin_id_list[index] in ref_fin_id:
            print("no new candidate")
            return
    receiver.send({"id": wait_user_check_fin_list, 
                   "path": wait_user_check_fin_path_list,
                  "similarity": similarity[index, wait_user_check_fin_list]})
    # Receive from GUI
    user_confirmed_fin_list = receiver.recv()
    print("user confirmed: %d/%d"%(len(user_confirmed_fin_list), len(wait_user_check_fin_list)))
    same_fin_list =  user_confirmed_fin_list 

    existed_id = []
    for i in same_fin_list:
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
    for i in same_fin_list :
        fin_id_list[i] = cur_fin_id
    print("cur_fin_id: ", cur_fin_id)
    print("labeled fin: %d/%d"%(np.sum(fin_id_list >0), len(fin_id_list)))

if __name__ == "__main__":

    features = FeatureDataset.from_file(sys.argv[1])
    root_dir = os.path.split(sys.argv[1])[0]
    features.metadata.path = root_dir + "/" +  features.metadata.path
    matcher = CosineSimilarity()
    similarity = matcher(features, features)

    with multiprocessing.connection.Listener(
            ('localhost', 1126), authkey=b'dolphin') as checkbox_server:
        with checkbox_server.accept() as receiver :
            fin_id_list = np.zeros(len(features), dtype=np.int32)
            for i in range(len(features)):
                print("%d/%d"%(i, len(features)))
                interactive_label_fin_2(features, similarity, i, 0.75, fin_id_list)
                print_fin_distribution(fin_id_list)
            np.save(sys.argv[1]+"_fin_id.npz", fin_id_list)
