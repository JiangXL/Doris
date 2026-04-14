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

def print_fin_distribution(_fin_id_list):
    fin_stats = stats_fin_id(_fin_id_list)
    stats_text = "unclassified fin: %d\n"%(fin_stats[0])
    for i in range(1, len(fin_stats)):
        stats_text = stats_text + "fin %d: %d\n"%(i, fin_stats[i])
    print(stats_text)

# fin id/label count from 1
def interactive_label_fin_2(root_dir, metainfo, similarity, index, threshold, fin_id_list):
    similar_fin_list = []
    wait_user_check_fin_list = []
    wait_user_check_fin_path_list = []
    wait_user_check_fin_annotation_list = []
    labeled_fin_list = []
    unlabeled_fin_list = []
    user_confirmed_fin_list  = []
    same_fin_list = []
    file_list = []

    sorted_indices = np.flip(np.argsort(similarity[index, :]))
    
    for i in range(0, len(similarity)):
        query_index = sorted_indices[i]
        if  similarity[index, query_index] > threshold:
            similar_fin_list.append(query_index)
    
    # Search fin has been labeled
    for fin in similar_fin_list:
        if not (fin==index) : # why exclue current fin
            if fin_id_list[fin] == 0 :
                unlabeled_fin_list.append(fin)
            else :
                labeled_fin_list.append(fin) 

    # Find the unique id in labeled fin list
    labeled_fin_ref_list = []
    ref_fin_id = set()
    for fin in labeled_fin_list:
        fin_id = fin_id_list[fin]
        if not (fin_id in ref_fin_id):
            ref_fin_id.add(fin_id)
            labeled_fin_ref_list.append(fin)
    
    # Generate waiting list = [index] + labeled_fin_ref_list + unlabeled_fin_list:
    wait_user_check_fin_list.append(index) # add the current index as reference
    wait_user_check_fin_annotation_list.append("Reference Image: " + metainfo.path[index][4:])
    wait_user_check_fin_path_list.append(root_dir + "/" +metainfo.path[index])
    for fin in labeled_fin_ref_list:
        fin_id = fin_id_list[fin]
        if fin_id != fin_id_list[index]:
            wait_user_check_fin_list.append(fin)
            wait_user_check_fin_annotation_list.append(
                "Labeled Image: %s\nSimilarity: %0.3f"%(metainfo.path[fin][4:], similarity[index, fin]))
            wait_user_check_fin_path_list.append(root_dir + "/" +  metainfo.path[fin])
    for fin in unlabeled_fin_list:
        wait_user_check_fin_list.append(fin)
        wait_user_check_fin_annotation_list.append(
            "Unlabeled Image: %s\nSimilarity: %0.3f"%(metainfo.path[fin][4:], similarity[index, fin]))
        wait_user_check_fin_path_list.append(root_dir + "/" +  metainfo.path[fin])
    # Skip checking if no new candidate
    if len(wait_user_check_fin_list) < 2:
        #print("no new candidate")
        return [0, fin_id_list, similarity]
    elif len(wait_user_check_fin_list) == 2:
        # if current fin id is same with ref fin id
        if fin_id_list[index] in ref_fin_id:
            #print("no new candidate")
            return [0, fin_id_list, similarity]
    # Send list to GUI
    receiver.send({"idx": wait_user_check_fin_list, 
                   "path": wait_user_check_fin_path_list,
                  "annotation": wait_user_check_fin_annotation_list})
    # Receive result from GUI
    user_confirmed_fin_list = receiver.recv()
    print("labeled fin idx:", labeled_fin_ref_list)
    print("user confirmed: %d/%d"%(len(user_confirmed_fin_list), len(wait_user_check_fin_list)))
    same_fin_list =  user_confirmed_fin_list 
    ## Assign new fin label
    existed_id = []
    for fin in same_fin_list:
        fin_id = fin_id_list[fin]
        if not (fin_id == 0):
            if not (fin_id in existed_id):
                existed_id.append(fin_id)
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
    for fin in same_fin_list :
        fin_id_list[fin] = cur_fin_id
    # Apply yes/no in similarity list to avoid furture confirmation
    # TODO: propagate to already confirmed label
    # 1. Propagate the fin has cur_fin_id to 1
    fin_has_same_cur_id_list = []
    # search all fin idx have this id
    for fin in range(len(fin_id_list)):
        if fin_id_list[fin] == cur_fin_id:
            fin_has_same_cur_id_list.append(fin) 
    # change it on row one by one
    for extend_fin in fin_has_same_cur_id_list:
        similarity[extend_fin, fin_has_same_cur_id_list] = 1
    # 2. Propagate the fin is/are excluded to 0
    for waiting_fin in wait_user_check_fin_list:
        if not (waiting_fin in same_fin_list):
            excluded_fin = waiting_fin
            excluded_fin_id = fin_id_list[excluded_fin]
            if excluded_fin_id == 0:
                fin_has_same_excluded_id_list = [excluded_fin]
            else: # if excluded fin has id
                fin_has_same_excluded_id_list = []
                for fin_ in range(len(fin_id_list)):
                    if fin_id_list[fin_] == excluded_fin_id:
                        fin_has_same_excluded_id_list.append(fin_) 
            for extend_excluded_fin in fin_has_same_excluded_id_list:
                for extend_fin in fin_has_same_cur_id_list:
                    similarity[extend_fin, extend_excluded_fin] = 0
                    similarity[extend_excluded_fin, extend_fin] = 0
    return [1, fin_id_list, similarity]
    print("cur_fin_id: ", cur_fin_id)

if __name__ == "__main__":
    features = FeatureDataset.from_file(sys.argv[1])
    root_dir = os.path.split(sys.argv[1])[0]
    metainfo = features.metadata
    similarity = np.load(root_dir + "/FIN_SIMILARITY.npy")
    threshold = 0.65

    with multiprocessing.connection.Listener(
            ('localhost', 1126), authkey=b'dolphin') as checkbox_server:
        with checkbox_server.accept() as receiver :
            fin_id_list = features.metadata.FinID.values
            #fin_id_list = np.zeros(len(features), dtype=np.int32)
            print("Please select images with same fin of reference image")
            for i in range(len(features)):
                update, fin_id_list, similarity = interactive_label_fin_2(
                        root_dir, metainfo, similarity, i, threshold, fin_id_list)
                if update == 1:
                    print("\nProcessing %d/%d"%(i, len(features)))
                    features.metadata["FinID"] = fin_id_list
                    features.save(sys.argv[1])
                    features.metadata.to_csv(root_dir + "/FIN_METAINFO_SELECTED.csv")
                    np.save(root_dir+"/FIN_SIMILARITY", similarity)
                    print("Labeled fin: %d/%d"%(np.sum(fin_id_list >0), len(fin_id_list)))
                    print("FinID count:", len(np.unique(fin_id_list))-1)
                    #print_fin_distribution(fin_id_list)
