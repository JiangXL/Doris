#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from blur_detector_torch import BlurDetector
from matplotlib import pyplot as plt


blur_detector = BlurDetector("models/blur_detection_resnet101_final.pth")

root_dir = "/media/filming/2025-白海豚/20240825-JM_02-2/"
dataset_name = root_dir.split("/")[-2]
METAINFO_CSV = root_dir + "/METAINFO/FIN_METAINFO.csv"


fin_dataset = pd.read_csv(METAINFO_CSV, index_col=0)


# test on one fin image
ret = blur_detector.predict(root_dir + "/" + fin_dataset.path[1])
print(ret)

clearness_list = []
for i in tqdm(fin_dataset.path):
    ret = blur_detector.predict( root_dir + "/" + i)
    clearness_list.append( ret['probabilities']['clear'] )
    fin_dataset["clearness"] = clearness_list
    fin_dataset.to_csv(METAINFO_CSV))

plt.subplot(2, 1, 1)
plt.title("Fin Clearness-"+dataset_name)
plt.plot(fin_dataset.clearness, "*")
plt.ylabel("Clearness")
plt.subplot(2, 1, 2)
plt.hist(fin_dataset.clearness, bins=256)
plt.xlabel("Clearness")
plt.savefig(root_dir + "/METAINFO/FinClearness.png")
plt.show()
